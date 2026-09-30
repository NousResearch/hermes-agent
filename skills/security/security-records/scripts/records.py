#!/usr/bin/env python3
"""Stateless Mithril security record checks.

Ports the API evaluators for CMDB structure, attachment-name signals, and
NIST CSF 2.0 fit. Nothing is stored, scanned, or sent.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

DATA = Path(__file__).resolve().parent.parent / "data"
ASSET_ID = re.compile(r"^[A-Za-z0-9._:/-]{1,120}$")
MAX_ASSETS = 5000
MAX_DEPENDENCIES = 20000
MAX_MESSAGES = 500

ARCHIVE = {"zip", "7z", "rar", "tar", "gz"}
EXECUTABLE = {"exe", "scr", "js", "vbs", "bat", "cmd", "msi", "jar", "ps1", "lnk", "com", "hta"}
MACRO = {"docm", "xlsm", "pptm", "dotm", "xlsb"}
DECOY = {"pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "txt", "jpg", "jpeg", "png", "gif"}

FRAMEWORK = {
    "id": "nist-csf-20",
    "name": "NIST Cybersecurity Framework 2.0",
    "kind": "framework",
    "publisher": "NIST",
    "sourceUrl": "https://www.nist.gov/cyberframework",
}
SERVICE_CATEGORIES = {
    "ctem": ["ti", "cspm", "vm-scanner"],
    "dast": ["sast-dast", "waf"],
    "dr": ["backup-recovery"],
    "grc": ["grc", "risk-quant"],
    "ir": ["soar", "ddil-forensics"],
    "sast": ["sast-dast"],
    "spear-phishing": ["email-sec", "ti", "awareness"],
    "vm": ["vm-scanner", "cspm", "siem"],
}
DIRECT_CONTROLS = {
    "grc": ["GV.OV-01", "GV.OV-03", "GV.RR-02", "GV.SC-04", "GV.SC-05", "GV.SC-07", "GV.SC-08", "GV.SC-10"],
    "ir": ["RS.CO-02", "RS.MA-05"],
    "dr": ["RC.CO-03", "RC.CO-04", "RC.RP-04", "RC.RP-06"],
}
CATEGORY_CONTROLS = {
    "vm-scanner": ["ID.RA-01", "ID.RA-02", "ID.RA-05", "PR.PS-02", "DE.CM-09"],
    "sast-dast": ["ID.RA-01", "ID.AM-08", "PR.PS-01", "PR.PS-06"],
    "cspm": ["ID.AM-01", "ID.AM-02", "PR.IR-01", "DE.CM-01", "DE.CM-09"],
    "siem": ["DE.AE-02", "DE.AE-03", "RS.AN-03", "DE.CM-01", "DE.CM-03"],
    "ti": ["ID.RA-02", "DE.AE-02", "DE.AE-08", "GV.RM-04"],
    "risk-quant": ["GV.RM-02", "GV.RM-06", "ID.RA-04", "ID.RA-05", "ID.RA-06"],
    "soar": ["RS.MA-01", "RS.MA-02", "RS.MA-03", "RS.MA-04", "RS.MI-01", "RS.MI-02", "RS.CO-03"],
    "ddil-forensics": ["RS.AN-03", "RS.AN-06", "RS.AN-07", "RS.AN-08"],
    "backup-recovery": ["PR.DS-11", "RC.RP-01", "RC.RP-02", "RC.RP-03", "RC.RP-05"],
    "waf": ["PR.DS-02", "PR.IR-01", "DE.CM-01"],
    "grc": ["GV.RM-01", "GV.RM-02", "GV.RM-04", "GV.OV-01", "GV.OC-03", "ID.RA-05"],
    "iam": ["PR.AA-01", "PR.AA-02", "PR.AA-03", "PR.AA-05", "PR.AA-06"],
    "ztna": ["PR.AA-03", "PR.AA-05", "PR.IR-01", "PR.DS-01"],
    "secrets": ["PR.AA-01", "PR.AA-05", "PR.DS-01"],
    "edr-xdr": ["DE.CM-01", "DE.CM-03", "DE.CM-09", "RS.MI-01", "RS.MI-02"],
    "ptesp": ["ID.RA-01", "ID.RA-02", "PR.PS-02", "GV.RM-04"],
}


def _load(name: str) -> Any:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def _asset_id(value: Any) -> str:
    if not isinstance(value, str) or ASSET_ID.fullmatch(value) is None:
        raise ValueError("invalid asset id")
    return value


def _cycles(nodes: list[str], edges: dict[str, list[str]]) -> list[list[str]]:
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    found: list[list[str]] = []
    counter = 0
    for root in nodes:
        if root in index:
            continue
        work = [{"node": root, "next": 0}]
        while work:
            frame = work[-1]
            node = frame["node"]
            if frame["next"] == 0:
                index[node] = counter
                low[node] = counter
                counter += 1
                stack.append(node)
                on_stack.add(node)
            targets = edges.get(node, [])
            if frame["next"] < len(targets):
                target = targets[frame["next"]]
                frame["next"] += 1
                if target not in index:
                    work.append({"node": target, "next": 0})
                elif target in on_stack:
                    low[node] = min(low[node], index[target])
                continue
            work.pop()
            if work:
                parent = work[-1]["node"]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                component: list[str] = []
                member = None
                while member != node:
                    member = stack.pop()
                    on_stack.remove(member)
                    component.append(member)
                if len(component) > 1 or node in edges.get(node, []):
                    found.append(sorted(component))
    return sorted(found, key=lambda item: item[0])


def check_cmdb(payload: Mapping[str, Any]) -> dict[str, Any]:
    assets = payload.get("assets")
    dependencies = payload.get("dependencies")
    if not isinstance(assets, list) or not 1 <= len(assets) <= MAX_ASSETS:
        raise ValueError("invalid asset list")
    if not isinstance(dependencies, list) or len(dependencies) > MAX_DEPENDENCIES:
        raise ValueError("invalid dependency list")
    ids: set[str] = set()
    duplicate_assets: list[str] = []
    for asset in assets:
        if not isinstance(asset, Mapping):
            raise ValueError("invalid asset id")
        asset_id = _asset_id(asset.get("id"))
        if asset_id in ids:
            duplicate_assets.append(asset_id)
        ids.add(asset_id)
    seen: set[str] = set()
    duplicate_dependencies: list[dict[str, str]] = []
    unknown: list[dict[str, str]] = []
    edges: dict[str, list[str]] = {}
    connected: set[str] = set()
    for edge in dependencies:
        if not isinstance(edge, Mapping):
            raise ValueError("invalid asset id")
        origin = _asset_id(edge.get("from"))
        target = _asset_id(edge.get("to"))
        key = f"{origin}\u0000{target}"
        if key in seen:
            duplicate_dependencies.append({"from": origin, "to": target})
            continue
        seen.add(key)
        if origin not in ids or target not in ids:
            unknown.append({"from": origin, "to": target})
            continue
        edges.setdefault(origin, []).append(target)
        connected.add(origin)
        connected.add(target)
    listed = list(ids)
    return {
        "assets": len(ids),
        "dependencies": len(seen),
        "duplicateAssets": duplicate_assets,
        "duplicateDependencies": duplicate_dependencies,
        "unknownReferences": unknown,
        "isolated": [asset_id for asset_id in listed if asset_id not in connected],
        "cycles": _cycles(listed, edges),
        "boundary": "Structural check of the reported list. Nothing was discovered or stored.",
    }


def _extensions(name: str) -> list[str]:
    return name.lower().split(".")[1:]


def email_signals(payload: Mapping[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= MAX_MESSAGES:
        raise ValueError("invalid message list")
    rendered = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("invalid message")
        message_id = message.get("id")
        if not isinstance(message_id, str) or not 1 <= len(message_id) <= 200:
            raise ValueError("invalid message")
        attachments = message.get("attachments")
        if not isinstance(attachments, list) or len(attachments) > 50:
            raise ValueError("invalid attachment")
        signals: set[str] = set()
        for attachment in attachments:
            if not isinstance(attachment, Mapping):
                raise ValueError("invalid attachment")
            name = attachment.get("name")
            if not isinstance(name, str) or not 1 <= len(name) <= 255:
                raise ValueError("invalid attachment")
            parts = _extensions(name)
            last = parts[-1] if parts else ""
            if last in ARCHIVE and attachment.get("encrypted") is True:
                signals.add("password-protected-archive")
                if message.get("bodyMentionsPassword") is True:
                    signals.add("ppap-pattern")
            if last in EXECUTABLE:
                signals.add("executable-attachment")
            if last in MACRO:
                signals.add("macro-document")
            if len(parts) >= 2 and last in EXECUTABLE and parts[-2] in DECOY:
                signals.add("double-extension")
        rendered.append({"id": message_id, "signals": sorted(signals)})
    counts: dict[str, int] = {}
    for message in rendered:
        for signal in message["signals"]:
            counts[signal] = counts.get(signal, 0) + 1
    flagged = sum(1 for message in rendered if message["signals"])
    return {
        "messages": rendered,
        "flagged": flagged,
        "counts": counts,
        "boundary": "Signals come from attachment names and the caller's flags. No file or body was read, and a message without a signal is not safe.",
    }


def _unique(items: list[str]) -> list[str]:
    return sorted(set(items))


def compliance_fit(framework_id: str) -> dict[str, Any]:
    catalog = _load("compliance-catalog-meta.json")
    products = _load("compliance-fit-products.json")
    if framework_id not in catalog["frameworkIds"]:
        raise ValueError("unknown framework")
    csf = framework_id == FRAMEWORK["id"]

    def controls_for(service: str, categories: list[str]) -> list[str]:
        gathered = [control for category in categories for control in CATEGORY_CONTROLS.get(category, [])]
        gathered.extend(DIRECT_CONTROLS.get(service, []))
        return _unique(gathered)

    services = []
    for service, categories in sorted(SERVICE_CATEGORIES.items()):
        item: dict[str, Any] = {
            "service": service,
            "categories": categories,
            "controls": controls_for(service, categories) if csf else None,
        }
        if csf and service in DIRECT_CONTROLS:
            item["directControls"] = DIRECT_CONTROLS[service]
        services.append(item)
    live_controls = [
        control
        for product in products
        if product.get("status") == "api-live"
        for control in product.get("controls", [])
    ]
    all_controls = _unique([*(control for service in services for control in (service["controls"] or [])), *live_controls]) if csf else None
    framework = {**FRAMEWORK, "controlMapping": "committed"} if csf else {"id": framework_id, "controlMapping": "not-committed"}
    return {
        "ok": True,
        "framework": framework,
        "catalog": {"categoryCount": catalog["categoryCount"], "productCount": catalog["productCount"]},
        "transitiveChain": "service -> category -> control, plus service -> control (direct)",
        "services": services,
        "products": products if csf else None,
        "allControls": all_controls,
        "boundary": "A mapping of committed control ids. This is not an audit result.",
    }


def service_catalog() -> dict[str, Any]:
    return {
        "services": _load("security-services.json"),
        "boundary": "Catalog descriptions only. This does not scan, probe, or exploit.",
    }


def spear_phishing_defence() -> dict[str, Any]:
    return _load("spear-phishing-defence.json")


def _emit(payload: dict[str, Any], code: int = 0) -> int:
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return code


def _read_input(path: str | None) -> Any:
    if path:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return json.load(sys.stdin)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Mithril security records.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("cmdb", "email"):
        command = sub.add_parser(name)
        command.add_argument("--input")
    fit = sub.add_parser("fit")
    fit.add_argument("--framework", required=True)
    sub.add_parser("catalog")
    sub.add_parser("spear-phishing")
    args = parser.parse_args(argv)
    try:
        if args.command == "cmdb":
            payload = check_cmdb(_read_input(args.input))
        elif args.command == "email":
            payload = email_signals(_read_input(args.input))
        elif args.command == "fit":
            payload = compliance_fit(args.framework)
        elif args.command == "catalog":
            payload = service_catalog()
        else:
            payload = spear_phishing_defence()
    except (ValueError, OSError, json.JSONDecodeError, KeyError) as exc:
        return _emit({"error": str(exc)}, 2)
    return _emit(payload)


if __name__ == "__main__":
    raise SystemExit(main())
