#!/usr/bin/env python3
"""Bounded evidence review for the Mithril security app.

Ports the decisions in app.mithril.fund assurance, mobile, and operations.
It records what an operator reported. It does not enroll devices, send files,
query a registry, or issue a compliance decision.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any, Mapping

DEVICE_ID = re.compile(r"^[A-Za-z0-9._:-]{1,100}$")
SUPPORTED_PLATFORMS = {"macos", "ios", "ipados", "android", "windows", "linux"}
TRANSFER_METHODS = {"password-zip", "email-attachment", "managed-link"}
TRANSFER_ANSWERS = {"yes", "no", "unknown"}
EVIDENCE_ANSWERS = {"unknown", "observed", "missing"}

PPAP_SOURCE = "https://security-portal.nisc.go.jp/guidance/pdf/handbook/handbook-05.pdf"
INVENTORY_SOURCE = "https://csrc.nist.gov/pubs/sp/800/53/r5/upd1/final"
DISCOVERY_SOURCE = (
    "https://www.cisa.gov/news-events/directives/"
    "bod-23-01-improving-asset-visibility-and-vulnerability-detection-federal-networks"
)
MDM_ANNEX = (
    "https://dl.dod.cyber.mil/wp-content/uploads/stigs/pdf/"
    "U_DoD_Annex_for_MDM_PP_Server_V4-0_V1R3.pdf"
)
AGENT_ANNEX = (
    "https://dl.dod.cyber.mil/wp-content/uploads/stigs/pdf/"
    "U_DoD_Annex_for_PP-Module_for_MDM_Agents_V1-0_V1R3.pdf"
)
IOS_STIG = (
    "https://dl.dod.cyber.mil/wp-content/uploads/stigs/zip/"
    "U_Apple_iOS-iPadOS_26_V1R3_STIG.zip"
)
ANDROID_STIG = (
    "https://dl.dod.cyber.mil/wp-content/uploads/stigs/zip/"
    "U_Google_Android_16_Y26M05_STIG.zip"
)
SBOM_SOURCE = (
    "https://media.defense.gov/2023/Dec/14/2003359097/-1/-1/0/"
    "CSI-SCRM-SBOM-Management-v1.1.PDF"
)
SASE_SOURCE = (
    "https://dl.dod.cyber.mil/wp-content/uploads/devsecops/pdf/"
    "unclass-CNAP-RefDesign_ver-1.0.pdf"
)
SCRM_SOURCE = "https://www.esd.whs.mil/Portals/54/Documents/DD/issuances/dodi/520044p.pdf"

TRANSFER_CONTROLS = [
    {"id": "recipient", "label": "宛先と受取人の身元を確認できる"},
    {"id": "authentication", "label": "受取人の認証を必須にできる"},
    {"id": "expiry", "label": "アクセス期限を設定できる"},
    {"id": "revocation", "label": "送信者がアクセスを取り消せる"},
    {"id": "audit", "label": "アクセス記録を確認できる"},
    {"id": "scan", "label": "共有前にファイルを検査できる"},
]

MDM_CONTROLS = [
    {
        "id": "mdm-command-audit",
        "rule": "FAU_GEN.1.1(1)",
        "label": "MDM agent commands appear in audit records",
        "source": MDM_ANNEX,
    },
    {
        "id": "mdm-admin-roles",
        "rule": "FMT_SMR.1.1(1)",
        "label": "Administrator, configuration, device-group and auditor roles are defined",
        "source": MDM_ANNEX,
    },
    {
        "id": "mdm-log-transfer",
        "rule": "FMT_SMF.1.1(2)",
        "label": "Server audit logs transfer to another server for storage and analysis",
        "source": MDM_ANNEX,
    },
    {
        "id": "mdm-enrollment-alert",
        "rule": "FAU_ALT_EXT.2.1",
        "label": "Changes in enrollment status produce an alert",
        "source": AGENT_ANNEX,
    },
]

MTD_CONTROLS = {
    "ios": [
        {
            "id": "mtd-managed-ios",
            "rule": "AIOS-26-017700",
            "label": "Site-approved MTD app is listed as a managed deployment",
            "source": IOS_STIG,
        },
        {
            "id": "mtd-device-ios",
            "rule": "AIOS-26-017700",
            "label": "MTD app is installed on the managed iPhone or iPad",
            "source": IOS_STIG,
        },
    ],
    "android": [
        {
            "id": "mtd-managed-android",
            "rule": "GOOG-16-013400",
            "label": "MTD app is listed as a managed deployment in the MDM console",
            "source": ANDROID_STIG,
        },
        {
            "id": "mtd-device-android",
            "rule": "GOOG-16-013400",
            "label": "MTD app is installed on the managed Android device",
            "source": ANDROID_STIG,
        },
    ],
}

ASSURANCE_CONTROLS = {
    "sbom": [
        {
            "id": "sbom-components",
            "reference": "NSA SBOM Management v1.1 · p. 6",
            "label": "Delivered software has a component inventory, including third-party dependencies",
            "source": SBOM_SOURCE,
        },
        {
            "id": "sbom-runtime",
            "reference": "NSA SBOM Management v1.1 · pp. 6–7",
            "label": "Runtime dependencies outside the component inventory are documented",
            "source": SBOM_SOURCE,
        },
        {
            "id": "sbom-integrity",
            "reference": "NSA SBOM Management v1.1 · pp. 6, 9",
            "label": "SBOM and component integrity can be checked with hashes or signatures",
            "source": SBOM_SOURCE,
        },
        {
            "id": "sbom-vulnerabilities",
            "reference": "NSA SBOM Management v1.1 · pp. 9–10",
            "label": "Components can be related to updated vulnerability data and response decisions",
            "source": SBOM_SOURCE,
        },
    ],
    "sase": [
        {
            "id": "sase-identity",
            "reference": "DoD CNAP Reference Design v1r1 · §4.2",
            "label": "The access path authenticates the requesting entity and validates authorization",
            "source": SASE_SOURCE,
        },
        {
            "id": "sase-connector",
            "reference": "DoD CNAP Reference Design v1r1 · §4.2",
            "label": "If service-initiated ZTNA is used, an application connector establishes an outbound path to the provider",
            "source": SASE_SOURCE,
        },
        {
            "id": "sase-posture",
            "reference": "DoD CNAP Reference Design v1r1 · §5",
            "label": "Access decisions consider identity, need-to-know and device posture",
            "source": SASE_SOURCE,
        },
        {
            "id": "sase-logs",
            "reference": "DoD CNAP Reference Design v1r1 · §5.1",
            "label": "Access and security events reach centralized logging and monitoring",
            "source": SASE_SOURCE,
        },
    ],
    "scrm": [
        {
            "id": "scrm-criticality",
            "reference": "DoDI 5200.44 (2024) · §§1.1(a)(4), 1.2(a–b)",
            "label": "Mission-critical functions and components, including spare or replacement parts, are identified",
            "next_step": "Identify the mission function, its critical components and replacement parts.",
            "source": SCRM_SOURCE,
        },
        {
            "id": "scrm-suppliers",
            "reference": "DoDI 5200.44 (2024) · §1.2(c)",
            "label": "Critical-component supplier due diligence uses supply-chain visibility and relevant threat analysis",
            "next_step": "Record critical suppliers, sub-tier visibility and the threat evidence used for the decision.",
            "source": SCRM_SOURCE,
        },
        {
            "id": "scrm-lifecycle",
            "reference": "DoDI 5200.44 (2024) · §1.2(d)(2)",
            "label": "Hardware, software and subcomponent risks are reviewed across the system life cycle",
            "next_step": "Document risk and mitigation owners from acquisition through operation and sustainment.",
            "source": SCRM_SOURCE,
        },
        {
            "id": "scrm-integrity",
            "reference": "DoDI 5200.44 (2024) · §§1.2(d)(3–4)",
            "label": "Counterfeit or malicious components are addressed through prevention and testing",
            "next_step": "Define acceptance testing and a response path for suspect components or functions.",
            "source": SCRM_SOURCE,
        },
        {
            "id": "scrm-plan",
            "reference": "DoDI 5200.44 (2024) · §1.2(f)",
            "label": "Criticality, mitigations and risk acceptance are documented in the program protection plan",
            "next_step": "Document criticality, selected mitigations and explicit risk acceptance in the applicable plan.",
            "source": SCRM_SOURCE,
        },
    ],
}


def _number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _answers(raw: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("invalid evidence answer")
    return raw


def mobile_controls(view: str, platform: str | None = None) -> list[dict[str, Any]]:
    if view == "mdm":
        return MDM_CONTROLS
    if view == "mtd":
        return MTD_CONTROLS.get(platform or "", [])
    return []


def review_transfer(method: str, answers: Mapping[str, Any] | None) -> dict[str, Any]:
    if method not in TRANSFER_METHODS:
        raise ValueError("unsupported transfer method")
    given = _answers(answers)
    results = []
    for control in TRANSFER_CONTROLS:
        answer = given.get(control["id"], "unknown")
        if answer not in TRANSFER_ANSWERS:
            raise ValueError("invalid transfer answer")
        results.append({**control, "source": PPAP_SOURCE, "status": answer})
    missing = [row["label"] for row in results if row["status"] == "no"]
    unknown = [row["label"] for row in results if row["status"] == "unknown"]
    if method == "password-zip":
        decision = "replace-ppap"
    elif method == "email-attachment":
        decision = "review-attachment"
    elif missing or unknown:
        decision = "incomplete-plan"
    else:
        decision = "planned-only"
    return {
        "surface": "transfer",
        "method": method,
        "results": results,
        "missing": missing,
        "unknown": unknown,
        "decision": decision,
        "boundary": "No file bytes, password, address, or link were reviewed.",
    }


def _index_devices(rows: list[Any], source: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("invalid device id")
        device_id = row.get("id")
        if not isinstance(device_id, str) or DEVICE_ID.fullmatch(device_id) is None:
            raise ValueError("invalid device id")
        if device_id in indexed:
            raise ValueError("duplicate device id")
        indexed[device_id] = dict(row)
    return indexed


def reconcile_devices(
    roster: list[Any],
    reports: list[Any],
    now: Any,
    max_age_ms: Any,
) -> dict[str, Any]:
    if not (
        _number(now)
        and now >= 0
        and _number(max_age_ms)
        and max_age_ms > 0
    ):
        raise ValueError("invalid inventory clock or age")
    if not isinstance(roster, list) or not isinstance(reports, list):
        raise ValueError("invalid device id")
    if len(roster) == 0 or len(reports) == 0:
        reason = "roster-not-provided" if len(roster) == 0 else "reports-not-provided"
        return {
            "surface": "devices",
            "state": "unmeasured",
            "rows": [],
            "coverage": None,
            "reason": reason,
            "sources": {"inventory": INVENTORY_SOURCE, "discovery": DISCOVERY_SOURCE},
        }
    expected = _index_devices(roster, "roster")
    observed = _index_devices(reports, "reports")
    rows = []
    for item in roster:
        device_id = item["id"]
        platform = item.get("platform")
        report = observed.get(device_id)
        reported_at = None if report is None else report.get("reported_at", report.get("reported-at"))
        if report is None:
            status = "missing"
        elif report.get("platform") not in SUPPORTED_PLATFORMS:
            status = "invalid-report"
        elif not (_number(reported_at) and 0 <= reported_at <= now):
            status = "invalid-report"
        elif platform != report.get("platform"):
            status = "platform-mismatch"
        elif (now - reported_at) > max_age_ms:
            status = "stale"
        else:
            status = "matched"
        rows.append(
            {
                "id": device_id,
                "platform": platform,
                "owner": item.get("owner"),
                "status": status,
                "reported_at": reported_at,
            }
        )
    unexpected = [
        {"id": device_id, "platform": report.get("platform"), "status": "unexpected"}
        for device_id, report in observed.items()
        if device_id not in expected
    ]
    all_rows = rows + unexpected
    matched = sum(1 for row in rows if row["status"] == "matched")
    counts: dict[str, int] = {}
    for row in all_rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return {
        "surface": "devices",
        "state": "reconciled",
        "rows": all_rows,
        "counts": counts,
        "expected": len(roster),
        "reported": len(reports),
        "coverage": matched / len(roster),
        "sources": {"inventory": INVENTORY_SOURCE, "discovery": DISCOVERY_SOURCE},
        "boundary": "Roster reconciliation only. No discovery, enrollment, or command was run.",
    }


def _review_evidence(
    surface: str,
    items: list[dict[str, Any]],
    answers: Mapping[str, Any] | None,
    *,
    gap_status: str,
    gap_count_key: str,
    gap_conclusion: str,
) -> dict[str, Any]:
    if not items:
        if surface == "mobile":
            raise ValueError("unsupported mobile security review")
        raise ValueError("unsupported assurance review")
    given = _answers(answers)
    results = []
    for control in items:
        answer = given.get(control["id"], "unknown")
        if answer not in EVIDENCE_ANSWERS:
            raise ValueError("invalid evidence answer")
        status = {"observed": "reported-present", "missing": gap_status, "unknown": "unmeasured"}[answer]
        results.append({**control, "status": status})
    counts: dict[str, int] = {}
    for row in results:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    if counts.get(gap_status, 0) > 0:
        conclusion = gap_conclusion
    elif counts.get("unmeasured", 0) > 0:
        conclusion = "insufficient-evidence"
    else:
        conclusion = "self-reported-only"
    return {
        "surface": surface,
        "results": results,
        "reported_present": counts.get("reported-present", 0),
        gap_count_key: counts.get(gap_status, 0),
        "unmeasured": counts.get("unmeasured", 0),
        "conclusion": conclusion,
        "boundary": "Self-report only. This is not a certificate, authorization, or observed control execution.",
    }


def review_mobile(view: str, platform: str | None, answers: Mapping[str, Any] | None) -> dict[str, Any]:
    result = _review_evidence(
        "mobile",
        mobile_controls(view, platform),
        answers,
        gap_status="finding",
        gap_count_key="findings",
        gap_conclusion="reported-finding",
    )
    result["view"] = view
    result["platform"] = platform
    return result


def review_assurance(view: str, answers: Mapping[str, Any] | None) -> dict[str, Any]:
    result = _review_evidence(
        "assurance",
        list(ASSURANCE_CONTROLS.get(view, [])),
        answers,
        gap_status="reported-gap",
        gap_count_key="reported_gaps",
        gap_conclusion="reported-gap",
    )
    result["view"] = view
    return result


def list_controls(surface: str, view: str | None = None, platform: str | None = None) -> dict[str, Any]:
    if surface == "transfer":
        items = [{**control, "source": PPAP_SOURCE} for control in TRANSFER_CONTROLS]
    elif surface == "mobile":
        items = mobile_controls(view or "", platform)
        if view and not items:
            raise ValueError("unsupported mobile security review")
    elif surface == "assurance":
        items = list(ASSURANCE_CONTROLS.get(view or "", []))
        if view and not items:
            raise ValueError("unsupported assurance review")
    elif surface == "devices":
        items = []
    else:
        raise ValueError("unsupported assurance review")
    return {"surface": surface, "view": view, "platform": platform, "controls": items}


def review_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    surface = payload.get("surface")
    answers = payload.get("answers")
    if surface == "transfer":
        return review_transfer(str(payload.get("method") or ""), answers)
    if surface == "devices":
        return reconcile_devices(
            payload.get("roster"),
            payload.get("reports"),
            payload.get("now"),
            payload.get("max_age_ms", payload.get("max-age-ms")),
        )
    if surface == "mobile":
        return review_mobile(str(payload.get("view") or ""), payload.get("platform"), answers)
    if surface == "assurance":
        return review_assurance(str(payload.get("view") or ""), answers)
    raise ValueError("unsupported assurance review")


def _load_json(text: str | None, path: str | None, default: Any) -> Any:
    if path:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    if text is None:
        return default
    return json.loads(text)


def _emit(payload: dict[str, Any], code: int = 0) -> int:
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Review Mithril security evidence.")
    sub = parser.add_subparsers(dest="command", required=True)

    controls = sub.add_parser("controls")
    controls.add_argument("surface", choices=["transfer", "mobile", "assurance", "devices"])
    controls.add_argument("--view")
    controls.add_argument("--platform")

    review = sub.add_parser("review")
    review.add_argument("--input")
    review.add_argument("surface", nargs="?", choices=["transfer", "mobile", "assurance", "devices"])
    review.add_argument("--view")
    review.add_argument("--platform")
    review.add_argument("--method")
    review.add_argument("--answers")
    review.add_argument("--answers-file")
    review.add_argument("--roster")
    review.add_argument("--roster-file")
    review.add_argument("--reports")
    review.add_argument("--reports-file")
    review.add_argument("--now", type=float)
    review.add_argument("--max-age-ms", type=float)

    args = parser.parse_args(argv)
    try:
        if args.command == "controls":
            payload = list_controls(args.surface, args.view, args.platform)
        elif args.input:
            with open(args.input, encoding="utf-8") as handle:
                payload = review_request(json.load(handle))
        else:
            payload = review_request(
                {
                    "surface": args.surface,
                    "view": args.view,
                    "platform": args.platform,
                    "method": args.method,
                    "answers": _load_json(args.answers, args.answers_file, {}),
                    "roster": _load_json(args.roster, args.roster_file, []),
                    "reports": _load_json(args.reports, args.reports_file, []),
                    "now": args.now,
                    "max_age_ms": args.max_age_ms,
                }
            )
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        return _emit({"error": str(exc)}, 2)
    return _emit(payload)


if __name__ == "__main__":
    raise SystemExit(main())
