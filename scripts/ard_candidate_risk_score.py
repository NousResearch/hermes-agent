#!/usr/bin/env python3
"""Risk-score ARD candidates and emit install plans without side effects."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from hermes_constants import get_hermes_home

SECRET_FIELD_RE = re.compile(r"(?i)(api[_-]?key|token|secret|password|private[_-]?key)")
TOKEN_VALUE_RE = re.compile(r"(?i)(sk-[A-Za-z0-9_.-]{8,}|tk_[A-Za-z0-9_.-]{8,}|Bearer\s+[A-Za-z0-9._-]{8,})")
LOCAL_PATH_RE = re.compile(r"(/home/[A-Za-z0-9_.-]+/|/mnt/[a-z]/|C:\\Users\\|C:/Users/)")


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _has_secret_or_path(value: Any) -> bool:
    def walk(obj: Any) -> bool:
        if isinstance(obj, dict):
            for key, inner in obj.items():
                if isinstance(key, str) and SECRET_FIELD_RE.search(key):
                    return True
                if walk(inner):
                    return True
        elif isinstance(obj, list):
            return any(walk(item) for item in obj)
        elif isinstance(obj, str):
            return bool(TOKEN_VALUE_RE.search(obj) or LOCAL_PATH_RE.search(obj))
        return False
    return walk(value)


def _reasons_for_entry(entry: dict[str, Any], *, visibility: str = "private") -> list[str]:
    reasons: list[str] = []
    url = str(entry.get("url") or "")
    metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    parsed = urlparse(url)
    if url.startswith("stdio:") or metadata.get("transport") in {"stdio", "local-command"}:
        reasons.append("local_command_transport")
    if parsed.scheme == "http":
        reasons.append("plaintext_http_transport")
    if not url:
        reasons.append("missing_remote_url")
    if metadata.get("requiresEnv") or metadata.get("requiredEnv") or metadata.get("env"):
        reasons.append("requires_env_secrets")
    registry = metadata.get("mcpRegistry") if isinstance(metadata.get("mcpRegistry"), dict) else {}
    if registry.get("isLatest") is False:
        reasons.append("not_latest_registry_version")
    if registry.get("status") not in {None, "active"}:
        reasons.append("inactive_registry_status")
    if visibility == "public" and _has_secret_or_path(entry):
        reasons.append("public_secret_or_path_leak")
    return reasons


def score_entry(entry: dict[str, Any], *, visibility: str = "private") -> dict[str, Any]:
    reasons = _reasons_for_entry(entry, visibility=visibility)
    if "public_secret_or_path_leak" in reasons:
        decision = "deny"
        risk = "critical"
    elif any(r in reasons for r in {"local_command_transport", "requires_env_secrets"}):
        decision = "review"
        risk = "high"
    elif any(r in reasons for r in {"plaintext_http_transport", "missing_remote_url", "not_latest_registry_version", "inactive_registry_status"}):
        decision = "review"
        risk = "medium"
    else:
        decision = "allow"
        risk = "low"
    return {
        "identifier": entry.get("identifier"),
        "displayName": entry.get("displayName"),
        "type": entry.get("type"),
        "url": entry.get("url"),
        "decision": decision,
        "risk": risk,
        "reasons": reasons,
    }


def plan_install(entry: dict[str, Any], *, visibility: str = "private") -> dict[str, Any]:
    score = score_entry(entry, visibility=visibility)
    plan = {
        **score,
        "side_effects": [],
        "next_action": "manual_register_after_review",
        "rollback": "remove MCP server/card registration from Hermes config/cache",
        "steps": [
            "inspect ARD entry metadata and remote URL",
            "run risk score and conformance checks",
            "if allowed, register MCP/card explicitly via Hermes config or cache",
            "smoke test capability before adding to default tool routes",
        ],
    }
    if score["decision"] == "deny":
        plan["next_action"] = "do_not_install"
    elif score["decision"] == "review":
        plan["next_action"] = "manual_security_review_required"
    return plan


def default_catalog_paths() -> list[Path]:
    home = get_hermes_home()
    return [
        home / ".hub" / "ard-mcp-registry-cache.json",
        home / ".hub" / "ard-gitdb-candidates.json",
        home / ".hub" / "ard-cache.json",
        home / ".well-known" / "ai-catalog.json",
        home / "skills" / ".hub" / "ard-cache.json",
    ]


def _load_entry(catalog: Path, identifier: str) -> dict[str, Any]:
    data = json.loads(catalog.read_text(encoding="utf-8"))
    entries = data.get("entries") if isinstance(data, dict) else []
    for entry in entries if isinstance(entries, list) else []:
        if isinstance(entry, dict) and entry.get("identifier") == identifier:
            return entry
    raise KeyError(identifier)


def find_entry_in_catalogs(identifier: str, catalogs: list[Path] | None = None) -> dict[str, Any]:
    checked: list[str] = []
    for catalog in catalogs or default_catalog_paths():
        if not catalog.exists():
            continue
        checked.append(str(catalog))
        try:
            return _load_entry(catalog, identifier)
        except (KeyError, ValueError, OSError, json.JSONDecodeError):
            continue
    raise SystemExit(f"Entry not found: {identifier} (checked {len(checked)} catalog files)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Risk-score an ARD entry and print a plan-only install report")
    parser.add_argument("--catalog", type=Path, action="append", help="Catalog file to search. Repeatable. Defaults to profile ARD caches.")
    parser.add_argument("--identifier", required=True)
    parser.add_argument("--visibility", choices=["public", "private"], default="private")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    entry = find_entry_in_catalogs(args.identifier, args.catalog)
    result = plan_install(entry, visibility=args.visibility) if args.plan else score_entry(entry, visibility=args.visibility)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"{result['decision']} risk={result['risk']} {result.get('identifier')}")
        for reason in result["reasons"]:
            print(f"- {reason}")
    return 0 if result["decision"] in {"allow", "review"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
