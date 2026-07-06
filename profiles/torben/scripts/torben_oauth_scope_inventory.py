#!/usr/bin/env python3
"""Enumerate actually granted Google OAuth scopes for Torben accounts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCRIPT_PATH = Path(__file__).resolve()


def _repo_root() -> Path:
    current = SCRIPT_PATH
    for parent in current.parents:
        if (parent / "hermes_cli").exists():
            return parent
        if parent.name == ".hermes" and (parent / "hermes-agent" / "hermes_cli").exists():
            return parent / "hermes-agent"
    fallback = os.getenv("HERMES_REPO_ROOT")
    if fallback:
        return Path(fallback)
    return Path("/Users/ericfreeman/.hermes/hermes-agent")


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hermes_constants import get_hermes_home  # noqa: E402
from hermes_cli.signal_coo.google_auth import GoogleAccount, check_account, load_google_accounts  # noqa: E402


TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"
READ_SCOPES = {
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.metadata",
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.calendarlist.readonly",
    "https://www.googleapis.com/auth/calendar.events.readonly",
}
WRITE_SCOPES = {
    "https://mail.google.com/",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.labels",
    "https://www.googleapis.com/auth/gmail.insert",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/calendar.calendarlist",
}
GATED_CATEGORIES = ("gmail_archive", "gmail_trash", "calendar_edit")


def _iso(value: datetime | None = None) -> str:
    return (value or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def classify_scope(scope: str) -> dict[str, str]:
    service = "other"
    if "gmail" in scope or scope == "https://mail.google.com/":
        service = "gmail"
    elif "calendar" in scope:
        service = "calendar"
    if scope in WRITE_SCOPES:
        access = "write"
    elif scope in READ_SCOPES:
        access = "read"
    elif service in {"gmail", "calendar"}:
        access = "unknown"
    else:
        access = "other"
    return {"scope": scope, "service": service, "access": access}


def _read_token(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Token payload must be an object: {path}")
    return payload


def fetch_tokeninfo(access_token: str, *, opener: Callable[..., Any] | None = None) -> dict[str, Any]:
    query = urllib.parse.urlencode({"access_token": access_token})
    request = urllib.request.Request(f"{TOKENINFO_URL}?{query}", method="GET")
    open_fn = opener or urllib.request.urlopen
    with open_fn(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("tokeninfo response must be a JSON object")
    return payload


def _granted_scopes_from_tokeninfo(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("scope") or payload.get("scopes") or ""
    if isinstance(raw, str):
        return sorted(scope for scope in raw.split() if scope)
    if isinstance(raw, list):
        return sorted(str(scope) for scope in raw if str(scope).strip())
    return []


def _is_work_account(account: GoogleAccount) -> bool:
    role = (account.role or "").strip().lower()
    return role == "work" or account.alias.startswith("work_")


def enumerate_account_scopes(
    account: GoogleAccount,
    *,
    tokeninfo_fetcher: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    auth = check_account(account)
    result: dict[str, Any] = {
        "alias": account.alias,
        "email": account.email,
        "role": account.role,
        "enabled": account.enabled,
        "work_account": _is_work_account(account),
        "token_status": auth.status,
        "token_status_reason": auth.reason,
        "configured_scopes": sorted(account.scopes),
        "granted_scopes": [],
        "scope_source": "google_tokeninfo",
        "classifications": [],
        "findings": [],
    }
    if not auth.status.startswith("authenticated"):
        result["scope_source"] = "unavailable"
        return result
    token_payload = _read_token(account.token_path)
    access_token = str(token_payload.get("token") or "")
    if not access_token:
        result["token_status"] = "token_invalid"
        result["token_status_reason"] = "missing access token after auth check"
        result["scope_source"] = "unavailable"
        return result
    fetcher = tokeninfo_fetcher or (lambda token: fetch_tokeninfo(token))
    tokeninfo = fetcher(access_token)
    granted_scopes = _granted_scopes_from_tokeninfo(tokeninfo)
    classifications = [classify_scope(scope) for scope in granted_scopes]
    findings = [
        {
            "type": "work_google_write_scope",
            "severity": "type_1",
            "scope": item["scope"],
            "service": item["service"],
            "category_gate": "calendar_edit" if item["service"] == "calendar" else "gmail_archive,gmail_trash",
        }
        for item in classifications
        if result["work_account"] and item["service"] in {"gmail", "calendar"} and item["access"] == "write"
    ]
    result.update(
        {
            "granted_scopes": granted_scopes,
            "classifications": classifications,
            "findings": findings,
        }
    )
    return result


def build_scope_inventory(
    *,
    config_path: Path,
    tokeninfo_fetcher: Callable[[str], dict[str, Any]] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    accounts = list(load_google_accounts(config_path).values())
    account_results = [
        enumerate_account_scopes(account, tokeninfo_fetcher=tokeninfo_fetcher)
        for account in accounts
        if account.enabled
    ]
    type_1 = [
        {"account": account["alias"], "email": account["email"], **finding}
        for account in account_results
        for finding in account.get("findings", [])
        if finding.get("severity") == "type_1"
    ]
    gate_status = "blocked_type_1" if type_1 else "clear"
    gates = {
        category: {
            "status": gate_status,
            "floor": "packet_only" if type_1 else None,
            "reason": "work_account_google_write_scope" if type_1 else "no_work_account_write_scope_found",
        }
        for category in GATED_CATEGORIES
    }
    return {
        "schema": "torben.oauth-scope-inventory.v1",
        "generated_at": _iso(now),
        "status": "type_1_findings" if type_1 else "clean",
        "scope_source": "google_tokeninfo",
        "accounts": account_results,
        "type_1_findings": type_1,
        "category_gates": gates,
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def write_escalation(path: Path, inventory: dict[str, Any]) -> Path | None:
    findings = inventory.get("type_1_findings") or []
    if not findings:
        return None
    lines = [
        "Torben input needed: Google work-account OAuth write scopes found.",
        "",
        "P0-9 verified live token grants with Google tokeninfo. Until you decide otherwise, gmail_archive, gmail_trash, and calendar_edit stay pinned at packet_only.",
        "",
        "Findings:",
    ]
    for finding in findings:
        lines.append(f"- {finding['account']} <{finding['email']}>: {finding['service']} write scope {finding['scope']}")
    lines.extend(
        [
            "",
            "Decision needed: revoke/re-consent the work accounts read-only, or accept these grants with compensating controls before Phase 1 automation can promote those categories.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Google accounts config path")
    parser.add_argument("--output", help="Scope inventory output path")
    parser.add_argument("--escalation-output", help="Input-needed text output path")
    parser.add_argument("--json", action="store_true", help="Print JSON inventory")
    args = parser.parse_args(argv)

    home = get_hermes_home()
    state_dir = home / "state"
    inventory = build_scope_inventory(
        config_path=Path(args.config) if args.config else home / "config" / "google_accounts.yaml",
    )
    output_path = write_json_atomic(
        Path(args.output) if args.output else state_dir / "torben-oauth-scope-inventory.json",
        inventory,
    )
    escalation_path = write_escalation(
        Path(args.escalation_output) if args.escalation_output else state_dir / "torben-oauth-scope-input-needed.txt",
        inventory,
    )
    if args.json:
        print(json.dumps({"output_path": str(output_path), "escalation_path": str(escalation_path) if escalation_path else None, **inventory}, indent=2, sort_keys=True))
    elif escalation_path:
        print(escalation_path.read_text(encoding="utf-8"), end="")
    else:
        print(f"OAuth scope inventory clean: {output_path}")
    return 0


if __name__ == "__main__":
    if "--json" in sys.argv:
        raise SystemExit(main())
    from torben_job_contract import run_job

    raise SystemExit(run_job("torben-oauth-scope-inventory", main))
