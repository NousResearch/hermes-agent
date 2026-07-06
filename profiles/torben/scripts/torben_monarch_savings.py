#!/usr/bin/env python3
"""Run Torben's read-only Monarch savings research loops."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    return Path.cwd()


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hermes_constants import get_hermes_home  # noqa: E402
from hermes_cli.signal_coo.action_ledger import ActionLedger  # noqa: E402
from hermes_cli.signal_coo.monarch_savings import (  # noqa: E402
    MonarchReadOnlyClient,
    build_torben_monarch_savings_payload,
    collect_live_monarch_savings_packet,
    update_monarch_savings_ledger,
    write_monarch_savings_artifacts,
)

MONARCH_LOGIN_COMMAND = "/Users/ericfreeman/.hermes/profiles/torben/bin/torben-hermes mcp login monarch-money-mcp"
MONARCH_CONNECTOR_URL = "https://api.monarch.com/mcp"


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _token_file_state(profile_home: Path) -> list[dict[str, Any]]:
    token_dir = profile_home / "mcp-tokens"
    states: list[dict[str, Any]] = []
    for name in ("monarch-money-mcp.json", "monarch-money-mcp.client.json", "monarch-money-mcp.meta.json"):
        path = token_dir / name
        item: dict[str, Any] = {"name": name, "exists": path.exists()}
        if path.exists():
            stat = path.stat()
            item.update({"size_bytes": stat.st_size, "modified_at": _iso(datetime.fromtimestamp(stat.st_mtime, timezone.utc))})
        states.append(item)
    return states


def _reauth_step() -> dict[str, Any]:
    return {
        "command": MONARCH_LOGIN_COMMAND,
        "connector": "monarch-money-mcp",
        "endpoint": MONARCH_CONNECTOR_URL,
        "account": "Eric's intended Monarch Money account for Torben finance reads",
        "steps": [
            "Run the command on the Mac mini shell.",
            "Open the printed Monarch OAuth URL if the command does not open a browser automatically.",
            "Sign in to Monarch and approve MCP access for the intended account.",
            "Paste the callback/code back into the terminal if prompted.",
            "Rerun the daily Monarch read and confirm item_count is greater than 0.",
        ],
    }


def _all_empty_monarch_floor_payload(packet: dict[str, Any], *, loop: str, now: datetime, profile_home: Path) -> dict[str, Any] | None:
    read_calls = [item for item in (packet.get("read_tool_calls") or []) if isinstance(item, dict)]
    if not read_calls or packet.get("tool_errors"):
        return None
    if any(str(item.get("status") or "") != "ok" for item in read_calls):
        return None
    total_item_count = sum(_safe_int(item.get("item_count")) for item in read_calls)
    if total_item_count > 0:
        return None
    title = "Daily" if loop == "daily" else "Weekly"
    reauth = _reauth_step()
    text = (
        f"Torben / Monarch Savings {title}\n\n"
        "Failed: Monarch MCP returned successful read calls but every item_count was 0. "
        "This is treated as an auth/account source failure, not as 'nothing to do'.\n\n"
        f"Re-auth step: {reauth['command']}\n"
        "Approve Monarch MCP access for the intended Monarch Money account, then rerun the daily read.\n"
    )
    return {
        "task": f"torben_monarch_savings_{loop}",
        "loop": loop,
        "generated_at": _iso(now),
        "source": str(packet.get("source") or "monarch-money-mcp"),
        "source_window": packet.get("source_window") or {},
        "wakeAgent": True,
        "status": "failed",
        "reason": "monarch_live_read_all_empty",
        "empty_floor": {
            "status": "failed",
            "reason": "all_monarch_read_tools_returned_zero_items",
            "total_item_count": total_item_count,
            "read_call_count": len(read_calls),
        },
        "reauth": reauth,
        "token_files": _token_file_state(profile_home),
        "read_tool_calls": read_calls,
        "blocked_tool_calls": list(packet.get("blocked_tool_calls") or []),
        "tool_errors": list(packet.get("tool_errors") or []),
        "monarch_read_calls": len(read_calls),
        "monarch_write_calls": 0,
        "public_actions_taken": 0,
        "external_mutations": 0,
        "candidate_count": 0,
        "qualified_count": 0,
        "selected_count": 0,
        "recommendations_created": 0,
        "estimated_monthly_savings": 0,
        "estimated_annual_savings": 0,
        "candidates": [],
        "actions": [],
        "text": text,
    }


def _diagnose_monarch_packet(packet: dict[str, Any], *, loop: str, now: datetime, profile_home: Path) -> dict[str, Any]:
    read_calls = [item for item in (packet.get("read_tool_calls") or []) if isinstance(item, dict)]
    total_item_count = sum(_safe_int(item.get("item_count")) for item in read_calls)
    empty_floor = _all_empty_monarch_floor_payload(packet, loop=loop, now=now, profile_home=profile_home)
    if empty_floor is not None:
        status = "requires_reauth"
        reason = "all_monarch_read_tools_returned_zero_items"
    elif packet.get("tool_errors"):
        status = "failed"
        reason = "monarch_read_tool_errors"
    else:
        status = "ok"
        reason = "monarch_read_returned_data" if total_item_count > 0 else "no_read_calls"
    return {
        "status": status,
        "reason": reason,
        "loop": loop,
        "generated_at": _iso(now),
        "connector": "monarch-money-mcp",
        "endpoint": MONARCH_CONNECTOR_URL,
        "token_files": _token_file_state(profile_home),
        "read_tool_calls": read_calls,
        "tool_errors": list(packet.get("tool_errors") or []),
        "total_item_count": total_item_count,
        "login_step": _reauth_step() if status == "requires_reauth" else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop", choices=("daily", "weekly"), required=True)
    parser.add_argument("--fixture", help="Read a fixture packet instead of live Monarch MCP")
    parser.add_argument("--state-dir", help="Override Torben state directory")
    parser.add_argument("--json", action="store_true", help="Print the JSON artifact instead of Signal text")
    parser.add_argument("--no-stage", action="store_true", help="Do not stage FIN review handles")
    parser.add_argument("--force-wake", action="store_true", help="Ignore duplicate-delivery suppression")
    parser.add_argument("--diagnose", action="store_true", help="Print non-secret Monarch token/read diagnosis and exit")
    args = parser.parse_args(argv)

    loop = args.loop
    profile_home = get_hermes_home()
    state_dir = Path(args.state_dir) if args.state_dir else profile_home / "state"
    now = datetime.now(timezone.utc)

    if args.fixture:
        packet = _load_fixture(Path(args.fixture), loop=loop, now=now)
    else:
        packet = collect_live_monarch_savings_packet(client=MonarchReadOnlyClient(), loop=loop, now=now)

    if args.diagnose:
        diagnosis = _diagnose_monarch_packet(packet, loop=loop, now=now, profile_home=profile_home)
        print(json.dumps(diagnosis, indent=2, sort_keys=True))
        return 0

    ledger = ActionLedger(state_dir / "torben-action-ledger.jsonl")
    payload = _all_empty_monarch_floor_payload(packet, loop=loop, now=now, profile_home=profile_home)
    if payload is None:
        payload = build_torben_monarch_savings_payload(
            packet,
            loop=loop,
            ledger=ledger,
            state_path=state_dir / f"torben-monarch-savings-{loop}-state.json",
            now=now,
            stage_actions=not args.no_stage,
            force_wake=args.force_wake,
        )
    write_monarch_savings_artifacts(
        payload,
        json_path=state_dir / f"torben-monarch-savings-{loop}-latest.json",
        text_path=state_dir / f"torben-monarch-savings-{loop}-latest.txt",
    )
    update_monarch_savings_ledger(payload, ledger_path=state_dir / "torben-monarch-savings-ledger.json")

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    elif payload.get("wakeAgent"):
        print(str(payload.get("text") or "").rstrip())
    else:
        print(json.dumps({"wakeAgent": False, "reason": payload.get("reason"), "status": payload.get("status")}))
    return 1 if payload.get("status") == "failed" else 0


def _load_fixture(path: Path, *, loop: str, now: datetime) -> dict[str, Any]:
    packet = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(packet, dict):
        raise ValueError(f"Fixture root must be a JSON object: {path}")
    packet.setdefault("source", "fixture")
    packet.setdefault("loop", loop)
    packet.setdefault("generated_at", now.isoformat().replace("+00:00", "Z"))
    packet.setdefault("read_tool_calls", [])
    packet.setdefault("blocked_tool_calls", [])
    return packet


if __name__ == "__main__":
    from torben_job_contract import run_job

    raise SystemExit(run_job("torben-monarch-savings", main))
