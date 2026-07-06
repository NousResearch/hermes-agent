#!/usr/bin/env python3
"""Torben decision packet generator."""

from __future__ import annotations

import argparse
import json
import os
import re
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
    return Path("/Users/ericfreeman/.hermes/hermes-agent")


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hermes_constants import get_hermes_home  # noqa: E402
from hermes_cli.signal_coo.action_ledger import ActionLedger  # noqa: E402

APPROVE_OPTION_RE = re.compile(r"\bapprove\s+option\s+(?P<option>[1-9][0-9]*)\b", re.I)
PACKET_SECTIONS = [
    "decision needed",
    "context",
    "options",
    "recommendation",
    "approval request",
    "blocked actions",
]


def _iso(value: datetime | None = None) -> str:
    return (value or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def render_decision_packet(
    *,
    item: str,
    context: str,
    options: list[dict[str, Any]],
    recommendation: str,
    category: str,
) -> str:
    lines = [
        "decision needed",
        item,
        "",
        "context",
        context,
        "",
        "options",
    ]
    for index, option in enumerate(options, start=1):
        lines.append(
            f"{index}. {option['label']} | upside: {option.get('upside', 'unknown')} | "
            f"downside: {option.get('downside', 'unknown')} | cost/time: {option.get('cost_time', 'unknown')} | "
            f"risk: {option.get('risk', 'unknown')}"
        )
    lines.extend(
        [
            "",
            "recommendation",
            recommendation,
            "",
            "approval request",
            "Reply: approve option N / draft different / defer until [date] / drop",
            "",
            "blocked actions",
            f"Nothing is sent, booked, bought, filed, posted, or otherwise executed for {category} until explicit approval.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def stage_decision_packet(
    *,
    ledger_path: Path,
    loop_id: int,
    item: str,
    context: str,
    options: list[dict[str, Any]],
    recommendation: str,
    category: str,
    risk_class: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    packet = render_decision_packet(
        item=item,
        context=context,
        options=options,
        recommendation=recommendation,
        category=category,
    )
    ledger = ActionLedger(ledger_path)
    record = ledger.add_action(
        scope="EA",
        summary=f"Decision packet: {item}",
        evidence_ids=[f"open-loop:{loop_id}"],
        allowed_next_actions=["approve_option", "draft_different", "defer", "drop"],
        status="approval_required",
        risk_class=risk_class,
        ttl_hours=0,
        now=now,
        executor_state={
            "schema": "torben.decision-packet.v1",
            "mutation_type": "decision_packet",
            "loop_id": loop_id,
            "category": category,
            "risk_class": risk_class,
            "options": options,
            "recommendation": recommendation,
            "packet": packet,
        },
    )
    return {"schema": "torben.decision-packet.v1", "record": record.to_dict(), "packet": packet}


def resolve_decision_reply(
    *,
    ledger_path: Path,
    reply_text: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    ledger = ActionLedger(ledger_path)
    resolution = ledger.resolve_reply(reply_text, now=now)
    if resolution.record is None or resolution.status not in {"resolved", "resolved_recent", "resolved_alias"}:
        return {"status": "not_resolved", "resolution": resolution.to_dict()}
    record = resolution.record
    state = record.executor_state or {}
    if state.get("schema") != "torben.decision-packet.v1":
        return {"status": "ignored", "reason": "not_decision_packet", "resolution": resolution.to_dict()}
    match = APPROVE_OPTION_RE.search(reply_text)
    if not match:
        return {"status": "ignored", "reason": "not_approve_option_reply", "resolution": resolution.to_dict()}
    option_index = int(match.group("option"))
    options = list(state.get("options") or [])
    if option_index < 1 or option_index > len(options):
        return {"status": "rejected", "reason": "option_out_of_range", "option": option_index}
    records = ledger.load()
    for existing in records:
        if existing.handle == record.handle:
            existing.status = "approved"
            existing.executor_state["selected_option"] = option_index
            existing.resolution_history.append(
                {
                    "at": _iso(now),
                    "status": "approved",
                    "selected_option": option_index,
                    "reason": "decision_packet_reply",
                }
            )
            break
    ledger.save(records)
    refreshed = ledger.get(record.handle)
    return {
        "status": "approved",
        "handle": record.handle,
        "selected_option": option_index,
        "record": refreshed.to_dict() if refreshed else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", help="Action ledger path")
    parser.add_argument("--json", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage = subparsers.add_parser("stage")
    stage.add_argument("--loop-id", type=int, required=True)
    stage.add_argument("--item", required=True)
    stage.add_argument("--context", required=True)
    stage.add_argument("--options-json", required=True)
    stage.add_argument("--recommendation", required=True)
    stage.add_argument("--category", required=True)
    stage.add_argument("--risk-class", required=True)

    resolve = subparsers.add_parser("resolve")
    resolve.add_argument("reply", nargs="+")
    args = parser.parse_args(argv)

    ledger_path = Path(args.ledger) if args.ledger else get_hermes_home() / "state" / "torben-action-ledger.jsonl"
    if args.command == "stage":
        options = json.loads(args.options_json)
        if not isinstance(options, list):
            raise ValueError("--options-json must be a JSON list")
        payload = stage_decision_packet(
            ledger_path=ledger_path,
            loop_id=args.loop_id,
            item=args.item,
            context=args.context,
            options=options,
            recommendation=args.recommendation,
            category=args.category,
            risk_class=args.risk_class,
        )
    elif args.command == "resolve":
        payload = resolve_decision_reply(ledger_path=ledger_path, reply_text=" ".join(args.reply))
    else:
        raise ValueError(f"Unhandled command: {args.command}")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
