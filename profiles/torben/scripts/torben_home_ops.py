#!/usr/bin/env python3
"""Torben generic home-ops coordination over open loops."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from torben_open_loops import LoopRow, load_loops


SCHEMA = "torben.home-ops.v1"
HOME_OPS_DOMAINS = {"home", "admin"}
EXISTING_LADDER_CATEGORIES = {
    "booking",
    "form_filing",
    "payment_adjacent",
    "calendar_edit",
    "gmail_archive",
    "gmail_trash",
    "gtm_post",
}


def _iso(value: datetime | None = None) -> str:
    return (value or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def infer_ladder_category(proposed_action: str, explicit_category: str | None = None) -> str:
    if explicit_category:
        if explicit_category not in EXISTING_LADDER_CATEGORIES:
            raise ValueError(f"Unsupported ladder category: {explicit_category}")
        return explicit_category
    text = proposed_action.lower()
    if re.search(r"\b(book|schedule|appointment|service call|repair|provider|vendor)\b", text):
        return "booking"
    if re.search(r"\b(form|file|paperwork|submit|reimbursement)\b", text):
        return "form_filing"
    if re.search(r"\b(pay|refund|subscription|cancel|invoice|charge)\b", text):
        return "payment_adjacent"
    if re.search(r"\b(calendar|invite|reschedule|event)\b", text):
        return "calendar_edit"
    return "booking"


def load_loop_by_id(path: Path, loop_id: int) -> LoopRow:
    for row in load_loops(path):
        if row.id == loop_id:
            return row
    raise KeyError(f"Loop id not found: {loop_id}")


def build_home_ops_packet(
    *,
    loop: LoopRow,
    proposed_action: str,
    category: str | None = None,
    context: str = "",
    now: datetime | None = None,
) -> dict[str, Any]:
    if loop.domain not in HOME_OPS_DOMAINS:
        return {
            "schema": SCHEMA,
            "status": "refused",
            "reason": "loop_not_home_ops_domain",
            "loop": loop.to_dict(),
            "external_actions_taken": [],
        }
    ladder_category = infer_ladder_category(proposed_action, category)
    return {
        "schema": SCHEMA,
        "created_at": _iso(now),
        "status": "packet_only",
        "loop": loop.to_dict(),
        "category": ladder_category,
        "proposed_action": _clean_text(proposed_action),
        "context": _clean_text(context),
        "decision_options": [
            {
                "label": f"Proceed with: {_clean_text(proposed_action)}",
                "upside": "Moves the home/admin loop forward",
                "downside": "External action may need provider/source confirmation",
                "cost_time": "unknown",
                "risk": "medium" if ladder_category in {"booking", "form_filing"} else "high",
            },
            {
                "label": "Ask a clarifying question first",
                "upside": "Reduces ambiguity before action",
                "downside": "Adds one follow-up step",
                "cost_time": "5m",
                "risk": "low",
            },
        ],
        "approval_gate": {
            "requires_explicit_approval": True,
            "ladder_category": ladder_category,
            "allowed_without_approval": False,
        },
        "blocked_actions": ["no external action without approved decision packet"],
        "external_actions_taken": [],
    }


def stage_home_ops_decision(
    *,
    ledger_path: Path,
    packet: dict[str, Any],
) -> dict[str, Any]:
    from torben_decision_packet import stage_decision_packet

    if packet.get("status") != "packet_only":
        return {"schema": SCHEMA, "status": "refused", "reason": packet.get("reason") or "not_packet_only"}
    loop = packet.get("loop") or {}
    options = list(packet.get("decision_options") or [])
    return stage_decision_packet(
        ledger_path=ledger_path,
        loop_id=int(loop.get("id") or 0),
        item=f"Coordinate home/admin loop: {loop.get('item') or packet.get('proposed_action')}",
        context=packet.get("context") or packet.get("proposed_action") or "",
        options=options,
        recommendation=options[0]["label"] if options else "Proceed only after approval",
        category=packet.get("category") or "booking",
        risk_class=options[0].get("risk", "medium") if options else "medium",
    )


def gate_external_action(
    *,
    ledger_path: Path,
    handle: str | None,
    category: str,
) -> dict[str, Any]:
    from hermes_cli.signal_coo.action_ledger import ActionLedger
    from torben_autonomy_ladder import evaluate_dispatch

    if category not in EXISTING_LADDER_CATEGORIES:
        return {"schema": SCHEMA, "status": "refused", "reason": "unsupported_ladder_category", "category": category}
    status = "approval_required"
    if handle:
        record = ActionLedger(ledger_path).get(handle)
        if record is not None:
            status = record.status
    dispatch = evaluate_dispatch(category=category, item_status=status, requested_count=1, ledger_path=ledger_path)
    if status != "approved":
        return {
            "schema": SCHEMA,
            "status": "blocked",
            "reason": "explicit_approval_required",
            "category": category,
            "dispatch": dispatch,
            "allowed": False,
        }
    return {
        "schema": SCHEMA,
        "status": "approved_to_execute",
        "category": category,
        "dispatch": dispatch,
        "allowed": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    packet = subparsers.add_parser("packet")
    packet.add_argument("--loops", required=True)
    packet.add_argument("--loop-id", type=int, required=True)
    packet.add_argument("--proposed-action", required=True)
    packet.add_argument("--category")
    packet.add_argument("--context", default="")

    stage = subparsers.add_parser("stage")
    stage.add_argument("--ledger", required=True)
    stage.add_argument("--packet-json", required=True)

    gate = subparsers.add_parser("gate")
    gate.add_argument("--ledger", required=True)
    gate.add_argument("--category", required=True)
    gate.add_argument("--handle")
    args = parser.parse_args(argv)

    if args.command == "packet":
        payload = build_home_ops_packet(
            loop=load_loop_by_id(Path(args.loops), args.loop_id),
            proposed_action=args.proposed_action,
            category=args.category,
            context=args.context,
        )
    elif args.command == "stage":
        payload = stage_home_ops_decision(ledger_path=Path(args.ledger), packet=json.loads(args.packet_json))
    elif args.command == "gate":
        payload = gate_external_action(ledger_path=Path(args.ledger), handle=args.handle, category=args.category)
    else:
        raise ValueError(f"Unhandled command: {args.command}")

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
