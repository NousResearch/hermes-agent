#!/usr/bin/env python3
"""Unified Torben mutation spine for cron and agent-session executors."""

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
    return Path("/Users/ericfreeman/.hermes/hermes-agent")


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hermes_constants import get_hermes_home  # noqa: E402
from hermes_cli.signal_coo.action_ledger import ActionLedger, ActionRecord  # noqa: E402


SCHEMA = "torben.mutation-spine.v1"
EXECUTORS = {"cron", "agent"}
CATEGORIES = {
    "gmail_archive",
    "gmail_trash",
    "calendar_edit",
    "booking",
    "form_filing",
    "gtm_post",
    "payment_adjacent",
}
RISK_CLASSES = {"low", "medium", "high"}
DEFAULT_UNDO_CAPABLE = {
    "gmail_archive": True,
    "gmail_trash": True,
    "calendar_edit": True,
    "booking": False,
    "form_filing": False,
    "gtm_post": False,
    "payment_adjacent": False,
}


def _iso(value: datetime | None = None) -> str:
    return (value or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate(category: str, executor: str, risk_class: str) -> None:
    if category not in CATEGORIES:
        raise ValueError(f"Unknown mutation category: {category}")
    if executor not in EXECUTORS:
        raise ValueError(f"Unknown mutation executor: {executor}")
    if risk_class not in RISK_CLASSES:
        raise ValueError(f"Unknown mutation risk class: {risk_class}")


def mutation_executor_state(
    *,
    category: str,
    executor: str,
    undo_pointer: str | None,
    surface: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "mutation_type": "external_mutation",
        "category": category,
        "executor": executor,
        "undo": undo_pointer,
        "surface": surface,
        "metadata": metadata or {},
    }


def record_mutation(
    *,
    ledger_path: Path,
    category: str,
    executor: str,
    risk_class: str,
    summary: str,
    undo_pointer: str | None,
    surface: str | None = None,
    metadata: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    _validate(category, executor, risk_class)
    ledger = ActionLedger(ledger_path)
    record = ledger.add_action(
        scope="EA",
        summary=summary,
        evidence_ids=[],
        allowed_next_actions=["audit"],
        status="executed",
        risk_class=risk_class,
        ttl_hours=0,
        now=now,
        executor_state=mutation_executor_state(
            category=category,
            executor=executor,
            undo_pointer=undo_pointer,
            surface=surface,
            metadata=metadata,
        ),
    )
    return {"schema": SCHEMA, "record": record.to_dict()}


def mutation_record_view(record: ActionRecord) -> dict[str, Any] | None:
    state = record.executor_state or {}
    if state.get("schema") != SCHEMA:
        return None
    return {
        "handle": record.handle,
        "category": state.get("category"),
        "executor": state.get("executor"),
        "risk_class": record.risk_class,
        "undo": state.get("undo"),
        "surface": state.get("surface"),
        "status": record.status,
    }


def evaluate_promotion_undo_gate(
    *,
    category: str,
    target_rung: str,
    undo_capable: bool | None = None,
) -> dict[str, Any]:
    if category not in CATEGORIES:
        raise ValueError(f"Unknown mutation category: {category}")
    capable = DEFAULT_UNDO_CAPABLE[category] if undo_capable is None else bool(undo_capable)
    if target_rung == "auto_within_caps" and not capable:
        return {
            "status": "refused",
            "category": category,
            "target_rung": target_rung,
            "reason": "surface_has_no_real_undo",
        }
    return {
        "status": "allowed",
        "category": category,
        "target_rung": target_rung,
        "reason": "undo_capable_or_not_top_rung",
    }


def list_mutations(*, ledger_path: Path) -> list[dict[str, Any]]:
    ledger = ActionLedger(ledger_path)
    return [view for record in ledger.load() if (view := mutation_record_view(record)) is not None]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", help="Action ledger path")
    parser.add_argument("--json", action="store_true", help="Print JSON")
    subparsers = parser.add_subparsers(dest="command", required=True)

    record_parser = subparsers.add_parser("record", help="Append one mutation-spine record")
    record_parser.add_argument("--category", required=True, choices=sorted(CATEGORIES))
    record_parser.add_argument("--executor", required=True, choices=sorted(EXECUTORS))
    record_parser.add_argument("--risk-class", required=True, choices=sorted(RISK_CLASSES))
    record_parser.add_argument("--summary", required=True)
    record_parser.add_argument("--undo")
    record_parser.add_argument("--surface")
    record_parser.add_argument("--metadata-json", default="{}")

    promote_parser = subparsers.add_parser("promotion-gate", help="Evaluate no-undo promotion gate")
    promote_parser.add_argument("--category", required=True, choices=sorted(CATEGORIES))
    promote_parser.add_argument("--target-rung", required=True)
    promote_parser.add_argument("--undo-capable", choices=("true", "false"))

    subparsers.add_parser("list", help="List mutation-spine records")
    args = parser.parse_args(argv)

    ledger_path = Path(args.ledger) if args.ledger else get_hermes_home() / "state" / "torben-action-ledger.jsonl"
    if args.command == "record":
        metadata = json.loads(args.metadata_json or "{}")
        if not isinstance(metadata, dict):
            raise ValueError("--metadata-json must be a JSON object")
        payload = record_mutation(
            ledger_path=ledger_path,
            category=args.category,
            executor=args.executor,
            risk_class=args.risk_class,
            summary=args.summary,
            undo_pointer=args.undo,
            surface=args.surface,
            metadata=metadata,
        )
    elif args.command == "promotion-gate":
        undo_capable = None if args.undo_capable is None else args.undo_capable == "true"
        payload = evaluate_promotion_undo_gate(
            category=args.category,
            target_rung=args.target_rung,
            undo_capable=undo_capable,
        )
    elif args.command == "list":
        payload = {"schema": SCHEMA, "records": list_mutations(ledger_path=ledger_path)}
    else:
        raise ValueError(f"Unhandled command: {args.command}")

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
