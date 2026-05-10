"""CLI surface for read-only run ledger retrieval."""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from typing import Any

from agent.run_ledger_reader import (
    DEFAULT_EVENT_LIMIT,
    RunLedgerReadError,
    fetch_run_events,
    list_run_ledgers,
    read_run_capsule,
    recover_run,
)


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))


def _print_runs_human(payload: dict[str, Any]) -> None:
    runs = payload["runs"]
    if not runs:
        print("No run ledgers found.")
        return
    for run in runs:
        print(
            f"{run['run_id']}  events={run['event_count']}  "
            f"last={run.get('last_event_id') or '-'}:{run.get('last_event_type') or '-'}  "
            f"in_flight={run['in_flight_count']}  corrupt={run['corrupt_line_count']}"
        )
        if run.get("latest_capsule"):
            print(f"  capsule: {run['latest_capsule']}")
        print(f"  root: {run['run_root']}")


def _print_events_human(payload: dict[str, Any]) -> None:
    for event in payload["events"]:
        print(
            f"{event.get('event_id') or '-'}  {event.get('type') or '-'}  "
            f"tool={event.get('tool_name') or '-'}  status={event.get('status') or '-'}"
        )
    if payload["truncated"]:
        print(f"truncated: next_start={payload.get('next_start')}")
    if payload["corrupt_lines"]:
        print(f"corrupt lines: {len(payload['corrupt_lines'])}")


def _print_capsule_human(payload: dict[str, Any]) -> None:
    capsule = payload["capsule"]
    print(f"{capsule.get('capsule_id') or payload['relative_path']}")
    print(f"run: {payload['run_id']}")
    print(f"path: {payload['relative_path']}")
    span = capsule.get("event_span") or {}
    print(f"event span: {span.get('start_event_id') or span.get('start_seq') or '-'}..{span.get('end_event_id') or span.get('end_seq') or '-'}")
    if capsule.get("next_action"):
        print(f"next action: {capsule['next_action']}")
    if capsule.get("blockers"):
        print(f"blockers: {len(capsule['blockers'])}")
    print(f"in flight: {len(capsule.get('in_flight') or {})}")
    print(f"recent completed: {len(capsule.get('recent_completed_tools') or [])}")
    print(f"artifact refs: {len(capsule.get('artifact_refs') or [])}")


def _print_recovery_human(payload: dict[str, Any]) -> None:
    recovery = payload["recovery"]
    print(f"run: {payload['run_id']}")
    print(f"root: {payload['run_root']}")
    print(f"in flight: {len(recovery['in_flight'])}")
    for call_id, event in recovery["in_flight"].items():
        print(f"  {call_id}: {event.get('tool_name') or '-'} started at {event.get('event_id') or '-'}")
    print(f"recent completed: {len(recovery['recent_completed_tools'])}")
    for item in recovery["recent_completed_tools"]:
        print(
            f"  {item.get('tool_call_id') or '-'}: {item.get('tool_name') or '-'} "
            f"{item.get('status') or '-'} at {item.get('event_id') or '-'}"
        )
    print(f"artifact refs: {len(recovery['artifact_refs'])}")
    print(f"corrupt lines: {len(payload['corrupt_lines'])}")


def runs_command(args: Namespace) -> int:
    try:
        action = getattr(args, "runs_action", None)
        if action == "list":
            payload = {
                "runs": list_run_ledgers(
                    limit=getattr(args, "limit", None),
                    lock_timeout_seconds=getattr(args, "lock_timeout", 2.0),
                )
            }
            if getattr(args, "json", False):
                _print_json(payload)
            else:
                _print_runs_human(payload)
            return 0

        if action == "events":
            filters = {
                "type": getattr(args, "type", None),
                "tool_name": getattr(args, "tool", None),
                "session_id": getattr(args, "session", None),
                "status": getattr(args, "status", None),
            }
            payload = fetch_run_events(
                args.span,
                filters=filters,
                limit=getattr(args, "limit", DEFAULT_EVENT_LIMIT),
                lock_timeout_seconds=getattr(args, "lock_timeout", 2.0),
            )
            if getattr(args, "json", False):
                _print_json(payload)
            else:
                _print_events_human(payload)
            return 0

        if action == "capsule":
            payload = read_run_capsule(
                args.run_id,
                latest=getattr(args, "latest", False),
                capsule=getattr(args, "capsule", None),
            )
            if getattr(args, "json", False):
                _print_json(payload)
            else:
                _print_capsule_human(payload)
            return 0

        if action == "recover":
            payload = recover_run(
                args.run_id,
                max_completed=getattr(args, "limit", DEFAULT_EVENT_LIMIT),
                lock_timeout_seconds=getattr(args, "lock_timeout", 2.0),
            )
            if getattr(args, "json", False):
                _print_json(payload)
            else:
                _print_recovery_human(payload)
            return 0

        print("Error: missing runs subcommand", file=sys.stderr)
        return 2
    except RunLedgerReadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
