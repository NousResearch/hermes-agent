"""Command line helpers for autonomous contract ledgers."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .compiler import compile_ledger_seed, generate_worker_packet, load_contract, parse_iso_datetime
from .ledger import (
    LedgerError,
    export_state,
    initialize_ledger,
    ready_sprints,
    record_cleanup_entry,
    resolve_gate,
    transition_sprint,
    update_cleanup_state,
    verify_contract_lock,
    write_projection_files,
)
from .models import CleanupRecord
from .schema import write_schema_files


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autonomous-contract", description="Autonomous contract PM ledger utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate executable contract YAML/JSON")
    validate.add_argument("contract")

    schemas = sub.add_parser("write-schemas", help="write canonical JSON schemas")
    schemas.add_argument("output_dir")

    compile_cmd = sub.add_parser("compile", help="compile contract to ledger seed JSON")
    compile_cmd.add_argument("contract")
    compile_cmd.add_argument("--output", required=True)
    compile_cmd.add_argument("--approved-by")
    compile_cmd.add_argument("--approved-at", help="ISO-8601 approval timestamp for deterministic lock output")

    init = sub.add_parser("init-ledger", help="initialize SQLite ledger from contract")
    init.add_argument("contract")
    init.add_argument("--db", required=True)
    init.add_argument("--approved-by", default="galt")
    init.add_argument("--force", action="store_true")
    init.add_argument("--projection-dir")

    packet = sub.add_parser("packet", help="generate a scoped worker packet")
    packet.add_argument("contract")
    packet.add_argument("sprint_id")
    packet.add_argument("--worker-role", required=True, choices=["implementer", "reviewer", "dogfood", "research"])
    packet.add_argument("--assigned-worker", required=True)
    packet.add_argument("--output", required=True)
    packet.add_argument("--session-id")
    packet.add_argument("--context-json", default="{}")

    state = sub.add_parser("state", help="export current ledger state")
    state.add_argument("--db", required=True)
    state.add_argument("--output")

    ready = sub.add_parser("ready", help="print ready sprint ids")
    ready.add_argument("--db", required=True)

    trans = sub.add_parser("transition", help="transition a sprint state")
    trans.add_argument("--db", required=True)
    trans.add_argument("--sprint", required=True)
    trans.add_argument(
        "--state",
        required=True,
        choices=[
            "not_started",
            "ready",
            "packet_generated",
            "dispatched",
            "in_progress",
            "review_required",
            "verification_required",
            "blocked_galt",
            "blocked_human",
            "failed",
            "completed",
            "completed_with_warnings",
            "skipped_by_galt_decision",
            "superseded",
        ],
    )
    trans.add_argument("--actor", required=True)
    trans.add_argument("--evidence-json", default="{}")
    trans.add_argument("--artifact-path")
    trans.add_argument("--projection-dir")

    gate = sub.add_parser("resolve-gate", help="resolve a ledger gate with evidence")
    gate.add_argument("--db", required=True)
    gate.add_argument("--gate", required=True)
    gate.add_argument("--actor", required=True)
    gate.add_argument("--evidence", action="append", required=True)
    gate.add_argument("--artifact-path")
    gate.add_argument("--projection-dir")

    cleanup = sub.add_parser("record-cleanup", help="record a cleanup item in the ledger")
    cleanup.add_argument("--db", required=True)
    cleanup.add_argument("--id", required=True)
    cleanup.add_argument("--type", required=True, choices=["tmux_session", "worktree", "process", "port", "discord_thread", "kanban_card", "cron_job", "container", "temp_file", "browser_session"])
    cleanup.add_argument("--sprint", required=True)
    cleanup.add_argument("--actor", required=True)
    cleanup.add_argument("--created-by")
    cleanup.add_argument("--state", default="active_needed", choices=["active_needed", "closed", "archived", "retained_with_reason", "orphaned_blocker"])
    cleanup.add_argument("--created-at", help="ISO-8601 created timestamp; defaults to now")
    cleanup.add_argument("--identifier", required=True)
    cleanup.add_argument("--owner", required=True)
    cleanup.add_argument("--close-condition", required=True)
    cleanup.add_argument("--notes")
    cleanup.add_argument("--projection-dir")

    cleanup_state = sub.add_parser("update-cleanup", help="update cleanup item state")
    cleanup_state.add_argument("--db", required=True)
    cleanup_state.add_argument("--id", required=True)
    cleanup_state.add_argument("--state", required=True, choices=["active_needed", "closed", "archived", "retained_with_reason", "orphaned_blocker"])
    cleanup_state.add_argument("--actor", required=True)
    cleanup_state.add_argument("--notes")
    cleanup_state.add_argument("--projection-dir")

    verify = sub.add_parser("verify-lock", help="verify a contract payload against ledger lock hash")
    verify.add_argument("contract")
    verify.add_argument("--db", required=True)

    project = sub.add_parser("write-projections", help="write current-state/sprint-ledger/events projections")
    project.add_argument("--db", required=True)
    project.add_argument("--output-dir", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            contract = load_contract(args.contract)
            print(f"OK {contract.contractId} {contract.contractVersion} sprints={len(contract.sprints)}")
            return 0
        if args.command == "write-schemas":
            written = write_schema_files(args.output_dir)
            for path in written:
                print(path)
            return 0
        if args.command == "compile":
            contract = load_contract(args.contract)
            approved_at = parse_iso_datetime(args.approved_at) if args.approved_at else None
            seed = compile_ledger_seed(contract, approved_by=args.approved_by, approved_at=approved_at)
            _write_json(Path(args.output), seed.model_dump(mode="json"))
            print(args.output)
            return 0
        if args.command == "init-ledger":
            contract = load_contract(args.contract)
            seed = compile_ledger_seed(contract, approved_by=args.approved_by)
            initialize_ledger(args.db, seed, actor=args.approved_by, force=args.force)
            if args.projection_dir:
                write_projection_files(args.db, args.projection_dir)
            print(args.db)
            return 0
        if args.command == "packet":
            context = json.loads(args.context_json)
            if not isinstance(context, dict):
                raise LedgerError("--context-json must decode to an object")
            contract = load_contract(args.contract)
            pkt = generate_worker_packet(
                contract,
                args.sprint_id,
                worker_role=args.worker_role,
                assigned_worker=args.assigned_worker,
                session_id=args.session_id,
                context=context,
            )
            _write_json(Path(args.output), pkt.model_dump(mode="json"))
            print(args.output)
            return 0
        if args.command == "state":
            state = export_state(args.db)
            if args.output:
                _write_json(Path(args.output), state)
                print(args.output)
            else:
                print(json.dumps(state, indent=2, sort_keys=True, ensure_ascii=False))
            return 0
        if args.command == "ready":
            for sprint_id in ready_sprints(args.db):
                print(sprint_id)
            return 0
        if args.command == "transition":
            evidence = json.loads(args.evidence_json)
            if not isinstance(evidence, dict):
                raise LedgerError("--evidence-json must decode to an object")
            transition_sprint(args.db, args.sprint, args.state, actor=args.actor, evidence=evidence, artifact_path=args.artifact_path)
            if args.projection_dir:
                write_projection_files(args.db, args.projection_dir)
            print(f"{args.sprint} -> {args.state}")
            return 0
        if args.command == "resolve-gate":
            resolve_gate(args.db, args.gate, actor=args.actor, evidence=args.evidence, artifact_path=args.artifact_path)
            if args.projection_dir:
                write_projection_files(args.db, args.projection_dir)
            print(args.gate)
            return 0
        if args.command == "record-cleanup":
            created_at = parse_iso_datetime(args.created_at).isoformat() if args.created_at else datetime.now(timezone.utc).isoformat()
            record = CleanupRecord(
                id=args.id,
                type=args.type,
                createdBy=args.created_by or args.actor,
                sprintId=args.sprint,
                createdAt=created_at,
                state=args.state,
                identifier=args.identifier,
                owner=args.owner,
                closeCondition=args.close_condition,
                notes=args.notes,
            )
            record_cleanup_entry(args.db, record, actor=args.actor)
            if args.projection_dir:
                write_projection_files(args.db, args.projection_dir)
            print(args.id)
            return 0
        if args.command == "update-cleanup":
            update_cleanup_state(args.db, args.id, args.state, actor=args.actor, notes=args.notes)
            if args.projection_dir:
                write_projection_files(args.db, args.projection_dir)
            print(f"{args.id} -> {args.state}")
            return 0
        if args.command == "verify-lock":
            contract = load_contract(args.contract)
            if not verify_contract_lock(args.db, contract):
                print("LOCK_MISMATCH", file=sys.stderr)
                return 2
            print("OK")
            return 0
        if args.command == "write-projections":
            for path in write_projection_files(args.db, args.output_dir):
                print(path)
            return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
