"""Operator and agent CLI for engineering evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .contracts import workflow_contract
from .correlation import build_deployment_correlation_receipt
from .experience import (
    promote_experience_candidate,
    search_experience_candidates,
    store_experience_candidate,
)
from .kanban import attach_receipt_to_task, list_task_receipts
from .receipts import ReceiptValidationError, validate_exact_head, validate_receipt
from .scanner import build_codebase_snapshot
from .testing import build_test_discovery_receipt
from .workflow import create_evidence_children, evaluate_issue_to_pr_gate


def register_cli(subparser: argparse.ArgumentParser) -> None:
    subs = subparser.add_subparsers(dest="engineering_evidence_action")

    snapshot = subs.add_parser("snapshot", help="Create a read-only exact-HEAD codebase snapshot")
    snapshot.add_argument("--repo", required=True, type=Path)
    snapshot.add_argument("--query", default="")
    snapshot.add_argument("--output", type=Path)

    validate = subs.add_parser("validate", help="Validate a receipt JSON file")
    validate.add_argument("--input", required=True, type=Path)
    validate.add_argument("--current-head", help="Reject the receipt if it is stale")

    test = subs.add_parser("test-receipt", help="Create a test-discovery receipt from JSON inputs")
    test.add_argument("--repository", required=True)
    test.add_argument("--head-sha", required=True)
    test.add_argument("--target", required=True)
    test.add_argument("--cases", required=True, type=Path)
    test.add_argument("--commands", required=True, type=Path)
    test.add_argument("--task-id")
    test.add_argument("--output", type=Path)

    experience = subs.add_parser("experience", help="Store a sanitized diagnostic learning candidate")
    experience.add_argument("--vault", type=Path, default=Path(os.environ.get("HERMES_LEARNING_VAULT", "~/LunaBotVault")))
    experience.add_argument("--repository", required=True)
    experience.add_argument("--head-sha", required=True)
    experience.add_argument("--task-id")
    experience.add_argument("--problem", required=True)
    experience.add_argument("--solution", required=True)
    experience.add_argument("--outcome", required=True, choices=["resolved", "partially-resolved", "blocked", "not-reproduced"])
    experience.add_argument("--affected-file", action="append", default=[])
    experience.add_argument("--failure", action="append", default=[])
    experience.add_argument("--test-json", type=Path)

    promote = subs.add_parser("experience-promote", help="Validate a candidate without changing it")
    promote.add_argument("--input", required=True, type=Path)
    promote.add_argument("--validator-role", default="memory-validator")
    promote.add_argument("--rationale", required=True)

    search = subs.add_parser("experience-search", help="Retrieve validated exact-HEAD experience")
    search.add_argument("--vault", type=Path, default=Path(os.environ.get("HERMES_LEARNING_VAULT", "~/LunaBotVault")))
    search.add_argument("--repository", required=True)
    search.add_argument("--head-sha", required=True)
    search.add_argument("--term", action="append", default=[])
    search.add_argument("--affected-file", action="append", default=[])
    search.add_argument("--limit", type=int, default=10)

    subs.add_parser("contract", help="Show receipt ownership and typed events")

    correlate = subs.add_parser("correlate", help="Create a diagnostic deployment correlation receipt")
    correlate.add_argument("--input", required=True, type=Path)

    gate = subs.add_parser("pr-gate", help="Evaluate the fail-closed issue-to-PR admission gate")
    gate.add_argument("--input", required=True, type=Path)

    attach = subs.add_parser("kanban-attach", help="Attach a validated receipt to an existing Kanban task")
    attach.add_argument("--task-id", required=True)
    attach.add_argument("--input", required=True, type=Path)
    attach.add_argument("--db", type=Path)
    attach.add_argument("--board")
    attach.add_argument("--uploaded-by", default="engineering-evidence")

    task_receipts = subs.add_parser("kanban-receipts", help="List valid engineering receipts on a Kanban task")
    task_receipts.add_argument("--task-id", required=True)
    task_receipts.add_argument("--db", type=Path)
    task_receipts.add_argument("--board")

    workflow = subs.add_parser("kanban-workflow", help="Create an explicit bounded evidence child graph")
    workflow.add_argument("--parent-task-id", required=True)
    workflow.add_argument("--repository", required=True)
    workflow.add_argument("--head-sha", required=True)
    workflow.add_argument("--db", type=Path)
    workflow.add_argument("--board")

    subparser.set_defaults(func=engineering_evidence_command)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_or_print(payload: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if output:
        output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


def engineering_evidence_command(args: argparse.Namespace) -> int:
    action = getattr(args, "engineering_evidence_action", None)
    try:
        if action == "snapshot":
            _write_or_print(build_codebase_snapshot(args.repo, query=args.query), args.output)
            return 0
        if action == "validate":
            receipt = _load_json(args.input)
            receipt = (
                validate_exact_head(receipt, args.current_head)
                if args.current_head
                else validate_receipt(receipt)
            )
            print(json.dumps({"valid": True, "receipt_id": receipt["receipt_id"]}, sort_keys=True))
            return 0
        if action == "test-receipt":
            receipt = build_test_discovery_receipt(
                repository=args.repository,
                head_sha=args.head_sha,
                target=args.target,
                cases=_load_json(args.cases),
                commands=_load_json(args.commands),
                task_id=args.task_id,
            )
            _write_or_print(receipt, args.output)
            return 0
        if action == "experience":
            tests = _load_json(args.test_json) if args.test_json else []
            path = store_experience_candidate(
                args.vault,
                {
                    "repository": args.repository,
                    "head_sha": args.head_sha,
                    "task_id": args.task_id,
                    "problem": args.problem,
                    "affected_files": args.affected_file,
                    "failure_mechanisms": args.failure,
                    "solution": args.solution,
                    "tests": tests,
                    "outcome": args.outcome,
                },
            )
            print(json.dumps({"stored": True, "path": str(path)}, sort_keys=True))
            return 0
        if action == "experience-promote":
            promotion = promote_experience_candidate(
                args.input,
                validator_role=args.validator_role,
                rationale=args.rationale,
            )
            print(json.dumps(promotion, sort_keys=True))
            return 0
        if action == "experience-search":
            matches = search_experience_candidates(
                args.vault,
                args.repository,
                args.head_sha,
                terms=args.term,
                affected_files=args.affected_file,
                limit=args.limit,
            )
            print(json.dumps({"matches": matches}, indent=2, sort_keys=True))
            return 0
        if action == "contract":
            print(json.dumps(workflow_contract(), indent=2, sort_keys=True))
            return 0
        if action == "correlate":
            receipt = build_deployment_correlation_receipt(**_load_json(args.input))
            _write_or_print(receipt, None)
            return 0
        if action == "pr-gate":
            decision = evaluate_issue_to_pr_gate(**_load_json(args.input))
            print(json.dumps(decision, indent=2, sort_keys=True))
            return 0
        if action == "kanban-attach":
            attachment = attach_receipt_to_task(
                _load_json(args.input),
                task_id=args.task_id,
                db_path=args.db,
                board=args.board,
                uploaded_by=args.uploaded_by,
            )
            print(json.dumps({"attached": True, "attachment": attachment}, sort_keys=True))
            return 0
        if action == "kanban-receipts":
            receipts = list_task_receipts(task_id=args.task_id, db_path=args.db, board=args.board)
            print(json.dumps({"receipts": receipts}, indent=2, sort_keys=True))
            return 0
        if action == "kanban-workflow":
            result = create_evidence_children(
                parent_id=args.parent_task_id,
                repository=args.repository,
                head_sha=args.head_sha,
                db_path=args.db,
                board=args.board,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        print("Usage: hermes engineering-evidence {snapshot|validate|test-receipt|experience|experience-promote|experience-search|contract|correlate|pr-gate|kanban-attach|kanban-receipts|kanban-workflow}")
        return 2
    except (OSError, ValueError, ReceiptValidationError, json.JSONDecodeError) as exc:
        print(json.dumps({"stored": False, "error": str(exc)}, sort_keys=True))
        return 1
