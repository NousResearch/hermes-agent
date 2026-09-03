"""``hermes task`` subcommand — the task-ownership controller.

Durable local tracking for tasks an agent or operator owns end to end:
state, next action, blockers/decisions, bounded retries with fallback
recording, idempotent external receipts, and verification evidence gating
completion. See ``hermes_cli/task_ownership_db.py`` for the storage engine
and state machine, and ``docs/task-ownership-controller.md`` for the
operator guide.

This module intentionally has no side effects at import time — main.py
wires the argparse subparsers on demand (same convention as curator.py).

Feature flag: ``task_ownership.enabled`` in config.yaml, default False.
Explicit CRUD commands (create/update/outcome/receipt/verify/done/...)
always work — they are deliberate operator/worker actions, not passive
automation. Only ``hermes task age-check`` (the one command that runs
unattended, e.g. from cron) is gated: with the flag off it is a strict
no-op (shadow mode) — it prints nothing and mutates nothing — so wiring
the cron job ahead of opting in can never spam or silently rot state.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from hermes_cli import task_ownership_db as tdb


def _is_enabled() -> bool:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
    except Exception:
        return False
    return bool(cfg.get("task_ownership", {}).get("enabled", False))


def _aging_hours() -> tuple:
    try:
        from hermes_cli.config import load_config

        cfg = load_config().get("task_ownership", {})
    except Exception:
        cfg = {}
    return (
        int(cfg.get("aging_warn_hours", 24) or 24),
        int(cfg.get("aging_stale_hours", 72) or 72),
    )


def _default_max_retries() -> int:
    try:
        from hermes_cli.config import load_config

        cfg = load_config().get("task_ownership", {})
    except Exception:
        cfg = {}
    return int(cfg.get("default_max_retries", 3) or 3)


def _print_task(task: Dict[str, Any], *, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(task, indent=2, sort_keys=True))
        return
    print(f"{task['id']}  [{task['state']}]  {task['title']}")
    if task.get("owner"):
        print(f"  owner: {task['owner']}")
    if task.get("next_action"):
        print(f"  next_action: {task['next_action']}")
    if task.get("blocker"):
        print(f"  blocker: {task['blocker']}")
    if task.get("decision"):
        print(f"  decision: {task['decision']}")
    if task.get("fallback"):
        print(f"  fallback: {task['fallback']}")
    print(f"  retries: {task['retry_count']}/{task['max_retries']}")
    if task.get("approval_required"):
        approved = f"by {task['approved_by']} at {task['approved_at']}" if task.get(
            "approved_by"
        ) else "PENDING"
        print(f"  approval: {approved}")
    if task.get("verification_evidence"):
        print(f"  verified: {task['verification_evidence']} (at {task['verified_at']})")
    print(f"  created_at: {task['created_at']}  updated_at: {task['updated_at']}")


def _cmd_init(args) -> int:
    path = tdb.init_db()
    print(f"task-ownership db ready at {path}")
    return 0


def _cmd_create(args) -> int:
    conn = tdb.connect()
    task = tdb.create_task(
        conn,
        title=args.title,
        next_action=args.next_action,
        owner=args.owner,
        max_retries=args.max_retries if args.max_retries is not None else _default_max_retries(),
        approval_required=args.approval_required,
    )
    _print_task(task, as_json=args.json)
    return 0


def _cmd_list(args) -> int:
    conn = tdb.connect()
    tasks = tdb.list_tasks(conn, state=args.state, include_terminal=args.all)
    if args.json:
        print(json.dumps(tasks, indent=2, sort_keys=True))
        return 0
    if not tasks:
        print("no tasks")
        return 0
    for task in tasks:
        print(f"{task['id']}  [{task['state']:<16}]  {task['title']}")
    return 0


def _cmd_show(args) -> int:
    conn = tdb.connect()
    try:
        task = tdb.get_task(conn, args.task_id)
    except tdb.TaskNotFoundError:
        print(f"no such task: {args.task_id}", file=sys.stderr)
        return 1
    _print_task(task, as_json=args.json)
    return 0


def _cmd_update(args) -> int:
    conn = tdb.connect()
    try:
        task = tdb.update_task(
            conn,
            args.task_id,
            next_action=args.next_action,
            blocker=args.blocker,
            decision=args.decision,
            owner=args.owner,
            fallback=args.fallback,
            max_retries=args.max_retries,
            state=args.state,
        )
    except tdb.TaskNotFoundError:
        print(f"no such task: {args.task_id}", file=sys.stderr)
        return 1
    except tdb.InvalidTransitionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _print_task(task, as_json=args.json)
    return 0


def _cmd_outcome(args) -> int:
    conn = tdb.connect()
    try:
        task = tdb.record_outcome(
            conn,
            args.task_id,
            result=args.result,
            detail=args.detail,
            retry=args.retry,
            fallback=args.fallback,
        )
    except tdb.TaskNotFoundError:
        print(f"no such task: {args.task_id}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _print_task(task, as_json=args.json)
    return 0


def _cmd_receipt(args) -> int:
    conn = tdb.connect()
    try:
        receipt = tdb.record_receipt(
            conn,
            args.task_id,
            receipt_id=args.receipt_id,
            source=args.source,
            payload=args.payload,
        )
    except tdb.TaskNotFoundError:
        print(f"no such task: {args.task_id}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    elif receipt["duplicate"]:
        print(f"receipt {args.receipt_id} already recorded for {args.task_id} — no-op")
    else:
        print(f"receipt {args.receipt_id} recorded for {args.task_id}")
    return 0


def _cmd_verify(args) -> int:
    conn = tdb.connect()
    try:
        task = tdb.record_verification(conn, args.task_id, args.evidence)
    except tdb.TaskNotFoundError:
        print(f"no such task: {args.task_id}", file=sys.stderr)
        return 1
    except (ValueError, tdb.InvalidTransitionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _print_task(task, as_json=args.json)
    return 0


def _cmd_done(args) -> int:
    conn = tdb.connect()
    try:
        task = tdb.mark_done(conn, args.task_id, evidence=args.evidence)
    except tdb.TaskNotFoundError:
        print(f"no such task: {args.task_id}", file=sys.stderr)
        return 1
    except (
        tdb.VerificationRequiredError,
        tdb.ApprovalRequiredError,
        tdb.InvalidTransitionError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _print_task(task, as_json=args.json)
    return 0


def _cmd_approve(args) -> int:
    conn = tdb.connect()
    try:
        task = tdb.approve_task(conn, args.task_id, args.by)
    except tdb.TaskNotFoundError:
        print(f"no such task: {args.task_id}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _print_task(task, as_json=args.json)
    return 0


def _cmd_block(args) -> int:
    conn = tdb.connect()
    try:
        task = tdb.block_task(conn, args.task_id, args.reason)
    except tdb.TaskNotFoundError:
        print(f"no such task: {args.task_id}", file=sys.stderr)
        return 1
    except tdb.InvalidTransitionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _print_task(task, as_json=args.json)
    return 0


def _cmd_cancel(args) -> int:
    conn = tdb.connect()
    try:
        task = tdb.cancel_task(conn, args.task_id, args.reason)
    except tdb.TaskNotFoundError:
        print(f"no such task: {args.task_id}", file=sys.stderr)
        return 1
    except tdb.InvalidTransitionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _print_task(task, as_json=args.json)
    return 0


def _cmd_events(args) -> int:
    conn = tdb.connect()
    try:
        tdb.get_task(conn, args.task_id)
    except tdb.TaskNotFoundError:
        print(f"no such task: {args.task_id}", file=sys.stderr)
        return 1
    events = tdb.list_events(conn, args.task_id)
    if args.json:
        print(json.dumps(events, indent=2, sort_keys=True))
        return 0
    for ev in events:
        arrow = f"{ev['from_state']} -> {ev['to_state']}" if ev["to_state"] else ""
        detail = f"  ({ev['detail']})" if ev["detail"] else ""
        print(f"{ev['created_at']}  {ev['event']:<22} {arrow}{detail}")
    return 0


def _cmd_age_check(args) -> int:
    enabled = _is_enabled()
    warn_hours, stale_hours = _aging_hours()
    conn = tdb.connect()

    if not enabled and not args.dry_run:
        # Shadow mode: strictly inert. No stdout (so a cron --no-agent job
        # wired ahead of opt-in stays silent), no mutation.
        if args.verbose:
            preview = tdb.age_check(
                conn,
                enabled=False,
                dry_run=True,
                warn_hours=warn_hours,
                stale_hours=stale_hours,
            )
            print(
                f"task_ownership disabled (shadow mode) — {len(preview)} task(s) would "
                "age; run `hermes task enable` to activate",
                file=sys.stderr,
            )
        return 0

    flagged = tdb.age_check(
        conn,
        enabled=enabled,
        dry_run=args.dry_run,
        warn_hours=warn_hours,
        stale_hours=stale_hours,
    )
    if args.json:
        print(json.dumps(flagged, indent=2, sort_keys=True))
        return 0
    if not flagged:
        return 0
    prefix = "[DRY RUN] " if args.dry_run else ""
    for item in flagged:
        print(
            f"{prefix}{item['tier']:<4} {item['task_id']}  {item['age_hours']}h  "
            f"[{item['state']}]  {item['title']}"
        )
    return 0


def _cmd_enable(args) -> int:
    from hermes_cli.config import set_config_value

    set_config_value("task_ownership.enabled", "true")
    print("task-ownership controller: enabled")
    return 0


def _cmd_disable(args) -> int:
    from hermes_cli.config import set_config_value

    set_config_value("task_ownership.enabled", "false")
    print("task-ownership controller: disabled (age-check is now a strict no-op)")
    return 0


def _cmd_status(args) -> int:
    conn = tdb.connect()
    tasks = tdb.list_tasks(conn, include_terminal=True)
    counts: Dict[str, int] = {}
    for task in tasks:
        counts[task["state"]] = counts.get(task["state"], 0) + 1
    enabled = _is_enabled()
    warn_hours, stale_hours = _aging_hours()
    if args.json:
        print(
            json.dumps(
                {
                    "enabled": enabled,
                    "aging_warn_hours": warn_hours,
                    "aging_stale_hours": stale_hours,
                    "total_tasks": len(tasks),
                    "by_state": counts,
                    "db_path": str(tdb.db_path()),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    print(f"task-ownership controller: {'enabled' if enabled else 'disabled (shadow mode)'}")
    print(f"aging thresholds: warn={warn_hours}h stale={stale_hours}h")
    print(f"db: {tdb.db_path()}")
    print(f"total tasks: {len(tasks)}")
    for state in tdb.STATES:
        if counts.get(state):
            print(f"  {state:<18} {counts[state]}")
    return 0


def register_cli(parent: argparse.ArgumentParser) -> None:
    """Attach `task` subcommands to *parent*.

    main.py calls this with the ArgumentParser returned by
    ``subparsers.add_parser("task", ...)``.
    """
    parent.set_defaults(func=_cmd_status)
    subs = parent.add_subparsers(dest="task_command")

    def _add_json(p):
        p.add_argument("--json", action="store_true", help="Emit JSON instead of a table")

    p_init = subs.add_parser("init", help="Create task_ownership.db if missing (idempotent)")
    p_init.set_defaults(func=_cmd_init)

    p_status = subs.add_parser("status", help="Show feature-flag state and task counts")
    _add_json(p_status)
    p_status.set_defaults(func=_cmd_status)

    p_create = subs.add_parser("create", help="Create a new task (state=NEW)")
    p_create.add_argument("title", help="Task title / description")
    p_create.add_argument("--next-action", dest="next_action", help="What to do next")
    p_create.add_argument("--owner", help="Who/what owns this task")
    p_create.add_argument(
        "--max-retries", dest="max_retries", type=int, default=None,
        help="Bounded retry limit before auto-BLOCKED (default from config, 3)",
    )
    p_create.add_argument(
        "--approval-required", dest="approval_required", action="store_true",
        help="Require `hermes task approve` before this task can be marked DONE",
    )
    _add_json(p_create)
    p_create.set_defaults(func=_cmd_create)

    p_list = subs.add_parser("list", help="List tasks")
    p_list.add_argument("--state", choices=sorted(tdb.STATES), help="Filter by exact state")
    p_list.add_argument(
        "--all", action="store_true",
        help="Include DONE/CANCELLED tasks (excluded by default when --state is not given)",
    )
    _add_json(p_list)
    p_list.set_defaults(func=_cmd_list)

    p_show = subs.add_parser("show", help="Show one task")
    p_show.add_argument("task_id")
    _add_json(p_show)
    p_show.set_defaults(func=_cmd_show)

    p_update = subs.add_parser("update", help="Update task fields and/or explicit state")
    p_update.add_argument("task_id")
    p_update.add_argument("--next-action", dest="next_action")
    p_update.add_argument("--blocker", dest="blocker")
    p_update.add_argument("--decision", dest="decision")
    p_update.add_argument("--owner", dest="owner")
    p_update.add_argument("--fallback", dest="fallback")
    p_update.add_argument("--max-retries", dest="max_retries", type=int)
    p_update.add_argument(
        "--state", dest="state", choices=sorted(tdb.STATES),
        help="Explicit state transition, validated against the state machine",
    )
    _add_json(p_update)
    p_update.set_defaults(func=_cmd_update)

    p_outcome = subs.add_parser("outcome", help="Record a worker attempt outcome")
    p_outcome.add_argument("task_id")
    p_outcome.add_argument("--result", required=True, choices=("success", "failure", "partial"))
    p_outcome.add_argument("--detail", help="Free-text detail about the attempt")
    p_outcome.add_argument(
        "--retry", action="store_true",
        help="On failure, bump retry_count and move to RETRYING (or BLOCKED once max-retries is exceeded)",
    )
    p_outcome.add_argument(
        "--fallback", help="Fallback plan to record if the retry limit is exceeded by this outcome",
    )
    _add_json(p_outcome)
    p_outcome.set_defaults(func=_cmd_outcome)

    p_receipt = subs.add_parser(
        "receipt", help="Record an idempotent external receipt (safe to retry)"
    )
    p_receipt.add_argument("task_id")
    p_receipt.add_argument("--receipt-id", dest="receipt_id", required=True, help="External idempotency key")
    p_receipt.add_argument("--source", help="Where the receipt came from (e.g. stripe, email)")
    p_receipt.add_argument("--payload", help="Free-text/JSON payload to store alongside the receipt")
    _add_json(p_receipt)
    p_receipt.set_defaults(func=_cmd_receipt)

    p_verify = subs.add_parser("verify", help="Record verification evidence (moves to VERIFYING)")
    p_verify.add_argument("task_id")
    p_verify.add_argument("--evidence", required=True, help="Concrete evidence the task's outcome was checked")
    _add_json(p_verify)
    p_verify.set_defaults(func=_cmd_verify)

    p_done = subs.add_parser(
        "done", help="Mark a task DONE — refuses without verification evidence on file"
    )
    p_done.add_argument("task_id")
    p_done.add_argument(
        "--evidence", help="Verification evidence, if not already recorded via `task verify`"
    )
    _add_json(p_done)
    p_done.set_defaults(func=_cmd_done)

    p_approve = subs.add_parser("approve", help="Record approval for an approval-required task")
    p_approve.add_argument("task_id")
    p_approve.add_argument("--by", required=True, help="Who approved it")
    _add_json(p_approve)
    p_approve.set_defaults(func=_cmd_approve)

    p_block = subs.add_parser("block", help="Move a task to BLOCKED with a reason")
    p_block.add_argument("task_id")
    p_block.add_argument("--reason", required=True)
    _add_json(p_block)
    p_block.set_defaults(func=_cmd_block)

    p_cancel = subs.add_parser("cancel", help="Move a task to CANCELLED")
    p_cancel.add_argument("task_id")
    p_cancel.add_argument("--reason")
    _add_json(p_cancel)
    p_cancel.set_defaults(func=_cmd_cancel)

    p_events = subs.add_parser("events", help="Show the audit trail for a task")
    p_events.add_argument("task_id")
    _add_json(p_events)
    p_events.set_defaults(func=_cmd_events)

    p_age = subs.add_parser(
        "age-check",
        help="Evaluate 24h/72h aging thresholds (no-op unless enabled, unless --dry-run)",
    )
    p_age.add_argument(
        "--dry-run", action="store_true",
        help="Preview aging regardless of the feature flag; never mutates state",
    )
    p_age.add_argument(
        "--verbose", action="store_true",
        help="When disabled and not --dry-run, print a shadow-mode summary to stderr",
    )
    _add_json(p_age)
    p_age.set_defaults(func=_cmd_age_check)

    p_enable = subs.add_parser("enable", help="Turn on the task-ownership controller")
    p_enable.set_defaults(func=_cmd_enable)

    p_disable = subs.add_parser(
        "disable", help="Turn off the task-ownership controller (clean rollback: age-check goes inert)"
    )
    p_disable.set_defaults(func=_cmd_disable)
