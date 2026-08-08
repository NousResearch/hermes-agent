"""plan-secretary CLI entry point.

Usage: ``python -m plugins.plan_secretary <command> ...``
"""
from __future__ import annotations

import argparse
import sys

from . import core


def cmd_scan_text(args: argparse.Namespace) -> int:
    text = args.text if args.text is not None else args.file
    if args.file:
        text = open(args.file, encoding="utf-8").read()
    captures = core.scan_text(
        text,
        source=args.source,
        source_id=args.source_id,
        source_role=args.source_role,
        source_session_id=args.source_session_id,
        source_message_id=args.source_message_id,
    )
    print(f"CAPTURED {len(captures)}")
    for capture in captures:
        print(f"{capture['id']} | pending | keywords={','.join(capture['matched_keywords'])}")
        print(f"  text: {capture['text']}")
    return 0


def cmd_list_captures(args: argparse.Namespace) -> int:
    captures = core.load_captures()["captures"]
    if not args.all:
        captures = [c for c in captures if c.get("status") == "pending"]
    if not captures:
        print("NO_CAPTURES")
        return 0
    for capture in sorted(captures, key=lambda c: c.get("created_at") or ""):
        print(f"{capture.get('id')} | {capture.get('status')} | keywords={','.join(capture.get('matched_keywords') or [])}")
        print(f"  text: {capture.get('text')}")
    return 0


def cmd_confirm_capture(args: argparse.Namespace) -> int:
    plan = core.confirm_capture(
        args.capture_id,
        due=args.due,
        mode=args.mode,
        title=args.title,
        owner=args.owner,
        worker=args.worker,
        priority=args.priority,
        prereq=args.prereq,
        next_action=args.next_action,
    )
    print(f"CONFIRMED_CAPTURE {args.capture_id} -> {plan['id']}")
    return 0


def cmd_ignore_capture(args: argparse.Namespace) -> int:
    capture = core.ignore_capture(args.capture_id, reason=args.reason)
    print(f"IGNORED_CAPTURE {capture['id']}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    plans = core.load_registry()["plans"]
    if not args.all:
        plans = [p for p in plans if p.get("status") not in core.TERMINAL_STATUSES]
    if not plans:
        print("NO_PLANS")
        return 0
    for plan in sorted(plans, key=lambda p: p.get("due") or ""):
        print(f"{plan.get('id')} | {plan.get('status')} | due={plan.get('due')}")
        print(f"  title: {plan.get('title')}")
        print(f"  next_action: {plan.get('next_action') or '-'}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    registry = core.load_registry()
    plans = registry["plans"]
    active = [p for p in plans if p.get("status") in core.ACTIVE_STATUSES]
    print(f"total={len(plans)} unfinished={len([p for p in plans if p.get('status') not in core.TERMINAL_STATUSES])} active={len(active)}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    registry = core.load_registry()
    now = core.now_local()
    due = []
    for plan in registry["plans"]:
        if plan.get("status") not in core.ACTIVE_STATUSES:
            continue
        due_at = core.parse_plan_time(plan.get("due"))
        if not due_at or due_at > now:
            continue
        if not core.is_prereq_satisfied(plan):
            plan["status"] = "blocked"
            plan["block_reason"] = f"prereq unmet: {plan.get('prereq')}"
            core.save_registry(registry)
            print(f"PREREQ_BLOCKED {plan.get('id')}")
            continue
        due.append(plan)
    if not due:
        print("NO_DUE_PLANS")
    for plan in due:
        print(f"DUE {plan.get('id')} | {plan.get('title')} | due={plan.get('due')} | next_action={plan.get('next_action') or '-'}")
    return 0


def cmd_notify(args: argparse.Namespace) -> int:
    messages = core.notify(
        session_id=args.session_id,
        state_path=args.state_path,
        default_due=args.default_due,
        due_repeat_minutes=args.due_repeat_minutes,
        repeat_pending=args.repeat_pending,
    )
    if messages:
        print("\n\n".join(messages))
    elif args.verbose:
        print("NO_PLAN_SECRETARY_NOTIFICATIONS")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m plugins.plan_secretary",
                                     description="Plan Secretary: human-confirmed, session-scoped task plans.")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan-text")
    source = scan.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--file")
    scan.add_argument("--source", default="")
    scan.add_argument("--source-id", default="")
    scan.add_argument("--source-role", default="")
    scan.add_argument("--source-session-id", default="")
    scan.add_argument("--source-message-id", default="")
    scan.set_defaults(func=cmd_scan_text)

    list_caps = sub.add_parser("list-captures")
    list_caps.add_argument("--all", action="store_true")
    list_caps.set_defaults(func=cmd_list_captures)

    confirm = sub.add_parser("confirm-capture")
    confirm.add_argument("capture_id")
    confirm.add_argument("--due", required=True)
    confirm.add_argument("--mode", choices=["parallel", "replace", "idea"], default="parallel")
    confirm.add_argument("--title", default="")
    confirm.add_argument("--owner", default="")
    confirm.add_argument("--worker", default="")
    confirm.add_argument("--priority", choices=["low", "normal", "high"], default="normal")
    confirm.add_argument("--prereq", default="")
    confirm.add_argument("--next-action", default="")
    confirm.set_defaults(func=cmd_confirm_capture)

    ignore = sub.add_parser("ignore-capture")
    ignore.add_argument("capture_id")
    ignore.add_argument("--reason", default="")
    ignore.set_defaults(func=cmd_ignore_capture)

    ls = sub.add_parser("list")
    ls.add_argument("--all", action="store_true")
    ls.set_defaults(func=cmd_list)

    status = sub.add_parser("status")
    status.set_defaults(func=cmd_status)

    check = sub.add_parser("check")
    check.set_defaults(func=cmd_check)

    notify = sub.add_parser("notify")
    notify.add_argument("--session-id", default="")
    notify.add_argument("--state-path", default=None)
    notify.add_argument("--default-due", default="10m")
    notify.add_argument("--due-repeat-minutes", type=int, default=10)
    notify.add_argument("--repeat-pending", action="store_true")
    notify.add_argument("--verbose", action="store_true")
    notify.set_defaults(func=cmd_notify)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, SystemExit) as exc:
        if isinstance(exc, SystemExit):
            raise
        print(f"ERROR {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
