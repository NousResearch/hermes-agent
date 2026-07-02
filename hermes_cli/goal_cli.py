"""Noninteractive goal recovery CLI."""

from __future__ import annotations

import argparse


def _goal_manager(args):
    from hermes_cli.goals import GoalManager

    return GoalManager(session_id=args.session_id)


def _cmd_goal_status(args) -> int:
    mgr = _goal_manager(args)
    print(mgr.status_line())
    return 0


def _cmd_goal_pause(args) -> int:
    mgr = _goal_manager(args)
    current = mgr.state
    if current is None:
        print("No active goal.")
        return 1
    if current.status != "active":
        print(f"cannot pause goal with status={current.status}")
        return 1
    state = mgr.pause(reason=getattr(args, "reason", None) or "cli-paused")
    print(f"Goal paused: {state.goal}")
    return 0


def _cmd_goal_resume(args) -> int:
    mgr = _goal_manager(args)
    current = mgr.state
    if current is None:
        print("No goal to resume.")
        return 1
    if current.status != "paused":
        print(f"cannot resume goal with status={current.status}")
        return 1
    state = mgr.resume(reason=getattr(args, "reason", None) or "cli-resumed")
    print(f"Goal resumed: {state.goal}")
    return 0


def _cmd_goal_clear(args) -> int:
    mgr = _goal_manager(args)
    if not mgr.has_goal():
        print("No active goal.")
        return 1
    goal = mgr.state.goal if mgr.state is not None else ""
    mgr.clear(reason=getattr(args, "reason", None) or "cli-cleared")
    print(f"Goal cleared: {goal}")
    return 0


def _cmd_goal_complete(args) -> int:
    mgr = _goal_manager(args)
    state = mgr.state
    if state is None or state.status not in {"active", "paused"}:
        print("No active goal.")
        return 1
    reason = (getattr(args, "reason", None) or "").strip()
    if not reason:
        print("Goal complete requires --reason.")
        return 2
    mgr.mark_done(reason)
    print(f"Goal completed: {state.goal}")
    return 0


def _add_session_and_reason(parser, *, reason_required: bool = False) -> None:
    parser.add_argument(
        "--session-id",
        required=True,
        help="Hermes session id whose goal state should be managed",
    )
    parser.add_argument(
        "--reason",
        required=reason_required,
        help="Audit reason for this goal state transition",
    )


def build_parser(subparsers) -> argparse.ArgumentParser:
    goal_parser = subparsers.add_parser(
        "goal",
        help="Inspect and recover persistent goal state noninteractively",
    )
    goal_parser.set_defaults(func=lambda _args: (goal_parser.print_help(), 0)[1])
    goal_sub = goal_parser.add_subparsers(dest="goal_command")

    status = goal_sub.add_parser("status", help="Show goal status")
    status.add_argument("--session-id", required=True)
    status.set_defaults(func=_cmd_goal_status)

    pause = goal_sub.add_parser("pause", help="Pause a goal")
    _add_session_and_reason(pause)
    pause.set_defaults(func=_cmd_goal_pause)

    resume = goal_sub.add_parser("resume", help="Resume a goal")
    _add_session_and_reason(resume)
    resume.set_defaults(func=_cmd_goal_resume)

    clear = goal_sub.add_parser("clear", help="Clear a goal")
    _add_session_and_reason(clear)
    clear.set_defaults(func=_cmd_goal_clear)

    complete = goal_sub.add_parser("complete", help="Mark a goal complete")
    _add_session_and_reason(complete, reason_required=True)
    complete.set_defaults(func=_cmd_goal_complete)
    return goal_parser
