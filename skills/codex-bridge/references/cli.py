#!/usr/bin/env python3
"""Productized CLI for Hermes Codex Bridge."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .validator import (
        SMOKE_SENTINEL,
        TERMINAL_STATUSES,
        ValidationError,
        parse_json_object,
        validate_approval_policy,
        validate_bridge_output,
        validate_interrupt_input,
        validate_respond_input,
        validate_sandbox,
        validate_smoke_test_result,
        validate_start_input,
        validate_status_input,
        validate_steer_input,
    )
except ImportError:
    from validator import (  # type: ignore
        SMOKE_SENTINEL,
        TERMINAL_STATUSES,
        ValidationError,
        parse_json_object,
        validate_approval_policy,
        validate_bridge_output,
        validate_interrupt_input,
        validate_respond_input,
        validate_sandbox,
        validate_smoke_test_result,
        validate_start_input,
        validate_status_input,
        validate_steer_input,
    )

from tools.codex_bridge_tool import DEFAULT_APPROVAL_POLICY, DEFAULT_SANDBOX, codex_bridge


def emit(data: dict[str, Any]) -> None:
    print(json.dumps(data, ensure_ascii=False, sort_keys=True))


def call_bridge(action: str, **kwargs: Any) -> dict[str, Any]:
    raw = codex_bridge(action=action, **kwargs)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"codex_bridge returned invalid JSON for {action}: {exc.msg}") from exc
    validate_bridge_output(action, data)
    return data


def _prompt_from_args(args: argparse.Namespace) -> str:
    prompt = args.prompt
    if prompt is None and args.prompt_text:
        prompt = " ".join(args.prompt_text)
    return prompt or ""


def cmd_start(args: argparse.Namespace) -> dict[str, Any]:
    prompt = _prompt_from_args(args)
    validate_start_input(prompt, args.cwd, args.sandbox, args.approval_policy)
    return call_bridge(
        "start",
        prompt=prompt,
        cwd=args.cwd,
        model=args.model,
        sandbox=args.sandbox,
        approval_policy=args.approval_policy,
        codex_home=args.codex_home,
    )


def cmd_status(args: argparse.Namespace) -> dict[str, Any]:
    validate_status_input(args.task_id)
    return call_bridge("status", task_id=args.task_id)


def cmd_list(args: argparse.Namespace) -> dict[str, Any]:
    return call_bridge("list", limit=args.limit)


def cmd_steer(args: argparse.Namespace) -> dict[str, Any]:
    validate_steer_input(args.task_id, args.instruction)
    return call_bridge("steer", task_id=args.task_id, instruction=args.instruction)


def cmd_interrupt(args: argparse.Namespace) -> dict[str, Any]:
    validate_interrupt_input(args.task_id)
    return call_bridge("interrupt", task_id=args.task_id)


def cmd_respond(args: argparse.Namespace) -> dict[str, Any]:
    answers = parse_json_object(args.answers, field_name="answers")
    validate_respond_input(args.task_id, args.request_id, args.decision, answers)
    return call_bridge(
        "respond",
        task_id=args.task_id,
        instruction=args.request_id,
        decision=args.decision,
        answers=answers,
    )


def _smoke_prompt(wait_seconds: int) -> str:
    return (
        f"Wait {wait_seconds} seconds asynchronously, then reply exactly {SMOKE_SENTINEL}. "
        "Do not modify files."
    )


def cmd_smoke_test(args: argparse.Namespace) -> dict[str, Any]:
    validate_start_input(_smoke_prompt(args.wait), args.cwd, args.sandbox, args.approval_policy)
    started = call_bridge(
        "start",
        prompt=_smoke_prompt(args.wait),
        cwd=args.cwd,
        model=args.model,
        sandbox=args.sandbox,
        approval_policy=args.approval_policy,
        codex_home=args.codex_home,
    )
    task_id = started["task"]["hermes_task_id"]
    deadline = time.monotonic() + args.timeout
    last_status: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        time.sleep(args.poll_interval)
        last_status = call_bridge("status", task_id=task_id)
        task = last_status.get("task") or {}
        if task.get("status") in TERMINAL_STATUSES:
            validate_smoke_test_result(last_status)
            return {
                "success": True,
                "task_id": task_id,
                "status": task.get("status"),
                "start": started,
                "final_status": last_status,
            }
    return {
        "success": False,
        "error": f"smoke-test timed out after {args.timeout} seconds.",
        "task_id": task_id,
        "start": started,
        "last_status": last_status,
    }


def add_common_start_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cwd", default=str(Path.cwd()), help="Working directory for Codex.")
    parser.add_argument("--model", default=None, help="Optional Codex model override.")
    parser.add_argument("--sandbox", default=DEFAULT_SANDBOX, type=validate_sandbox)
    parser.add_argument("--approval-policy", default=DEFAULT_APPROVAL_POLICY, type=validate_approval_policy)
    parser.add_argument("--codex-home", default=None, help="Optional CODEX_HOME override.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hermes Codex Bridge skill CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start", help="Start a Codex task.")
    start.add_argument("--prompt", help="Task prompt.")
    start.add_argument("prompt_text", nargs="*", help="Task prompt as positional text.")
    add_common_start_options(start)
    start.set_defaults(func=cmd_start)

    status = subparsers.add_parser("status", help="Show task status.")
    status.add_argument("task_id")
    status.set_defaults(func=cmd_status)

    list_parser = subparsers.add_parser("list", help="List recent Codex Bridge tasks.")
    list_parser.add_argument("--limit", type=int, default=10)
    list_parser.set_defaults(func=cmd_list)

    steer = subparsers.add_parser("steer", help="Steer an active Codex turn.")
    steer.add_argument("task_id")
    steer.add_argument("--instruction", required=True)
    steer.set_defaults(func=cmd_steer)

    interrupt = subparsers.add_parser("interrupt", help="Interrupt an active Codex turn.")
    interrupt.add_argument("task_id")
    interrupt.set_defaults(func=cmd_interrupt)

    respond = subparsers.add_parser("respond", help="Respond to a pending Codex request.")
    respond.add_argument("task_id")
    respond.add_argument("--request-id", required=True)
    respond.add_argument("--decision", default="decline")
    respond.add_argument("--answers", default=None, help="JSON object for user-input answers.")
    respond.set_defaults(func=cmd_respond)

    smoke = subparsers.add_parser("smoke-test", help="Run an async Codex Bridge smoke test.")
    smoke.add_argument("--wait", type=int, default=10)
    smoke.add_argument("--timeout", type=int, default=60)
    smoke.add_argument("--poll-interval", type=float, default=2.0)
    add_common_start_options(smoke)
    smoke.set_defaults(func=cmd_smoke_test)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        result = args.func(args)
        emit(result)
        return 0 if result.get("success") is True else 1
    except ValidationError as exc:
        emit({"success": False, "error": str(exc)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
