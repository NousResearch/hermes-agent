"""hermes hooks — inspect and manage shell-script hooks.

Usage::

    hermes hooks list
    hermes hooks test <event> [--for-tool X] [--payload-file F]
    hermes hooks revoke <command>
    hermes hooks doctor

Consent records live under ``~/.hermes/shell-hooks-allowlist.json`` and
hook definitions come from the ``hooks:`` block in ``~/.hermes/config.yaml``
(the same config read by the CLI / gateway at startup).

This module is a thin CLI shell over :mod:`agent.shell_hooks`; every
shared concern (payload serialisation, response parsing, allowlist
format) lives there.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def hooks_command(args) -> None:
    """Entry point for ``hermes hooks`` — dispatches to the requested action."""
    sub = getattr(args, "hooks_action", None)

    if not sub:
        print("Usage: hermes hooks {list|test|revoke|doctor}")
        print("Run 'hermes hooks --help' for details.")
        return

    if sub in {"list", "ls"}:
        _cmd_list(args)
    elif sub == "test":
        _cmd_test(args)
    elif sub in {"revoke", "remove", "rm"}:
        _cmd_revoke(args)
    elif sub == "doctor":
        _cmd_doctor(args)
    else:
        print(f"Unknown hooks subcommand: {sub}")


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

def _cmd_list(_args) -> None:
    from hermes_cli.config import load_config
    from agent import shell_hooks

    specs = shell_hooks.iter_configured_hooks(load_config())

    if not specs:
        print("No shell hooks configured in ~/.hermes/config.yaml.")
        print("See `hermes hooks --help` or")
        print("    website/docs/user-guide/features/hooks.md")
        print("for the config schema and worked examples.")
        return

    by_event: Dict[str, List] = {}
    for spec in specs:
        by_event.setdefault(spec.event, []).append(spec)

    allowlist = shell_hooks.load_allowlist()
    approved = {
        (e.get("event"), e.get("command"))
        for e in allowlist.get("approvals", [])
        if isinstance(e, dict)
    }

    print(f"Configured shell hooks ({len(specs)} total):\n")

    for event in sorted(by_event.keys()):
        print(f"  [{event}]")
        for spec in by_event[event]:
            is_approved = (spec.event, spec.command) in approved
            status = "✓ allowed" if is_approved else "✗ not allowlisted"
            matcher_part = f" matcher={spec.matcher!r}" if spec.matcher else ""
            print(
                f"    - {spec.command}{matcher_part} "
                f"(timeout={spec.timeout}s, {status})"
            )

            if is_approved:
                entry = shell_hooks.allowlist_entry_for(spec.event, spec.command)
                if entry and entry.get("approved_at"):
                    print(f"      approved_at: {entry['approved_at']}")
                    mtime_now = shell_hooks.script_mtime_iso(spec.command)
                    mtime_at = entry.get("script_mtime_at_approval")
                    if mtime_now and mtime_at and mtime_now > mtime_at:
                        print(
                            f"      ⚠ script modified since approval "
                            f"(was {mtime_at}, now {mtime_now}) — "
                            f"run `hermes hooks doctor` to re-validate"
                        )
        print()


# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------

# Synthetic kwargs matching the real invoke_hook() call sites — these are
# passed verbatim to agent.shell_hooks.run_once(), which routes them through
# the same _serialize_payload() that production firings use.  That way the
# stdin a script sees under `hermes hooks test` and `hermes hooks doctor`
# is identical in shape to what it will see at runtime.
_DEFAULT_PAYLOADS = {
    "pre_tool_call": {
        "tool_name": "terminal",
        "args": {"command": "echo hello"},
        "session_id": "test-session",
        "task_id": "test-task",
        "tool_call_id": "test-call",
    },
    "post_tool_call": {
        "tool_name": "terminal",
        "args": {"command": "echo hello"},
        "session_id": "test-session",
        "task_id": "test-task",
        "tool_call_id": "test-call",
        "result": '{"output": "hello"}',
        "duration_ms": 42,
    },
    "pre_llm_call": {
        "session_id": "test-session",
        "user_message": "What is the weather?",
        "conversation_history": [],
        "is_first_turn": True,
        "model": "gpt-4",
        "platform": "cli",
    },
    "post_llm_call": {
        "session_id": "test-session",
        "model": "gpt-4",
        "platform": "cli",
    },
    "on_session_start": {"session_id": "test-session"},
    "on_session_end": {"session_id": "test-session"},
    "on_session_finalize": {"session_id": "test-session"},
    "on_session_reset": {"session_id": "test-session"},
    "pre_api_request": {
        "session_id": "test-session",
        "task_id": "test-task",
        "platform": "cli",
        "model": "claude-sonnet-4-6",
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com",
        "api_mode": "anthropic_messages",
        "api_call_count": 1,
        "message_count": 4,
        "tool_count": 12,
        "approx_input_tokens": 2048,
        "request_char_count": 8192,
        "max_tokens": 4096,
    },
    "post_api_request": {
        "session_id": "test-session",
        "task_id": "test-task",
        "platform": "cli",
        "model": "claude-sonnet-4-6",
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com",
        "api_mode": "anthropic_messages",
        "api_call_count": 1,
        "api_duration": 1.234,
        "finish_reason": "stop",
        "message_count": 4,
        "response_model": "claude-sonnet-4-6",
        "usage": {"input_tokens": 2048, "output_tokens": 512},
        "assistant_content_chars": 1200,
        "assistant_tool_call_count": 0,
    },
    "subagent_stop": {
        "parent_session_id": "parent-sess",
        "child_role": None,
        "child_summary": "Synthetic summary for hooks test",
        "child_status": "completed",
        "duration_ms": 1234,
    },
}


def _cmd_test(args) -> None:
    from hermes_cli.config import load_config
    from hermes_cli.plugins import VALID_HOOKS
    from agent import shell_hooks

    event = args.event
    if event not in VALID_HOOKS:
        print(f"Unknown event: {event!r}")
        print(f"Valid events: {', '.join(sorted(VALID_HOOKS))}")
        return

    # Synthetic kwargs in the same shape invoke_hook() would pass.  Merged
    # with --for-tool (overrides tool_name) and --payload-file (extra kwargs).
    payload = dict(_DEFAULT_PAYLOADS.get(event, {"session_id": "test-session"}))

    if getattr(args, "for_tool", None):
        payload["tool_name"] = args.for_tool

    if getattr(args, "payload_file", None):
        try:
            custom = json.loads(Path(args.payload_file).read_text(encoding="utf-8"))
            if isinstance(custom, dict):
                payload.update(custom)
            else:
                print(f"Warning: {args.payload_file} is not a JSON object; ignoring")
        except Exception as exc:
            print(f"Error reading payload file: {exc}")
            return

    specs = shell_hooks.iter_configured_hooks(load_config())
    specs = [s for s in specs if s.event == event]

    if getattr(args, "for_tool", None):
        specs = [
            s for s in specs
            if s.event not in {"pre_tool_call", "post_tool_call"}
            or s.matches_tool(args.for_tool)
        ]

    if not specs:
        print(f"No shell hooks configured for event: {event}")
        if getattr(args, "for_tool", None):
            print(f"(with matcher filter --for-tool={args.for_tool})")
        return

    print(f"Firing {len(specs)} hook(s) for event '{event}':\n")
    for spec in specs:
        print(f"  → {spec.command}")
        result = shell_hooks.run_once(spec, payload)
        _print_run_result(result)
        print()


def _print_run_result(result: Dict[str, Any]) -> None:
    if result.get("error"):
        print(f"      ✗ error: {result['error']}")
        return
    if result.get("timed_out"):
        print(f"      ✗ timed out after {result['elapsed_seconds']}s")
        return

    rc = result.get("returncode")
    elapsed = result.get("elapsed_seconds", 0)
    print(f"      exit={rc}  elapsed={elapsed}s")

    stdout = (result.get("stdout") or "").strip()
    stderr = (result.get("stderr") or "").strip()
    if stdout:
        print(f"      stdout: {_truncate(stdout, 400)}")
    if stderr:
        print(f"      stderr: {_truncate(stderr, 400)}")

    parsed = result.get("parsed")
    if parsed:
        print(f"      parsed (Hermes wire shape): {json.dumps(parsed)}")
    else:
        print("      parsed: <none — hook contributed nothing to the dispatcher>")


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 3] + "..."


# ---------------------------------------------------------------------------
# revoke
# ---------------------------------------------------------------------------

def _cmd_revoke(args) -> None:
    from agent import shell_hooks

    removed = shell_hooks.revoke(args.command)
    if removed == 0:
        print(f"No allowlist entry found for command: {args.command}")
        return
    print(f"Removed {removed} allowlist entry/entries for: {args.command}")
    print(
        "Note: currently running CLI / gateway processes keep their "
        "already-registered callbacks until they restart."
    )


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------

def _cmd_doctor(_args) -> None:
    from hermes_cli.config import load_config
    from agent import shell_hooks

    specs = shell_hooks.iter_configured_hooks(load_config())

    if not specs:
        print("No shell hooks configured — nothing to check.")
        return

    print(f"Checking {len(specs)} configured shell hook(s)...\n")

    problems = 0
    for spec in specs:
        print(f"  [{spec.event}] {spec.command}")
        problems += _doctor_one(spec, shell_hooks)
        print()

    if problems:
        print(f"{problems} issue(s) found.  Fix before relying on these hooks.")
    else:
        print("All shell hooks look healthy.")


def _doctor_one(spec, shell_hooks) -> int:
    problems = 0

    # 1. Script exists and is executable
    if shell_hooks.script_is_executable(spec.command):
        print("      ✓ script exists and is executable")
    else:
        problems += 1
        print("      ✗ script missing or not executable "
              "(chmod +x the file, or fix the path)")

    # 2. Allowlist status
    entry = shell_hooks.allowlist_entry_for(spec.event, spec.command)
    if entry:
        print(f"      ✓ allowlisted (approved {entry.get('approved_at', '?')})")
    else:
        problems += 1
        print("      ✗ not allowlisted — hook will NOT fire at runtime "
              "(run with --accept-hooks once, or confirm at the TTY prompt)")

    # 3. Mtime drift
    if entry and entry.get("script_mtime_at_approval"):
        mtime_now = shell_hooks.script_mtime_iso(spec.command)
        mtime_at = entry["script_mtime_at_approval"]
        if mtime_now and mtime_at and mtime_now > mtime_at:
            problems += 1
            print(f"      ⚠ script modified since approval "
                  f"(was {mtime_at}, now {mtime_now}) — review changes, "
                  f"then `hermes hooks revoke` + re-approve to refresh")
        elif mtime_now and mtime_at and mtime_now == mtime_at:
            print("      ✓ script unchanged since approval")

    # 4. Produces valid JSON for a synthetic payload — only when the entry
    # is already allowlisted.  Otherwise `hermes hooks doctor` would execute
    # every script listed in a freshly-pulled config before the user has
    # reviewed them, which directly contradicts the documented workflow
    # ("spot newly-added hooks *before they register*").
    if not entry:
        print("      ℹ skipped JSON smoke test — not allowlisted yet. "
              "Approve the hook first (via TTY prompt or --accept-hooks), "
              "then re-run `hermes hooks doctor`.")
    elif shell_hooks.script_is_executable(spec.command):
        payload = _DEFAULT_PAYLOADS.get(spec.event, {"extra": {}})
        result = shell_hooks.run_once(spec, payload)
        if result.get("timed_out"):
            problems += 1
            print(f"      ✗ timed out after {result['elapsed_seconds']}s "
                  f"on synthetic payload (timeout={spec.timeout}s)")
        elif result.get("error"):
            problems += 1
            print(f"      ✗ execution error: {result['error']}")
        else:
            rc = result.get("returncode")
            elapsed = result.get("elapsed_seconds", 0)
            stdout = (result.get("stdout") or "").strip()
            if stdout:
                try:
                    json.loads(stdout)
                    print(f"      ✓ produced valid JSON on synthetic payload "
                          f"(exit={rc}, {elapsed}s)")
                except json.JSONDecodeError:
                    problems += 1
                    print(f"      ✗ stdout was not valid JSON (exit={rc}, "
                          f"{elapsed}s): {_truncate(stdout, 120)}")
            else:
                print(f"      ✓ ran clean with empty stdout "
                      f"(exit={rc}, {elapsed}s) — hook is observer-only")

    return problems
