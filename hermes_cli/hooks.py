"""hermes hooks — inspect and manage shell-script hooks."""

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
    handler = _ACTIONS.get(sub)
    if handler is None:
        print(f"Unknown hooks subcommand: {sub}")
        return
    handler(args)


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

def _cmd_list(_args) -> None:
    from hermes_cli.config import load_config
    from agent import outbound_webhooks, shell_hooks

    cfg = load_config()
    specs = shell_hooks.iter_configured_hooks(cfg)
    outbound = outbound_webhooks.iter_configured_targets(cfg)

    if not specs and not outbound:
        print("No shell hooks or outbound webhooks configured in ~/.hermes/config.yaml.")
        print("See `hermes hooks --help` or")
        print("    website/docs/user-guide/features/hooks.md")
        print("for the config schema and worked examples.")
        return

    if not specs:
        print("No shell hooks configured in ~/.hermes/config.yaml.")
    else:
        by_event: Dict[str, List] = {}
        for spec in specs:
            by_event.setdefault(spec.event, []).append(spec)
        approved = {
            (e.get("event"), e.get("command"))
            for e in shell_hooks.load_allowlist().get("approvals", [])
            if isinstance(e, dict)
        }
        print(f"Configured shell hooks ({len(specs)} total):\n")
        for event in sorted(by_event):
            print(f"  [{event}]")
            for spec in by_event[event]:
                is_approved = (spec.event, spec.command) in approved
                status = "✓ allowed" if is_approved else "✗ not allowlisted"
                print(f"    - {spec.command}{_matcher_part(spec)} (timeout={spec.timeout}s, {status})")
                entry = shell_hooks.allowlist_entry_for(spec.event, spec.command) if is_approved else None
                if entry and entry.get("approved_at"):
                    print(f"      approved_at: {entry['approved_at']}")
                    drift, mtime_at, mtime_now = _mtime_drift(shell_hooks, spec, entry)
                    if drift is True:
                        print(
                            f"      ⚠ script modified since approval "
                            f"(was {mtime_at}, now {mtime_now}) — "
                            f"run `hermes hooks doctor` to re-validate"
                        )
            print()

    if outbound:
        print(f"Configured outbound webhooks ({len(outbound)} total):\n")
        for target in outbound:
            signed = "signed" if target.secret else "UNSIGNED"
            print(f"  - {target.label}")
            print(f"      url:     {target.url}")
            print(f"      events:  {', '.join(target.events)}{_matcher_part(target)} (timeout={target.timeout}s, {signed})")
        print()


def _matcher_part(obj) -> str:
    return f" matcher={obj.matcher!r}" if obj.matcher else ""


def _mtime_drift(shell_hooks, spec, entry) -> tuple[bool | None, str, str]:
    """``(drift, mtime_at, mtime_now)``: True if the script changed since approval, False if
    unchanged, None when either mtime is unknown."""
    mtime_now = shell_hooks.script_mtime_iso(spec.command)
    mtime_at = entry.get("script_mtime_at_approval")
    if not (mtime_now and mtime_at):
        return None, mtime_at, mtime_now
    if mtime_now > mtime_at:
        return True, mtime_at, mtime_now
    return (False if mtime_now == mtime_at else None), mtime_at, mtime_now


# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------

# Synthetic kwargs matching the real invoke_hook() call sites. They go verbatim to
# agent.shell_hooks.run_once() -> the same _serialize_payload() production uses, so the
# stdin a script sees under `hooks test` / `hooks doctor` has the runtime shape.
_DEFAULT_PAYLOADS = {
    "pre_tool_call": {
        "tool_name": "terminal", "args": {"command": "echo hello"},
        "session_id": "test-session", "task_id": "test-task", "tool_call_id": "test-call",
    },
    "post_tool_call": {
        "tool_name": "terminal", "args": {"command": "echo hello"},
        "session_id": "test-session", "task_id": "test-task", "tool_call_id": "test-call",
        "result": '{"output": "hello"}', "duration_ms": 42,
    },
    "pre_llm_call": {
        "session_id": "test-session", "user_message": "What is the weather?",
        "conversation_history": [], "is_first_turn": True, "model": "gpt-4", "platform": "cli",
    },
    "post_llm_call": {"session_id": "test-session", "model": "gpt-4", "platform": "cli"},
    "pre_verify": {
        "session_id": "test-session", "platform": "cli", "model": "gpt-4", "coding": True,
        "attempt": 0, "final_response": "All done — the change is applied.",
        "changed_paths": ["src/app.tsx"],
    },
    "on_session_start": {"session_id": "test-session"},
    "on_session_end": {
        "session_id": "test-session", "task_id": "test-task", "turn_id": "test-turn",
        "completed": True, "failed": False, "interrupted": False,
        "turn_exit_reason": "text_response(stop)", "model": "gpt-4", "platform": "cli",
    },
    "on_session_finalize": {"session_id": "test-session"},
    "on_session_reset": {"session_id": "test-session"},
    "pre_api_request": {
        "session_id": "test-session", "task_id": "test-task", "platform": "cli",
        "model": "claude-sonnet-4-6", "provider": "anthropic",
        "base_url": "https://api.anthropic.com", "api_mode": "anthropic_messages",
        "api_call_count": 1, "message_count": 4, "tool_count": 12,
        "approx_input_tokens": 2048, "request_char_count": 8192, "max_tokens": 4096,
    },
    "post_api_request": {
        "session_id": "test-session", "task_id": "test-task", "platform": "cli",
        "model": "claude-sonnet-4-6", "provider": "anthropic",
        "base_url": "https://api.anthropic.com", "api_mode": "anthropic_messages",
        "api_call_count": 1, "api_duration": 1.234,
        "started_at": 1756000000.0, "ended_at": 1756000001.234, "first_chunk_at": 1756000000.512,
        "finish_reason": "stop", "message_count": 4, "response_model": "claude-sonnet-4-6",
        "usage": {"input_tokens": 2048, "output_tokens": 512},
        "assistant_content_chars": 1200, "assistant_tool_call_count": 0,
        # Per-advisor metrics on a MoA turn, None otherwise: MoA returns only the aggregator's
        # response, so without this an observer cannot see or price the fan-out.
        "moa_references": None,
    },
    "subagent_stop": {
        "parent_session_id": "parent-sess", "child_role": None,
        "child_summary": "Synthetic summary for hooks test", "child_status": "completed",
        "tool_call_history": [{
            "tool_name": "write_file",
            "tool_input": {"argument_keys": ["content", "path"], "targets": {"path": "/tmp/report.txt"}},
            "input_bytes": 128, "output_bytes": 32, "status": "ok",
        }],
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

    # Synthetic kwargs merged with --for-tool (overrides tool_name) and --payload-file (extra kwargs).
    payload = dict(_DEFAULT_PAYLOADS.get(event, {"session_id": "test-session"}))
    for_tool = getattr(args, "for_tool", None)
    if for_tool:
        payload["tool_name"] = for_tool
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

    specs = [s for s in shell_hooks.iter_configured_hooks(load_config()) if s.event == event]
    if for_tool:
        specs = [s for s in specs if s.event not in {"pre_tool_call", "post_tool_call"} or s.matches_tool(for_tool)]
    if not specs:
        print(f"No shell hooks configured for event: {event}")
        if for_tool:
            print(f"(with matcher filter --for-tool={for_tool})")
        return

    print(f"Firing {len(specs)} hook(s) for event '{event}':\n")
    for spec in specs:
        print(f"  → {spec.command}")
        _print_run_result(shell_hooks.run_once(spec, payload))
        print()


def _print_run_result(result: Dict[str, Any]) -> None:
    if result.get("error"):
        print(f"      ✗ error: {result['error']}")
        return
    if result.get("timed_out"):
        print(f"      ✗ timed out after {result['elapsed_seconds']}s")
        return
    print(f"      exit={result.get('returncode')}  elapsed={result.get('elapsed_seconds', 0)}s")
    for stream in ("stdout", "stderr"):
        text = (result.get(stream) or "").strip()
        if text:
            print(f"      {stream}: {_truncate(text, 400)}")
    parsed = result.get("parsed")
    if parsed:
        print(f"      parsed (Hermes wire shape): {json.dumps(parsed)}")
    else:
        print("      parsed: <none — hook contributed nothing to the dispatcher>")


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 3] + "..."


# ---------------------------------------------------------------------------
# revoke / doctor
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
    print(f"{problems} issue(s) found.  Fix before relying on these hooks." if problems else "All shell hooks look healthy.")


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
        drift, mtime_at, mtime_now = _mtime_drift(shell_hooks, spec, entry)
        if drift is True:
            problems += 1
            print(f"      ⚠ script modified since approval "
                  f"(was {mtime_at}, now {mtime_now}) — review changes, "
                  f"then `hermes hooks revoke` + re-approve to refresh")
        elif drift is False:
            print("      ✓ script unchanged since approval")
    # 4. JSON smoke test on a synthetic payload — ONLY when already allowlisted. Otherwise doctor
    # would execute every script in a freshly-pulled config before the user reviewed it, which
    # contradicts the documented workflow ("spot newly-added hooks *before they register*").
    if not entry:
        print("      ℹ skipped JSON smoke test — not allowlisted yet. "
              "Approve the hook first (via TTY prompt or --accept-hooks), "
              "then re-run `hermes hooks doctor`.")
    elif shell_hooks.script_is_executable(spec.command):
        result = shell_hooks.run_once(spec, _DEFAULT_PAYLOADS.get(spec.event, {"extra": {}}))
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
            if not stdout:
                print(f"      ✓ ran clean with empty stdout "
                      f"(exit={rc}, {elapsed}s) — hook is observer-only")
            else:
                try:
                    json.loads(stdout)
                    print(f"      ✓ produced valid JSON on synthetic payload "
                          f"(exit={rc}, {elapsed}s)")
                except json.JSONDecodeError:
                    problems += 1
                    print(f"      ✗ stdout was not valid JSON (exit={rc}, "
                          f"{elapsed}s): {_truncate(stdout, 120)}")
    return problems


_ACTIONS = {
    "list": _cmd_list, "ls": _cmd_list,
    "test": _cmd_test,
    "revoke": _cmd_revoke, "remove": _cmd_revoke, "rm": _cmd_revoke,
    "doctor": _cmd_doctor,
}
