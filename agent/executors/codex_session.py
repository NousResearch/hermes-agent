"""Hermes-owned Codex app-server executor session.

This module deliberately sits *under* the normal Hermes tool loop. It reuses the
Codex app-server session transport to perform bounded execution work, then
normalizes the result into structured evidence that Hermes must verify before
reporting completion. It is not a replacement for ``AIAgent.run_conversation``
and does not produce a user-facing final answer.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from typing import Any

from agent.transports.codex_app_server_session import CodexAppServerSession, TurnResult

_EVIDENCE_LIST_FIELDS = ("changed_files", "commands_run", "tests_run")
_EVIDENCE_TEXT_FIELDS = ("summary", "diff")


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list | tuple):
        return [str(item) for item in value if item is not None and str(item)]
    return [str(value)]


def _parse_codex_final_text(text: str) -> dict[str, Any]:
    """Parse a Codex final message if it is JSON; otherwise wrap as summary."""
    stripped = (text or "").strip()
    if not stripped:
        return {}
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return {"summary": stripped, "raw_codex_final_text": text}
    if not isinstance(parsed, Mapping):
        return {"summary": str(parsed), "raw_codex_final_text": text}
    return dict(parsed)


def _base_evidence(*, success: bool) -> dict[str, Any]:
    return {
        "success": success,
        "summary": "",
        "changed_files": [],
        "commands_run": [],
        "tests_run": [],
        "diff": "",
        "error": None,
        "user_facing_final": False,
        "requires_hermes_verification": True,
        "codex": {"thread_id": None, "turn_id": None, "tool_iterations": 0},
    }


def build_codex_executor_prompt(task: str) -> str:
    """Build the bounded Codex prompt for evidence-only executor turns."""
    return (
        "You are running as a bounded Codex executor under Hermes Agent.\n"
        "Hermes/Yuuka remains the user-facing agent and verifier. Do not write "
        "a final response to the user. Perform only the bounded task below, then "
        "return a single JSON object as execution evidence.\n\n"
        "Required JSON keys: success, summary, changed_files, commands_run, "
        "tests_run, diff, error. Use arrays for changed_files, commands_run, "
        "and tests_run. Put null in error when successful.\n\n"
        f"Task:\n{task.strip()}"
    )


def _normalize_success_evidence(result: TurnResult) -> dict[str, Any]:
    parsed = _parse_codex_final_text(result.final_text or "")
    evidence = _base_evidence(success=True)
    for key in _EVIDENCE_TEXT_FIELDS:
        value = parsed.get(key)
        if value is not None:
            evidence[key] = str(value)
    for key in _EVIDENCE_LIST_FIELDS:
        evidence[key] = _as_string_list(parsed.get(key))
    if "raw_codex_final_text" in parsed:
        evidence["raw_codex_final_text"] = str(parsed["raw_codex_final_text"])
    evidence["codex"] = {
        "thread_id": result.thread_id,
        "turn_id": result.turn_id,
        "tool_iterations": int(result.tool_iterations or 0),
    }
    evidence["interrupted"] = bool(result.interrupted)
    evidence["should_retire"] = bool(result.should_retire)
    return evidence


def _normalize_error_evidence(result: TurnResult) -> dict[str, Any]:
    evidence = _base_evidence(success=False)
    evidence["error"] = result.error or "codex session failed"
    evidence["summary"] = evidence["error"]
    evidence["codex"] = {
        "thread_id": result.thread_id,
        "turn_id": result.turn_id,
        "tool_iterations": int(result.tool_iterations or 0),
    }
    evidence["interrupted"] = bool(result.interrupted)
    evidence["should_retire"] = bool(result.should_retire)
    return evidence


def run_codex_session(
    *,
    task: str,
    cwd: str | None = None,
    turn_timeout: float = 600.0,
    codex_bin: str = "codex",
    codex_home: str | None = None,
    permission_profile: str | None = None,
    session_factory: Callable[..., Any] = CodexAppServerSession,
) -> dict[str, Any]:
    """Run one bounded Codex executor turn and return structured evidence.

    The returned dict is intentionally shaped for Hermes verification and
    closeout, not as user-facing prose. Callers must verify changed files,
    commands/tests, and diffs independently before reporting success.
    """
    if not task or not task.strip():
        raise ValueError("task is required")

    session = session_factory(
        cwd=cwd or os.getcwd(),
        codex_bin=codex_bin,
        codex_home=codex_home,
        permission_profile=permission_profile,
    )
    try:
        result = session.run_turn(
            build_codex_executor_prompt(task),
            turn_timeout=turn_timeout,
        )
    finally:
        close = getattr(session, "close", None)
        if callable(close):
            close()

    if getattr(result, "error", None):
        return _normalize_error_evidence(result)
    return _normalize_success_evidence(result)
