"""No-progress loop detection for repeated ineffective tool calls."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from agent.runtime_evidence import current_runtime_evidence, update_runtime_evidence

_TRACKED_TERMINAL_COMMAND_PATTERNS = (
    re.compile(r"\bgh\s+pr\s+(?:checks|view)\b", re.IGNORECASE),
    re.compile(r"\bgh\s+run\s+(?:view|rerun|watch)\b", re.IGNORECASE),
    re.compile(r"\bgh\s+workflow\s+run\b", re.IGNORECASE),
    re.compile(r"\bgit\s+status\b", re.IGNORECASE),
    re.compile(r"\bgit\s+diff\b", re.IGNORECASE),
    re.compile(r"\b(?:grep|rg|sed\s+-n|cat|head|tail|find|ls)\b", re.IGNORECASE),
)
_TRACKED_FILE_TOOL_NAMES = frozenset({"read_file", "search_files", "write_file", "patch"})
_TRACKED_WRITE_HINTS = (
    re.compile(r">>", re.IGNORECASE),
    re.compile(r">", re.IGNORECASE),
    re.compile(r"\btee\b", re.IGNORECASE),
    re.compile(r"\bsed\s+-i\b", re.IGNORECASE),
    re.compile(r"\bperl\s+-pi\b", re.IGNORECASE),
    re.compile(r"\bcp\b", re.IGNORECASE),
    re.compile(r"\bmv\b", re.IGNORECASE),
)


def should_track_no_progress(function_name: str, function_args: Mapping[str, Any] | None) -> bool:
    """Return True when a tool call is loop-prone and worth fingerprinting."""
    args = function_args if isinstance(function_args, Mapping) else {}
    if function_name in _TRACKED_FILE_TOOL_NAMES:
        return True
    if function_name == "execute_code":
        code = str(args.get("code") or "").strip()
        return bool(code and (_contains_write_intent(code) or _contains_read_status_intent(code)))
    if function_name == "terminal":
        command = _terminal_command(args)
        return bool(command and (_contains_read_status_intent(command) or _contains_write_intent(command)))
    return False


def preflight_no_progress_block(
    *,
    function_name: str,
    function_args: Mapping[str, Any] | None,
    task_id: str | None = None,
) -> str | None:
    """Block repeated no-progress calls before they burn another cycle."""
    if not should_track_no_progress(function_name, function_args):
        return None

    args = function_args if isinstance(function_args, Mapping) else {}
    fingerprint = _build_fingerprint(function_name, args, task_id=task_id)
    if fingerprint is None:
        return None

    evidence = current_runtime_evidence() or {}
    tracker = dict(evidence.get("no_progress_loop_state") or {})
    if not tracker:
        return None

    if tracker.get("fingerprint") != fingerprint:
        return None

    if tracker.get("evidence_hash") != _evidence_hash(evidence):
        return None

    block_cycles = int(tracker.get("block_cycles") or 0)
    if block_cycles > 0:
        return _loop_block_message(
            fingerprint=fingerprint,
            evidence=evidence,
            repeat_cycles=int(tracker.get("repeat_cycles") or 0),
            mode="blocked",
        )

    repeat_cycles = int(tracker.get("repeat_cycles") or 0)
    if repeat_cycles >= 2:
        return _loop_block_message(
            fingerprint=fingerprint,
            evidence=evidence,
            repeat_cycles=repeat_cycles,
            mode="blocked",
        )
    if repeat_cycles >= 1:
        return _loop_block_message(
            fingerprint=fingerprint,
            evidence=evidence,
            repeat_cycles=repeat_cycles,
            mode="self_audit",
        )
    return None


def record_no_progress_observation(
    *,
    function_name: str,
    function_args: Mapping[str, Any] | None,
    result: Any,
    task_id: str | None = None,
) -> dict[str, Any] | None:
    """Update loop state after a tool call completes."""
    if not should_track_no_progress(function_name, function_args):
        return None

    args = function_args if isinstance(function_args, Mapping) else {}
    fingerprint = _build_fingerprint(function_name, args, task_id=task_id)
    if fingerprint is None:
        return None

    evidence = current_runtime_evidence() or {}
    evidence_hash = _evidence_hash(evidence)
    result_hash, result_status = _result_fingerprint(result)
    tracker = dict(evidence.get("no_progress_loop_state") or {})

    repeat_cycles = 0
    if tracker.get("fingerprint") == fingerprint and tracker.get("evidence_hash") == evidence_hash and tracker.get("result_hash") == result_hash:
        repeat_cycles = int(tracker.get("repeat_cycles") or 0) + 1

    loop_state = {
        "fingerprint": fingerprint,
        "tool_name": function_name,
        "cwd": fingerprint["cwd"],
        "action": fingerprint["action"],
        "important_args": fingerprint["important_args"],
        "result_status": result_status,
        "result_hash": result_hash,
        "evidence_hash": evidence_hash,
        "repeat_cycles": repeat_cycles,
        "block_cycles": int(tracker.get("block_cycles") or 0),
        "session_id": fingerprint["session_id"],
        "task_id": fingerprint["task_id"],
    }
    update_runtime_evidence(no_progress_loop_state=loop_state)
    return loop_state


def record_no_progress_block(
    *,
    function_name: str,
    function_args: Mapping[str, Any] | None,
    result: Any,
    task_id: str | None = None,
) -> dict[str, Any] | None:
    """Record that the loop guard blocked a repeated action."""
    if not should_track_no_progress(function_name, function_args):
        return None

    args = function_args if isinstance(function_args, Mapping) else {}
    fingerprint = _build_fingerprint(function_name, args, task_id=task_id)
    if fingerprint is None:
        return None

    evidence = current_runtime_evidence() or {}
    evidence_hash = _evidence_hash(evidence)
    result_hash, result_status = _result_fingerprint(result)
    tracker = dict(evidence.get("no_progress_loop_state") or {})

    loop_state = {
        "fingerprint": fingerprint,
        "tool_name": function_name,
        "cwd": fingerprint["cwd"],
        "action": fingerprint["action"],
        "important_args": fingerprint["important_args"],
        "result_status": result_status,
        "result_hash": result_hash,
        "evidence_hash": evidence_hash,
        "repeat_cycles": int(tracker.get("repeat_cycles") or 0),
        "block_cycles": int(tracker.get("block_cycles") or 0) + 1,
        "session_id": fingerprint["session_id"],
        "task_id": fingerprint["task_id"],
    }
    update_runtime_evidence(no_progress_loop_state=loop_state)
    return loop_state


def _build_fingerprint(
    function_name: str,
    function_args: Mapping[str, Any],
    *,
    task_id: str | None = None,
) -> dict[str, Any] | None:
    if function_name == "terminal":
        command = _terminal_command(function_args)
        if not command:
            return None
        action = _classify_terminal_command(command)
        if action is None:
            return None
        cwd = _terminal_cwd(task_id)
        important_args = command
    elif function_name in _TRACKED_FILE_TOOL_NAMES:
        action = _classify_file_tool(function_name, function_args)
        cwd = _current_workspace_hint(task_id)
        important_args = _important_file_args(function_name, function_args)
    elif function_name == "execute_code":
        code = str(function_args.get("code") or "").strip()
        if not code:
            return None
        action = _classify_execute_code(code)
        if action is None:
            return None
        cwd = _current_workspace_hint(task_id)
        important_args = _short_hash(code)
    else:
        return None

    return {
        "tool_name": function_name,
        "cwd": cwd or "unknown",
        "action": action,
        "important_args": important_args,
        "task_id": task_id or "default",
        "session_id": str((current_runtime_evidence() or {}).get("session_id") or ""),
    }


def _terminal_command(args: Mapping[str, Any]) -> str:
    return str(args.get("command") or args.get("cmd") or "").strip()


def _terminal_cwd(task_id: str | None) -> str | None:
    try:
        from tools.terminal_tool import get_active_env

        env = get_active_env(task_id or "default")
    except Exception:
        return None
    if env is None:
        return None
    cwd = getattr(env, "cwd", None)
    return str(cwd).strip() if isinstance(cwd, str) and cwd.strip() else None


def _current_workspace_hint(task_id: str | None) -> str | None:
    evidence = current_runtime_evidence() or {}
    for key in ("audit_repo_root", "pr_context_repo_root"):
        value = str(evidence.get(key) or "").strip()
        if value:
            return value
    return _terminal_cwd(task_id)


def _classify_terminal_command(command: str) -> str | None:
    if _contains_read_status_intent(command):
        return "read/status"
    if _contains_write_intent(command):
        return "write"
    return None


def _classify_file_tool(function_name: str, function_args: Mapping[str, Any]) -> str:
    if function_name == "read_file":
        return "read"
    if function_name == "search_files":
        return "search"
    return "write"


def _classify_execute_code(code: str) -> str | None:
    if _contains_write_intent(code):
        return "write"
    if _contains_read_status_intent(code):
        return "read/status"
    return None


def _important_file_args(function_name: str, function_args: Mapping[str, Any]) -> str:
    if function_name == "read_file":
        return str(function_args.get("path") or "")
    if function_name == "search_files":
        return json.dumps(
            {
                "pattern": function_args.get("pattern"),
                "path": function_args.get("path"),
                "target": function_args.get("target"),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
    return json.dumps(
        {
            "path": function_args.get("path") or function_args.get("file_path") or "",
            "content_hash": _short_hash(str(function_args.get("content") or "")),
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def _contains_read_status_intent(text: str) -> bool:
    return any(pattern.search(text) for pattern in _TRACKED_TERMINAL_COMMAND_PATTERNS)


def _contains_write_intent(text: str) -> bool:
    return any(pattern.search(text) for pattern in _TRACKED_WRITE_HINTS)


def _result_fingerprint(result: Any) -> tuple[str, str]:
    if isinstance(result, str):
        text = result
    else:
        try:
            text = json.dumps(result, sort_keys=True, ensure_ascii=False, default=str)
        except Exception:
            text = str(result)
    parsed = None
    if isinstance(result, dict):
        parsed = result
    elif isinstance(result, str):
        try:
            parsed = json.loads(result)
        except Exception:
            parsed = None
    status = "unknown"
    if isinstance(parsed, dict):
        status = str(parsed.get("status") or parsed.get("state") or parsed.get("result") or "unknown")
        if "error" in parsed and status == "unknown":
            status = "error"
    return _short_hash(text), status


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12]


def _evidence_hash(evidence: Mapping[str, Any]) -> str:
    snapshot = {
        key: value
        for key, value in evidence.items()
        if not str(key).startswith("no_progress_loop")
    }
    try:
        payload = json.dumps(snapshot, sort_keys=True, default=str, ensure_ascii=False)
    except Exception:
        payload = str(snapshot)
    return _short_hash(payload)


def _loop_block_message(
    *,
    fingerprint: Mapping[str, Any],
    evidence: Mapping[str, Any],
    repeat_cycles: int,
    mode: str,
) -> str:
    latest_user_request = str(evidence.get("latest_user_request") or "").strip() or "unknown"
    last_new_fact = _summarize_last_new_fact(evidence, fingerprint)
    invariant_status = _summarize_invariant_status(evidence)
    next_action = "Change strategy, summarize the new facts, and choose one concrete next step."
    decision = "stop" if mode == "blocked" else "self-audit"
    return (
        f"Blocked no-progress loop: repeated {fingerprint.get('action')} action detected for "
        f"{fingerprint.get('tool_name')} in {fingerprint.get('cwd')}.\n"
        f"Current objective: {latest_user_request}\n"
        f"Latest real user request: {latest_user_request}\n"
        f"Last new fact: {last_new_fact}\n"
        f"Current invariant status: {invariant_status}\n"
        f"Next single action: {next_action}\n"
        f"Decision: {decision} repeated action after {repeat_cycles} no-progress cycle(s).\n"
        "Do not repeat the same ineffective check; change strategy or ask for a concrete next step."
    )


def _summarize_last_new_fact(evidence: Mapping[str, Any], fingerprint: Mapping[str, Any]) -> str:
    if evidence.get("ci_evidence_fetched_for_live_pr_head"):
        ci_sha = evidence.get("ci_checkout_sha") or "unknown"
        return f"CI evidence for live PR head at {ci_sha}"
    if evidence.get("pr_head_verified") is True:
        return f"verified live PR head {evidence.get('live_pr_head_sha') or 'unknown'}"
    if evidence.get("pre_edit_git_status"):
        return "pre-edit git status captured"
    return f"no new fact beyond repeated {fingerprint.get('action')} output"


def _summarize_invariant_status(evidence: Mapping[str, Any]) -> str:
    parts: list[str] = []
    if "pr_head_verified" in evidence:
        parts.append("PR head verified" if evidence.get("pr_head_verified") else "PR head not verified")
    if "review_thread_patch_clean" in evidence:
        parts.append("review-thread patch clean" if evidence.get("review_thread_patch_clean") else "review-thread patch still dirty")
    if "ci_evidence_ci_sha_matches_live_pr_head" in evidence:
        parts.append(
            "CI SHA matches live PR head" if evidence.get("ci_evidence_ci_sha_matches_live_pr_head")
            else "CI SHA stale relative to live PR head"
        )
    if not parts:
        return "no active invariant mismatch recorded"
    return "; ".join(parts)
