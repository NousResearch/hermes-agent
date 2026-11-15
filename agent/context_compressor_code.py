"""Bounded execute_code result stubs; execution outcomes must outlive stdout pruning."""

import json
from typing import Any


def _execution_payload(content: str) -> dict:
    # Tool results can carry a subdirectory-instruction hint after the JSON envelope.
    try:
        payload, _ = json.JSONDecoder().raw_decode(content.lstrip())
    except (ValueError, RecursionError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _kernel_note(payload: dict, status: str) -> str:
    kernel = payload.get("kernel")
    if not isinstance(kernel, dict):
        return ""
    if kernel.get("state_lost") is True or (
        status in ("timeout", "interrupted")
        and (kernel.get("mode") == "session" or kernel.get("remote") is True)
    ):
        return "kernel state lost"
    return next((note for key, note in (("state_reset", "kernel state reset"), ("ended", "kernel ended"))
                 if kernel.get(key) is True), "")


def _execution_outcome(payload: dict) -> str:
    from agent.context_compressor import _failure_suffix, _str_arg

    status = payload.get("status")
    status = status if isinstance(status, str) else ""
    label = {"error": "FAILED", "timeout": "TIMEOUT", "interrupted": "INTERRUPTED"}.get(status, "")
    error, exit_code = payload.get("error"), payload.get("exit_code")
    nonzero_exit = isinstance(exit_code, int) and not isinstance(exit_code, bool) and exit_code != 0
    if not label and (error or payload.get("success") is False or nonzero_exit):
        label = "FAILED"
    errors = payload.get("tool_errors")
    inner = next((e for e in errors if isinstance(e, dict) and e.get("error")), None) if isinstance(errors, list) else None
    notes = "; ".join(filter(None, (label, "inner tool failed" if inner else "", _kernel_note(payload, status))))
    if not notes:
        return ""
    reason: Any = error
    if not reason and inner:
        reason = f"{_str_arg(inner, 'tool', 'tool')}: {inner['error']}"
    if not reason and nonzero_exit:
        reason = f"exit code {exit_code}"
    if isinstance(reason, str) and reason.lstrip().startswith("Traceback (most recent call last):"):
        reason = reason.rstrip().splitlines()[-1]
    return _failure_suffix(reason, notes)


def summarize_execute_code(name, args, content, content_len, line_count):
    from agent.context_compressor import _PRUNE_MIN_CHARS, _str_arg, _summarize_refused_tool_result

    # Fallback anchor collection also summarizes results that an earlier prune already reduced.
    if content.startswith("[execute_code] ") and len(content) < _PRUNE_MIN_CHARS and "\n" not in content:
        return content
    payload = _execution_payload(content)
    # Preserve the shared refusal semantics even when trailing hints hid the envelope from its first pass.
    error = payload.get("error")
    if payload.get("status") in ("blocked", "pending_approval") or (
        isinstance(error, str) and error.lstrip().startswith("BLOCKED")
    ):
        refused = _summarize_refused_tool_result(name, args, json.dumps(payload))
        if refused is not None:
            return refused
    code = _str_arg(args, "code")
    preview = code[:60].replace("\n", " ") + ("..." if len(code) > 60 else "")
    stub = f"[execute_code] `{preview}` ({line_count} lines output)"
    suffix = _execution_outcome(payload)
    if not suffix:
        return stub
    # A later prune/fallback must not mistake this stub for an unclassified raw result.
    return (stub + suffix)[:_PRUNE_MIN_CHARS - 1]
