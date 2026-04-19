"""Stable JSON payloads for provider/SDK failures at the tool boundary.

Maps content-filter and policy-class failures to short, non-leaky messages so
agent loops and RL trajectory exports do not retain blocked user text verbatim.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

_CONTENT_FILTER_CODES = frozenset({
    "content_policy_violation",
    "content_filter",
    "invalid_prompt",
})

_POLICY_SUBSTRINGS = (
    "content_policy",
    "content policy",
    "content filtering",
    "content_filter",
    "responsible_ai_policy",
    "content management policy",
    "blocked by azure",
    "prompt filtered",
    "jailbreak",
)


def _openai_nested_code(exc: BaseException) -> Optional[str]:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            code = err.get("code")
            if isinstance(code, str):
                return code
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            data = getattr(resp, "json", lambda: None)()
            if isinstance(data, dict):
                err = data.get("error")
                if isinstance(err, dict) and isinstance(err.get("code"), str):
                    return err["code"]
        except Exception:
            pass
    return None


def classify_provider_exception(exc: BaseException) -> Optional[str]:
    """Return a machine-readable code for *exc*, or None if unknown."""
    nested = _openai_nested_code(exc)
    if nested:
        low = nested.lower()
        if low in _CONTENT_FILTER_CODES or "content" in low and "violat" in low:
            return "content_filter"
        if nested == "invalid_prompt":
            return "invalid_prompt"

    msg_l = str(exc).lower()
    type_name = type(exc).__name__.lower()

    if "policy" in type_name and "content" in msg_l:
        return "content_filter"

    if any(s in msg_l for s in _POLICY_SUBSTRINGS):
        return "content_filter"

    if nested and nested.lower() == "invalid_prompt":
        return "invalid_prompt"

    return None


def _sanitized_message(exc: BaseException, code: Optional[str]) -> str:
    """Human-readable message; omits verbatim blocked prompts when *code* indicates policy."""
    if code == "content_filter":
        return (
            "Request blocked by provider content or safety policy "
            "(details omitted)."
        )
    if code == "invalid_prompt":
        return "Request rejected as invalid by the provider (details omitted)."
    raw = str(exc)
    try:
        from agent.redact import redact_sensitive_text

        return redact_sensitive_text(raw)
    except Exception:
        return raw


def safe_tool_error_payload(exc: BaseException) -> Dict[str, Any]:
    """Build a dict suitable for ``json.dumps`` as a tool result."""
    code = classify_provider_exception(exc)
    msg = _sanitized_message(exc, code)
    out: Dict[str, Any] = {"error": msg}
    if code:
        out["error_code"] = code
    return out


def format_tool_boundary_error(function_name: str, exc: BaseException) -> str:
    """JSON string for exceptions raised while executing *function_name*."""
    payload = safe_tool_error_payload(exc)
    payload["error"] = f"Error executing tool '{function_name}': {payload['error']}"
    return json.dumps(payload, ensure_ascii=False)


def format_registry_dispatch_error(tool_name: str, exc: BaseException) -> str:
    """JSON string for failures inside ``registry.dispatch``."""
    typ = type(exc).__name__
    payload = safe_tool_error_payload(exc)
    if payload.get("error_code"):
        payload["error"] = f"Tool execution failed ({tool_name}): {payload['error']}"
    else:
        payload["error"] = f"Tool execution failed: {typ}: {payload['error']}"
    return json.dumps(payload, ensure_ascii=False)
