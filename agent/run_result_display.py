"""Display helpers for AIAgent.run_conversation() result envelopes.

These helpers live at UI/presentation boundaries. They intentionally do not
mutate conversation history and do not reinterpret runtime failures as
assistant-authored ``final_response`` text.
"""

from __future__ import annotations

import re
from typing import Any, Literal, cast

AgentResultStatus = Literal["complete", "partial", "error", "interrupted"]


def _as_dict(result: object) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    return cast(dict[str, Any], result)


def _redact(value: object) -> str:
    try:
        from agent.redact import redact_sensitive_text

        return redact_sensitive_text(str(value or ""), force=True)
    except Exception:
        return "[redacted: redaction failed]"


def _clean_detail(value: object, *, max_len: int = 1200) -> str:
    # Preserve useful provider line breaks/request IDs while normalizing
    # excessive horizontal whitespace.
    text = _redact(value).strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def agent_result_status(result: object) -> AgentResultStatus:
    data = _as_dict(result)
    if data is None:
        return "complete"
    if data.get("interrupted"):
        return "interrupted"
    if data.get("failed"):
        return "error"
    if data.get("partial"):
        return "partial"
    response = data.get("final_response")
    if (response is None or response == "") and (data.get("error") or data.get("display_error")):
        return "error"
    return "complete"


def agent_result_display_error(result: object) -> str:
    data = _as_dict(result)
    if data is None:
        return ""

    display_error = data.get("display_error")
    if display_error:
        return _clean_detail(display_error)

    if data.get("failed") or data.get("partial") or data.get("error"):
        detail = (
            data.get("error_summary")
            or data.get("error")
            or "The agent stopped before producing a response."
        )
        label = (
            "Request incomplete"
            if data.get("partial") and not data.get("failed")
            else "Request failed"
        )
        return f"⚠️ {label}: {_clean_detail(detail)}"

    return ""


def agent_result_error_metadata(result: object) -> dict[str, Any]:
    data = _as_dict(result)
    if data is None:
        return {}
    keys = ("error_kind", "status_code", "provider", "model", "base_url", "error_summary")
    metadata: dict[str, Any] = {}
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        metadata[key] = _clean_detail(value) if isinstance(value, str) else value
    return metadata


def agent_result_visible_text(result: object) -> str:
    if result is None:
        return ""
    data = _as_dict(result)
    if data is None:
        return str(result)

    response = data.get("final_response")
    if response and response != "(empty)":
        return str(response)

    if response == "(empty)":
        return (
            "⚠️ The model returned no visible response. "
            "Try again, rephrase, or switch models."
        )

    return agent_result_display_error(data)
