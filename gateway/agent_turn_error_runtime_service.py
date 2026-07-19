"""Production agent-turn error path for gateway foreground runs.

Promoted from ``GatewayRunner._handle_message_with_agent`` except block:
stop typing, crash-resilient user-turn persistence, and status-aware
user-facing error copy. Does not own handler ``finally`` cleanup.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(slots=True)
class GatewayAgentTurnErrorResult:
    """User-visible error response after an agent-turn exception."""

    response: str


async def handle_gateway_agent_turn_error(
    *,
    runner: Any,
    event: Any,
    source: Any,
    session_entry: Any,
    session_key: str,
    history: list | None,
    message_text: str | None,
    persist_user_message: Any,
    persist_user_timestamp: Any,
    exc: BaseException,
    logger: Any,
) -> GatewayAgentTurnErrorResult:
    """Handle a failed agent turn; return a safe user-facing error string."""
    e = exc
    history = list(history) if history is not None else []

    # Stop typing indicator on error too, retaining Slack thread/workspace
    # routing so a failed turn cannot leave its status visible.
    try:
        _err_adapter = runner._adapter_for_source(source)
        _stop_with_metadata = getattr(
            type(_err_adapter), "_stop_typing_with_metadata", None
        )
        _stop_typing = getattr(type(_err_adapter), "stop_typing", None)
        if _err_adapter and callable(_stop_with_metadata):
            await _err_adapter._stop_typing_with_metadata(
                source.chat_id,
                runner._thread_metadata_for_source(
                    source, runner._reply_anchor_for_event(event)
                ),
            )
        elif _err_adapter and callable(_stop_typing):
            await _err_adapter.stop_typing(source.chat_id)
    except Exception:
        pass
    logger.exception("Agent error in session %s", session_key)
    # Crash-resilience for failures that happen before AIAgent enters
    # run_conversation() (for example: provider/httpx client init
    # failures). In that path the agent cannot persist the current
    # inbound turn itself, so append the user message here once. If the
    # agent already reached its early turn-start persistence, the latest
    # transcript user row will match and we skip the duplicate.
    try:
        if message_text is not None and session_entry is not None:
            _already_persisted = False
            try:
                _recent_transcript = await runner.async_session_store.load_transcript(session_entry.session_id)
            except Exception:
                _recent_transcript = []
            for _msg in reversed(_recent_transcript[-10:]):
                if _msg.get("role") == "user":
                    _expected_user_content = (
                        persist_user_message
                        if persist_user_message is not None
                        else message_text
                    )
                    _already_persisted = (_msg.get("content") == _expected_user_content)
                    break
            if not _already_persisted:
                _user_entry = {
                    "role": "user",
                    "content": (
                        persist_user_message
                        if persist_user_message is not None
                        else message_text
                    ),
                    "timestamp": (
                        persist_user_timestamp
                        if persist_user_timestamp is not None
                        else time.time()
                    ),
                }
                if getattr(event, "message_id", None):
                    _user_entry["message_id"] = str(event.message_id)
                await runner.async_session_store.append_to_transcript(
                    session_entry.session_id,
                    _user_entry,
                )
    except Exception:
        logger.debug("Failed to persist inbound user message after agent exception", exc_info=True)
    # Log full details server-side only; never expose raw exception
    # types or messages to end users (info-leakage risk).
    status_hint = ""
    status_code = getattr(e, "status_code", None)
    _hist_len = len(history or [])
    if status_code == 401:
        status_hint = " Check your API key or run `claude /login` to refresh OAuth credentials."
    elif status_code == 402:
        status_hint = " Your API balance or quota is exhausted. Check your provider dashboard."
    elif status_code == 429:
        # Check if this is a plan usage limit (resets on a schedule) vs a transient rate limit
        _err_body = getattr(e, "response", None)
        _err_json = {}
        try:
            if _err_body is not None:
                _err_json = _err_body.json().get("error", {})
                if not isinstance(_err_json, dict):
                    _err_json = {}
        except Exception:
            pass
        if _err_json.get("type") == "usage_limit_reached":
            _resets_in = _err_json.get("resets_in_seconds")
            if _resets_in and _resets_in > 0:
                import math
                _hours = math.ceil(_resets_in / 3600)
                status_hint = f" Your plan's usage limit has been reached. It resets in ~{_hours}h."
            else:
                status_hint = " Your plan's usage limit has been reached. Please wait until it resets."
        else:
            status_hint = " You are being rate-limited. Please wait a moment and try again."
    elif status_code == 529:
        status_hint = " The API is temporarily overloaded. Please try again shortly."
    elif status_code in {400, 500}:
        # 400 with a large session is context overflow.
        # 500 with a large session often means the payload is too large
        # for the API to process — treat it the same way.
        if _hist_len > 50:
            return GatewayAgentTurnErrorResult(
                response=(
                    "⚠️ Session too large for the model's context window.\n"
                    "Use /compact to compress the conversation, or "
                    "/reset to start fresh."
                )
            )
        elif status_code == 400:
            status_hint = " The request was rejected by the API."
    return GatewayAgentTurnErrorResult(
        response=(
            f"Sorry, I encountered an unexpected error.{status_hint}\n"
            "Try again or use /reset to start a fresh session."
        )
    )
