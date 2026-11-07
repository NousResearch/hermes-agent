"""Completion checks for the native Anthropic Messages stream (``_StreamingCall._call_anthropic``)."""

import logging

from agent.errors import EmptyStreamError

logger = logging.getLogger(__name__)


def check_anthropic_message(message, *, tool_drop: bool = True):
    """Raise EmptyStreamError for a message the stream never completed: no
    content and no stop_reason (eventless -> retry), or with ``tool_drop`` a
    ``tool_use`` block and no stop_reason — the SSE closed mid tool call and
    its input is a partial snapshot (usually ``{}``), so raising blocks the
    empty-args execution (bounded retry, or stub/continuation after text)."""
    content = getattr(message, "content", None)
    if not content and getattr(message, "stop_reason", None) is None:
        raise EmptyStreamError(
            "Provider returned an empty stream with no stop_reason (possible upstream error or malformed event stream).")
    if tool_drop and getattr(message, "stop_reason", None) is None and any(
        getattr(block, "type", None) == "tool_use" for block in content or []):
        raise EmptyStreamError(
            "Stream ended with no stop_reason while a tool_use block was still incomplete; "
            "treating as a mid-tool-call stream drop (#80498).")
    return message


def completed_or_drop_stub(message, saw_message_delta: bool):
    """message_stop arrived but message_delta (stop_reason + final usage) never did: the stream
    was cut, not finished. Return the same clean-EOF stub the chat_completions wire uses (#32086)
    so the loop continues instead of stamping the truncated text as a finished reply."""
    # Port of MiniMax-AI/minimax-code#566.
    if saw_message_delta or getattr(message, "stop_reason", None) is not None:
        return message
    from agent.chat_completion_helpers import _build_partial_stream_stub

    blocks = getattr(message, "content", None) or []
    text = "".join(getattr(b, "text", "") or "" for b in blocks if getattr(b, "type", None) == "text")
    logger.warning(
        "Clean EOF, no stop_reason: Anthropic stream sent message_stop without message_delta after "
        "delivering %s chars of text. The server or a proxy closed the stream cleanly.", len(text))
    return _build_partial_stream_stub("assistant", text or None, None, getattr(message, "model", None),
        getattr(message, "usage", None), api_mode="anthropic_messages", clean_eof=True)
