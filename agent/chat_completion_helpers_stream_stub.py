"""Partial-stream stubs: the synthetic ``finish_reason="length"`` response the chat-completions
stream returns when a generation did not complete, so the turn loop continues or re-requests
instead of accepting the partial output as a final turn."""

from __future__ import annotations

import logging
from types import SimpleNamespace

from agent.errors import EmptyStreamError
from hermes_constants import FINISH_REASON_LENGTH, PARTIAL_STREAM_STUB_ID

logger = logging.getLogger("agent.chat_completion_helpers")


def _build_partial_stream_stub(role, full_content, full_reasoning, model_name, usage_obj, *,
    dropped_tool_names=None, overflow_terminal=False, api_mode=None, clean_eof=False):
    """Stub for an SSE stream that ended without ``finish_reason`` after
    delivering content. Tagged ``PARTIAL_STREAM_STUB_ID`` + ``FINISH_REASON_LENGTH``
    so the loop enters its continuation/retry path instead of accepting
    truncated output as a complete turn (#32086).

    ``overflow_terminal`` (``full_content=None``): the stream died on a
    context-overflow error. Seeding the recovered text as a continuation stub
    would grow every later request into the same overflow (#106260); the loop
    treats the marker as terminal and ends the turn via the recovery contract.

    ``api_mode="anthropic_messages"`` returns a Messages-shaped stub (``content``
    block list + ``stop_reason="max_tokens"``) so AnthropicTransport validates it
    and the loop continues instead of entering the invalid-response retry ladder
    (#45908). Empty content keeps one empty text block: validate_response rejects
    an empty list for ``max_tokens``.

    ``clean_eof``: the stream ended with no transport exception and no
    ``finish_reason`` (server/intermediary closed cleanly). Only the two
    clean-EOF sites in ``_finish_chat_stream`` pass True; the stub built after a
    real transport exception keeps False so the loop can word the two failure
    modes differently (#102766).
    """
    if api_mode == "anthropic_messages":
        return SimpleNamespace(
            id=PARTIAL_STREAM_STUB_ID,
            type="message",
            role=role,
            model=model_name,
            content=[SimpleNamespace(type="text", text=full_content or "")],
            stop_reason="max_tokens",
            stop_sequence=None,
            usage=usage_obj,
            _dropped_tool_names=dropped_tool_names or None,
            _overflow_terminal=overflow_terminal,
            _clean_eof=clean_eof,
        )
    return SimpleNamespace(
        id=PARTIAL_STREAM_STUB_ID,
        model=model_name,
        choices=[SimpleNamespace(
            index=0,
            message=SimpleNamespace(role=role, content=full_content, tool_calls=None,
                reasoning_content=full_reasoning),
            finish_reason=FINISH_REASON_LENGTH,
        )],
        usage=usage_obj,
        _dropped_tool_names=dropped_tool_names or None,
        _overflow_terminal=overflow_terminal,
        _clean_eof=clean_eof,
    )


def interrupted_stream_response(role, content_parts, reasoning_parts, refusal_parts, tool_calls_acc,
    finish_reason, full_content, full_reasoning, model_name, usage_obj):
    """A provider-interrupted generation (``_INTERRUPTED_FINISH_REASONS``) is never a final
    answer: nothing delivered → EmptyStreamError (fresh-connection stream retry); otherwise
    a partial-stream stub, so the loop asks for a continuation and tool calls are re-requested,
    never executed, even when their JSON parses (the batch or the intent may be incomplete).
    No ``dropped_tool_names``: that nudge tells the model its call was too LARGE, which is
    not what happened here."""
    if not (content_parts or reasoning_parts or refusal_parts or tool_calls_acc):
        raise EmptyStreamError(
            f"Provider stream ended with finish_reason={finish_reason} and no content "
            "(generation interrupted upstream).")
    dropped = [(tool_calls_acc[idx]["function"]["name"] or "?") for idx in sorted(tool_calls_acc)]
    logger.warning(
        "Provider ended the stream with finish_reason=%s (generation interrupted upstream) after "
        "partial output%s; requesting a continuation instead of accepting it as final.",
        finish_reason, f" (tool calls not executed: {dropped})" if dropped else "")
    return _build_partial_stream_stub(role, full_content, full_reasoning, model_name, usage_obj)
