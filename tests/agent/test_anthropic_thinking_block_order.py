"""Regression test for the Anthropic interleaved thinking-block 400.

Reproduces: HTTP 400 ``messages.N.content.M: thinking or redacted_thinking
blocks in the latest assistant message cannot be modified. These blocks must
remain as they were in the original response.``

Root cause under test
----------------------
With adaptive / interleaved thinking (Claude 4.6+, e.g. Opus 4.8), a single
assistant turn can emit content blocks in an interleaved order::

    thinking_1 (signed) · tool_use_1 · thinking_2 (signed) · tool_use_2

Anthropic signs each thinking block against the turn content that precedes it
at its position.  ``thinking_2`` is signed with ``tool_use_1`` before it.

``AnthropicTransport.normalize_response`` (agent/transports/anthropic.py)
splits the turn into two *parallel* lists — ``reasoning_details`` (thinking
blocks) and ``tool_calls`` (tool_use blocks) — discarding the cross-type
ordering.  ``run_agent`` stores those as separate fields on the assistant
message.  On replay, ``_convert_assistant_message`` (agent/anthropic_adapter.py)
rebuilds the content as ``[all thinking][text][all tool_use]``, which reorders
``thinking_2`` ahead of ``tool_use_1``.  The signature no longer matches its
original position, so Anthropic rejects the latest assistant message with the
400 above.

This test asserts that an interleaved turn round-trips through
normalize_response -> stored message -> convert_messages_to_anthropic with its
block order preserved.  It FAILS on the current code (documenting the bug) and
should PASS once block ordering is preserved on replay.
"""

import json
from types import SimpleNamespace

import pytest

from agent.transports import get_transport
from agent.anthropic_adapter import convert_messages_to_anthropic


def _thinking_block(text: str, signature: str) -> SimpleNamespace:
    """A signed Anthropic thinking block, shaped like the SDK object."""
    return SimpleNamespace(type="thinking", thinking=text, signature=signature)


def _tool_use_block(block_id: str, name: str, payload: dict) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=payload)


def _interleaved_response() -> SimpleNamespace:
    """An assistant turn with thinking interleaved between two tool_use blocks."""
    return SimpleNamespace(
        content=[
            _thinking_block("Plan: inspect file A first.", "sig-AAA"),
            _tool_use_block("toolu_1", "read_file", {"path": "a.py"}),
            _thinking_block("A looked fine; now inspect B.", "sig-BBB"),
            _tool_use_block("toolu_2", "read_file", {"path": "b.py"}),
        ],
        stop_reason="tool_use",
        usage=None,
    )


def _stored_assistant_message(normalized) -> dict:
    """Reconstruct the OpenAI-style assistant message the way run_agent stores it.

    run_agent.py persists assistant turns as separate fields: content,
    reasoning_details (from provider_data), and tool_calls.  See
    run_agent.py L1513-1516 and hermes_state.py.
    """
    provider_data = normalized.provider_data or {}
    tool_calls = []
    for tc in (normalized.tool_calls or []):
        tool_calls.append({
            "id": tc.id,
            "type": "function",
            "function": {"name": tc.name, "arguments": tc.arguments},
        })
    msg = {
        "role": "assistant",
        "content": normalized.content or "",
        "reasoning_details": provider_data.get("reasoning_details"),
        "tool_calls": tool_calls,
    }
    # build_assistant_message lifts the verbatim ordered-block channel onto
    # the stored message; mirror that here.
    blocks = provider_data.get("anthropic_content_blocks")
    if blocks:
        msg["anthropic_content_blocks"] = blocks
    return msg


def _original_block_order(response) -> list:
    """The (type, key) sequence of the original interleaved response."""
    order = []
    for b in response.content:
        if b.type == "thinking":
            order.append(("thinking", b.signature))
        elif b.type == "tool_use":
            order.append(("tool_use", b.id))
    return order


def _replayed_block_order(assistant_content) -> list:
    order = []
    for b in assistant_content:
        if not isinstance(b, dict):
            continue
        if b.get("type") in ("thinking", "redacted_thinking"):
            order.append(("thinking", b.get("signature")))
        elif b.get("type") == "tool_use":
            order.append(("tool_use", b.get("id")))
    return order


class TestInterleavedThinkingBlockOrder:
    def test_normalize_response_loses_interleaving(self):
        """Confirm the lossy split: normalize_response stores thinking and
        tool_use in independent fields with no positional linkage."""
        transport = get_transport("anthropic_messages")
        normalized = transport.normalize_response(_interleaved_response())

        # Both thinking blocks are captured...
        details = (normalized.provider_data or {}).get("reasoning_details")
        assert details is not None and len(details) == 2
        # ...and both tool calls...
        assert normalized.tool_calls is not None and len(normalized.tool_calls) == 2
        # ...but they live in separate fields. There is no single ordered
        # structure recording that thinking_2 sat between the two tool calls.
        # (This is the structural precondition for the reorder bug.)

    def test_interleaved_order_preserved_on_replay(self):
        """The latest assistant message must replay blocks in their ORIGINAL
        order, or Anthropic rejects the signed thinking blocks with a 400.

        FAILS on current code: _convert_assistant_message front-loads all
        thinking blocks, producing
            thinking_1 · thinking_2 · tool_use_1 · tool_use_2
        instead of the original
            thinking_1 · tool_use_1 · thinking_2 · tool_use_2
        """
        response = _interleaved_response()
        original_order = _original_block_order(response)

        transport = get_transport("anthropic_messages")
        normalized = transport.normalize_response(response)
        assistant_msg = _stored_assistant_message(normalized)

        # Build a minimal conversation where this assistant turn is the LATEST
        # assistant message (the one whose signed blocks are sent verbatim).
        messages = [
            {"role": "user", "content": "Inspect a.py and b.py."},
            assistant_msg,
            {"role": "tool", "tool_call_id": "toolu_1", "content": "a.py: ok"},
            {"role": "tool", "tool_call_id": "toolu_2", "content": "b.py: ok"},
        ]

        _system, anthropic_messages = convert_messages_to_anthropic(
            messages,
            base_url=None,             # direct Anthropic
            model="claude-opus-4-8",   # adaptive thinking family
        )

        # Find the (latest) assistant message in the converted output.
        assistant_out = [m for m in anthropic_messages if m.get("role") == "assistant"]
        assert assistant_out, "no assistant message in converted output"
        replayed_order = _replayed_block_order(assistant_out[-1]["content"])

        assert replayed_order == original_order, (
            "Interleaved thinking/tool_use order was not preserved on replay.\n"
            f"  original: {original_order}\n"
            f"  replayed: {replayed_order}\n"
            "Anthropic signs thinking blocks against their original position; "
            "reordering invalidates the signature -> HTTP 400 'thinking blocks "
            "in the latest assistant message cannot be modified'."
        )

    def test_interleaved_order_survives_db_roundtrip(self, tmp_path):
        """The ordered-block channel must survive SQLite persistence + reload.

        This is the exact path that fails after a gateway crash: the session
        is reloaded from state.db via get_messages_as_conversation, then
        replayed. If the verbatim block list is dropped or not deserialized,
        the reconstruction reorders signed thinking blocks -> HTTP 400.
        """
        import hermes_state

        response = _interleaved_response()
        original_order = _original_block_order(response)

        transport = get_transport("anthropic_messages")
        normalized = transport.normalize_response(response)
        assistant_msg = _stored_assistant_message(normalized)

        db = hermes_state.SessionDB(tmp_path / "state.db")
        sid = "sess_roundtrip"
        db.create_session(sid, source="test")
        db.append_message(
            session_id=sid,
            role="assistant",
            content=assistant_msg["content"],
            tool_calls=assistant_msg["tool_calls"],
            reasoning_details=assistant_msg.get("reasoning_details"),
            anthropic_content_blocks=assistant_msg.get("anthropic_content_blocks"),
        )
        db.append_message(session_id=sid, role="tool", tool_call_id="toolu_1", content="a ok")
        db.append_message(session_id=sid, role="tool", tool_call_id="toolu_2", content="b ok")

        # Reload via the conversation-restore path used on resume / crash recovery.
        loaded = db.get_messages_as_conversation(sid)
        reloaded_assistant = [m for m in loaded if m.get("role") == "assistant"]
        assert reloaded_assistant, "no assistant message after DB reload"
        # The ordered-block channel must come back as a deserialized list.
        blocks = reloaded_assistant[0].get("anthropic_content_blocks")
        assert isinstance(blocks, list) and len(blocks) == 4, (
            "anthropic_content_blocks was not persisted/deserialized correctly"
        )

        _system, anthropic_messages = convert_messages_to_anthropic(
            loaded, base_url=None, model="claude-opus-4-8",
        )
        assistant_out = [m for m in anthropic_messages if m.get("role") == "assistant"]
        assert assistant_out, "no assistant message in converted output"
        replayed_order = _replayed_block_order(assistant_out[-1]["content"])

        assert replayed_order == original_order, (
            "Interleaved block order was lost across the SQLite round-trip.\n"
            f"  original: {original_order}\n"
            f"  replayed: {replayed_order}"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
