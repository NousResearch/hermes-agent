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
from agent.providers.anthropic_adapter import convert_messages_to_anthropic


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

    def test_replay_falls_back_gracefully_without_ordered_blocks(self):
        """Without the ordered-block channel, conversion must not crash.

        The channel is intentionally NOT persisted to state.db (in-memory
        only): a session reloaded from disk after a crash loses the field
        and falls back to reconstruction. That replay may take one HTTP 400,
        which the thinking-signature recovery (#43667) absorbs by stripping
        reasoning_details and retrying. This test pins the fallback shape:
        conversion still produces a valid assistant message from the
        parallel reasoning_details + tool_calls fields.
        """
        response = _interleaved_response()
        transport = get_transport("anthropic_messages")
        normalized = transport.normalize_response(response)
        assistant_msg = _stored_assistant_message(normalized)
        # Simulate a disk reload: the in-memory-only channel is gone.
        assistant_msg.pop("anthropic_content_blocks", None)

        messages = [
            assistant_msg,
            {"role": "tool", "tool_call_id": "toolu_1", "content": "a ok"},
            {"role": "tool", "tool_call_id": "toolu_2", "content": "b ok"},
        ]
        _system, anthropic_messages = convert_messages_to_anthropic(
            messages, base_url=None, model="claude-opus-4-8",
        )
        assistant_out = [m for m in anthropic_messages if m.get("role") == "assistant"]
        assert assistant_out, "no assistant message in converted output"
        content = assistant_out[-1]["content"]
        assert isinstance(content, list) and content, "fallback produced empty content"
        # Reconstruction keeps both tool_use blocks (answered by results).
        tool_ids = [b.get("id") for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
        assert set(tool_ids) == {"toolu_1", "toolu_2"}


class TestInterleavedReplayCredentialRedaction:
    """The verbatim-replay fast path must not leak un-redacted secrets.

    anthropic_content_blocks captures each tool_use ``input`` from the RAW API
    response (normalize_response), which is NOT credential-redacted. The
    parallel tool_calls[].function.arguments IS redacted at storage time
    (build_assistant_message, #19798). If the fast path replays the block's raw
    input verbatim, a secret the model inlined into a tool call rides back onto
    the wire — even though it is redacted everywhere else in history. The fix
    re-sources tool_use input from the redacted tool_calls map by id.
    """

    def test_tool_use_input_resourced_from_redacted_tool_calls(self):
        REDACTED = "[REDACTED_SECRET]"
        # Ordered channel: raw input carries the live secret (as captured from
        # the unredacted API response).
        ordered = [
            {"type": "thinking", "thinking": "Call the API.", "signature": "sig-AAA"},
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "terminal",
                "input": {"command": "curl -H 'Authorization: Bearer sk-LIVE-SECRET-123'"},
            },
            {"type": "thinking", "thinking": "Now the second call.", "signature": "sig-BBB"},
            {
                "type": "tool_use",
                "id": "toolu_2",
                "name": "terminal",
                "input": {"command": "echo done"},
            },
        ]
        # Stored tool_calls: arguments already redacted (the #19798 path).
        assistant_msg = {
            "role": "assistant",
            "content": "",
            "reasoning_details": [b for b in ordered if b["type"] == "thinking"],
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {
                        "name": "terminal",
                        "arguments": json.dumps(
                            {"command": f"curl -H 'Authorization: Bearer {REDACTED}'"}
                        ),
                    },
                },
                {
                    "id": "toolu_2",
                    "type": "function",
                    "function": {
                        "name": "terminal",
                        "arguments": json.dumps({"command": "echo done"}),
                    },
                },
            ],
            "anthropic_content_blocks": ordered,
        }
        messages = [
            {"role": "user", "content": "Hit the API twice."},
            assistant_msg,
            {"role": "tool", "tool_call_id": "toolu_1", "content": "200 OK"},
            {"role": "tool", "tool_call_id": "toolu_2", "content": "done"},
        ]

        _system, anthropic_messages = convert_messages_to_anthropic(
            messages, base_url=None, model="claude-opus-4-8",
        )
        assistant_out = [m for m in anthropic_messages if m.get("role") == "assistant"]
        assert assistant_out, "no assistant message in converted output"
        blocks = assistant_out[-1]["content"]

        tool_uses = {b["id"]: b for b in blocks if b.get("type") == "tool_use"}
        assert set(tool_uses) == {"toolu_1", "toolu_2"}, "tool_use blocks missing/renamed"

        # The replayed input must be the REDACTED value, not the live secret.
        replayed_cmd = tool_uses["toolu_1"]["input"]["command"]
        assert "sk-LIVE-SECRET-123" not in replayed_cmd, (
            "Un-redacted secret leaked onto the wire via the verbatim-replay "
            "fast path. tool_use input must be re-sourced from the redacted "
            "tool_calls map, not the raw captured block."
        )
        assert REDACTED in replayed_cmd

        # Interleave order is still preserved (the reason the channel exists).
        order = [
            ("thinking", b.get("signature")) if b.get("type") == "thinking"
            else ("tool_use", b.get("id"))
            for b in blocks if b.get("type") in ("thinking", "tool_use")
        ]
        assert order == [
            ("thinking", "sig-AAA"),
            ("tool_use", "toolu_1"),
            ("thinking", "sig-BBB"),
            ("tool_use", "toolu_2"),
        ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
