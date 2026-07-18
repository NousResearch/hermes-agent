"""Regression test: plain-turn (no tool_use) Anthropic thinking must replay.

Scope
-----
The Anthropic Messages path has two replay mechanisms for assistant thinking:

1. ``anthropic_content_blocks`` — a verbatim, order-preserving block list,
   populated only for turns that interleave SIGNED thinking with tool_use
   (covered by tests/agent/test_anthropic_thinking_block_order.py).
2. ``reasoning_details`` — the preserved-thinking channel every turn should
   hit: ``AnthropicTransport.normalize_response`` collects each thinking block
   (with signature) into ``provider_data["reasoning_details"]``,
   ``build_assistant_message`` copies it onto the stored assistant message, and
   ``_convert_assistant_message`` replays it through
   ``_extract_preserved_thinking_blocks``.

The plain-turn path (thinking + text, no tool_use) depends *entirely* on (2):
if the stored turn loses ``reasoning_details``, the next-turn request ships the
previous assistant turn as bare text — the model silently loses its prior
chain-of-thought (observed live: a two-turn "remember what you reasoned"
conversation fails on the second turn).

These tests pin the whole chain for plain turns so the silent-loss shape cannot
regress: normalize -> stored message -> converted replay request.
"""

from types import SimpleNamespace

from agent.transports import get_transport
from agent.anthropic_adapter import convert_messages_to_anthropic

SIG = "sig-plain-4340"


def _plain_turn_response() -> SimpleNamespace:
    """A thinking+text assistant turn with NO tool_use (shaped like the SDK object)."""
    return SimpleNamespace(
        content=[
            SimpleNamespace(
                type="thinking",
                thinking="Picking five numbers: 7, 23, 41, 56, 88. Reveal three: 7, 23, 56.",
                signature=SIG,
            ),
            SimpleNamespace(type="text", text="7 23 56"),
        ],
        stop_reason="end_turn",
        usage=None,
    )


def _stored_assistant_message(normalized) -> dict:
    """Rebuild the stored assistant message the way build_assistant_message does
    (content + reasoning_details from provider_data, no tool_calls for plain turns)."""
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
    }
    if tool_calls:
        msg["tool_calls"] = tool_calls
    blocks = provider_data.get("anthropic_content_blocks")
    if blocks:
        msg["anthropic_content_blocks"] = blocks
    return msg


class TestPlainTurnThinkingCapture:
    def test_normalize_captures_plain_thinking_into_reasoning_details(self):
        """normalize_response must preserve the plain thinking block (with
        signature) in provider_data.reasoning_details — this is the ONLY channel
        that can carry it across turns when there is no tool_use."""
        transport = get_transport("anthropic_messages")
        normalized = transport.normalize_response(_plain_turn_response())

        details = (normalized.provider_data or {}).get("reasoning_details")
        assert details, "plain-turn thinking block was not captured in reasoning_details"
        thinking = [b for b in details if isinstance(b, dict) and b.get("type") == "thinking"]
        assert thinking, f"no thinking block in reasoning_details: {details!r}"
        assert thinking[0].get("signature") == SIG

        # No tool_use in the turn -> no ordered-blocks channel is expected;
        # replay must therefore succeed via reasoning_details alone.
        assert not (normalized.provider_data or {}).get("anthropic_content_blocks")

    def test_plain_turn_thinking_replayed_on_next_turn(self):
        """Full chain: normalize -> stored message -> next-turn request.

        The converted request must carry the previous turn's thinking block
        (same signature, positioned before the text per the Messages contract).
        If reasoning_details is dropped anywhere along the chain, this test
        fails exactly the way the silent live failure looked."""
        transport = get_transport("anthropic_messages")
        normalized = transport.normalize_response(_plain_turn_response())
        assistant_msg = _stored_assistant_message(normalized)

        messages = [
            {"role": "user", "content": "想五个随机数字，告诉我其中三个。"},
            assistant_msg,
            {"role": "user", "content": "另外两个是什么？"},
        ]
        _system, anthropic_messages = convert_messages_to_anthropic(
            messages,
            base_url=None,
            model="claude-opus-4-8",
        )

        assistant_out = [m for m in anthropic_messages if m.get("role") == "assistant"]
        assert len(assistant_out) == 1
        content = assistant_out[0]["content"]
        assert isinstance(content, list), (
            "assistant turn was flattened to bare text — plain-turn thinking lost: "
            f"{content!r}"
        )
        types = [b.get("type") for b in content if isinstance(b, dict)]
        assert "thinking" in types, (
            "no thinking block replayed for the plain assistant turn; "
            f"converted content types: {types}"
        )
        thinking_block = next(b for b in content if isinstance(b, dict) and b.get("type") == "thinking")
        assert thinking_block.get("signature") == SIG, (
            "thinking block replayed without its original signature — Anthropic "
            "would reject tampered thinking with a 400"
        )
        # Protocol order: thinking precedes text.
        assert types.index("thinking") < types.index("text")

    def test_unsigned_thinking_block_survives_as_text_not_lost(self):
        """Unsigned thinking (e.g. Moonshot/Kimi empty signature) must NOT be
        silently dropped: the intended fallback on direct Anthropic is demotion
        to a text block so the reasoning content still reaches the next turn.
        This pins that contract — a regression to "content lost" fails here."""
        response = SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="reasoning without signature", signature=""),
                SimpleNamespace(type="text", text="answer"),
            ],
            stop_reason="end_turn",
            usage=None,
        )
        transport = get_transport("anthropic_messages")
        normalized = transport.normalize_response(response)
        assistant_msg = _stored_assistant_message(normalized)

        messages = [
            {"role": "user", "content": "q"},
            assistant_msg,
            {"role": "user", "content": "q2"},
        ]
        _system, anthropic_messages = convert_messages_to_anthropic(
            messages,
            base_url=None,
            model="claude-opus-4-8",
        )
        assistant_out = [m for m in anthropic_messages if m.get("role") == "assistant"]
        content = assistant_out[0]["content"]
        assert isinstance(content, list)
        types = [b.get("type") for b in content if isinstance(b, dict)]
        assert "thinking" not in types, (
            "unsigned thinking must not be replayed as a thinking block "
            f"(Anthropic cannot validate a missing signature): {types}"
        )
        texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        assert any("reasoning without signature" in t for t in texts), (
            "unsigned thinking was silently LOST instead of demoted to text; "
            f"text blocks on replay: {texts}"
        )
