"""Regression test for #123923: Anthropic 400 'thinking blocks cannot be modified' on
heartbeat resume of an interleaved-thinking session.

Shape under test
----------------
A live interleaved assistant turn replays verbatim through the in-memory
``anthropic_content_blocks`` channel. That channel is NOT persisted (schema migration 09
dropped the column), so a fresh process — every Paperclip heartbeat spawns one — restores
the turn from state.db as parallel ``reasoning_details`` + ``tool_calls`` fields, and
``_convert_assistant_message`` rebuilds it as [all thinking][all tool_use]. The second
thinking block was signed against a prefix containing tool_use_1, so the rebuilt layout can
never validate and Anthropic 400s on the first request after every resume.

Contract: a restored turn whose reasoning_details carry >=2 signed thinking blocks plus
tool_calls (the interleaved fingerprint) must not replay signed blocks in the reconstructed
order — they are demoted to text on the latest assistant turn (existing dead-signature
policy) while the verbatim channel keeps working for in-memory turns. A single-signed-block
turn hoists onto an identical prefix and must KEEP replaying signed.
"""

from types import SimpleNamespace

from agent.anthropic_message_convert import convert_messages_to_anthropic
from agent.transports import get_transport


def _interleaved_response() -> SimpleNamespace:
    """Two signed thinking blocks interleaved with two tool_use blocks (#123923's dump)."""
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="Plan: inspect file A first.", signature="sig-A" * 80),
            SimpleNamespace(type="tool_use", id="toolu_1", name="read_file", input={"path": "a.py"}),
            SimpleNamespace(type="thinking", thinking="A looked fine; now inspect B.", signature="sig-B" * 80),
            SimpleNamespace(type="tool_use", id="toolu_2", name="read_file", input={"path": "b.py"}),
        ],
        stop_reason="tool_use",
        usage=None,
    )


def _stored_message(normalized) -> dict:
    """OpenAI-style assistant message as persisted: NO anthropic_content_blocks (in-memory only),
    parallel reasoning_details + tool_calls — the state.db restore shape."""
    provider_data = normalized.provider_data or {}
    assert provider_data.get("anthropic_content_blocks"), "fixture expects interleaved response"
    return {
        "role": "assistant",
        "content": normalized.content or "",
        "reasoning_details": provider_data.get("reasoning_details"),
        "tool_calls": [
            {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
            for tc in (normalized.tool_calls or [])
        ],
    }


def _convert(messages: list) -> list:
    _system, converted = convert_messages_to_anthropic(messages, base_url=None, model="claude-opus-5-5")
    return converted


def _last_assistant(converted: list) -> dict:
    return [m for m in converted if m.get("role") == "assistant"][-1]


class TestInterleavedResumeSignatures:
    def test_restored_interleaved_turn_replays_no_signed_blocks(self):
        """The restored interleaved turn must not put signed thinking blocks on the wire in the
        reconstructed (hoisted) order — that layout 400s on every resume (#123923)."""
        normalized = get_transport("anthropic_messages").normalize_response(_interleaved_response())
        restored = _stored_message(normalized)

        converted = _convert([
            {"role": "user", "content": "Inspect a.py and b.py."},
            restored,
            {"role": "tool", "tool_call_id": "toolu_1", "content": "a.py: ok"},
            {"role": "tool", "tool_call_id": "toolu_2", "content": "b.py: ok"},
        ])

        content = _last_assistant(converted)["content"]
        signed = [
            b for b in content
            if isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking")
            and (b.get("signature") or b.get("data"))
        ]
        assert not signed, (
            "Restored interleaved turn replayed signed thinking blocks in the hoisted "
            f"[all thinking][all tool_use] order; the 2nd+ signatures are bound to the "
            f"original interleaved positions and Anthropic 400s. content: {content}"
        )

    def test_restored_interleaved_turn_keeps_reasoning_text_and_tool_pairs(self):
        """Demotion must not lose the turn: reasoning prose survives as text, both tool_use
        blocks stay for their tool_result followers."""
        normalized = get_transport("anthropic_messages").normalize_response(_interleaved_response())
        restored = _stored_message(normalized)

        converted = _convert([
            {"role": "user", "content": "Inspect a.py and b.py."},
            restored,
            {"role": "tool", "tool_call_id": "toolu_1", "content": "a.py: ok"},
            {"role": "tool", "tool_call_id": "toolu_2", "content": "b.py: ok"},
        ])

        content = _last_assistant(converted)["content"]
        text = " ".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
        assert "inspect file A" in text and "inspect B" in text
        tool_ids = {b.get("id") for b in content if isinstance(b, dict) and b.get("type") == "tool_use"}
        assert tool_ids == {"toolu_1", "toolu_2"}

    def test_single_signed_thinking_turn_still_replays_signed(self):
        """One signed thinking block hoists onto an IDENTICAL prefix (it preceded everything);
        its signature stays valid, so the verbatim replay path must keep it."""
        response = SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="Plan: inspect A.", signature="sig-A" * 80),
                SimpleNamespace(type="tool_use", id="toolu_1", name="read_file", input={"path": "a.py"}),
            ],
            stop_reason="tool_use",
            usage=None,
        )
        normalized = get_transport("anthropic_messages").normalize_response(response)
        provider_data = normalized.provider_data or {}
        assert not provider_data.get("anthropic_content_blocks") or True  # channel shape irrelevant here
        restored = {
            "role": "assistant",
            "content": normalized.content or "",
            "reasoning_details": provider_data.get("reasoning_details"),
            "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                for tc in (normalized.tool_calls or [])
            ],
        }

        converted = _convert([
            {"role": "user", "content": "Inspect a.py."},
            restored,
            {"role": "tool", "tool_call_id": "toolu_1", "content": "a.py: ok"},
        ])

        content = _last_assistant(converted)["content"]
        signed = [
            b for b in content
            if isinstance(b, dict) and b.get("type") == "thinking" and b.get("signature")
        ]
        assert signed, "single-block turn hoists to an identical prefix; signed replay must survive"

    def test_in_memory_ordered_channel_unaffected(self):
        """Live sessions replay through anthropic_content_blocks verbatim and must not be demoted."""
        normalized = get_transport("anthropic_messages").normalize_response(_interleaved_response())
        provider_data = normalized.provider_data or {}
        restored = _stored_message(normalized)
        restored["anthropic_content_blocks"] = provider_data.get("anthropic_content_blocks")

        converted = _convert([
            {"role": "user", "content": "Inspect a.py and b.py."},
            restored,
            {"role": "tool", "tool_call_id": "toolu_1", "content": "a.py: ok"},
            {"role": "tool", "tool_call_id": "toolu_2", "content": "b.py: ok"},
        ])

        content = _last_assistant(converted)["content"]
        signed = [b for b in content if isinstance(b, dict) and b.get("type") == "thinking" and b.get("signature")]
        assert len(signed) == 2, "verbatim ordered replay must keep both signed blocks"
