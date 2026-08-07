"""MiniMax /anthropic endpoints don't enforce thinking signatures — replay as-is.

Live-probed 2026-08-01 against api.minimax.io/anthropic: verbatim AND
content-mutated signed thinking blocks both replay with HTTP 200, so the
third-party strip-all path (pre-fix) silently dropped the prior turn's
chain-of-thought and broke interleaved reasoning on multi-turn work.
"""

from types import SimpleNamespace

from agent.transports import get_transport
from agent.anthropic_adapter import convert_messages_to_anthropic

SIG = "sig-minimax"

MINIMAX_IO = "https://api.minimax.io/anthropic"
MINIMAX_CN = "https://api.minimaxi.com/anthropic"


def _thinking_on_replay(base_url, with_tool_use=False):
    """Normalize a thinking(+tool_use) turn, store it, convert to the next-turn
    request, and return its thinking blocks."""
    content = [
        SimpleNamespace(type="thinking", thinking="five: 5 11 27 63 88", signature=SIG),
        SimpleNamespace(type="text", text="5 27 88"),
    ]
    if with_tool_use:
        content.append(SimpleNamespace(type="tool_use", id="toolu_1", name="read_file", input={"path": "a.py"}))
    response = SimpleNamespace(
        content=content,
        stop_reason="tool_use" if with_tool_use else "end_turn",
        usage=None,
    )
    normalized = get_transport("anthropic_messages").normalize_response(response)
    provider_data = normalized.provider_data or {}
    stored = {
        "role": "assistant",
        "content": normalized.content or "",
        "reasoning_details": provider_data.get("reasoning_details"),
        "tool_calls": [
            {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
            for tc in (normalized.tool_calls or [])
        ],
    }
    if provider_data.get("anthropic_content_blocks"):
        stored["anthropic_content_blocks"] = provider_data["anthropic_content_blocks"]
    messages = [
        {"role": "user", "content": "q1"},
        stored,
        {"role": "tool", "tool_call_id": "toolu_1", "content": "a.py: ok"} if with_tool_use else {"role": "user", "content": "q2"},
    ]
    if not with_tool_use:
        messages = messages[:2] + [{"role": "user", "content": "q2"}]
    _sys, out = convert_messages_to_anthropic(messages, base_url=base_url, model="MiniMax-M3")
    assistant = [m for m in out if m.get("role") == "assistant"][0]
    return [b for b in assistant["content"] if isinstance(b, dict) and b.get("type") == "thinking"]


def test_minimax_io_keeps_signed_thinking():
    thinking = _thinking_on_replay(MINIMAX_IO)
    assert thinking and thinking[0].get("signature") == SIG


def test_minimax_cn_keeps_signed_thinking():
    thinking = _thinking_on_replay(MINIMAX_CN)
    assert thinking and thinking[0].get("signature") == SIG


def test_minimax_interleaved_tool_turn_keeps_thinking():
    """Thinking + tool_use turn replayed into the next API call (the tool loop)
    must keep its thinking block — the exact multi-turn failure the strip caused."""
    thinking = _thinking_on_replay(MINIMAX_IO, with_tool_use=True)
    assert thinking and thinking[0].get("signature") == SIG
