"""Kimi-family endpoints don't need thinking blocks stripped on replay; DeepSeek does."""

from types import SimpleNamespace

from agent.transports import get_transport
from agent.anthropic_adapter import convert_messages_to_anthropic

SIG = "sig-k3"

KIMI = "https://api.kimi.com/coding"
MOONSHOT = "https://api.moonshot.cn/anthropic"
DEEPSEEK = "https://api.deepseek.com/anthropic"


def _thinking_on_replay(base_url, signature=SIG, model="k3"):
    """Normalize a thinking+text turn, store it, convert to the next-turn request,
    and return its thinking blocks."""
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="five: 5 11 27 63 88", signature=signature),
            SimpleNamespace(type="text", text="5 27 88"),
        ],
        stop_reason="end_turn",
        usage=None,
    )
    normalized = get_transport("anthropic_messages").normalize_response(response)
    stored = {
        "role": "assistant",
        "content": normalized.content or "",
        "reasoning_details": (normalized.provider_data or {}).get("reasoning_details"),
    }
    messages = [
        {"role": "user", "content": "q1"},
        stored,
        {"role": "user", "content": "q2"},
    ]
    _sys, out = convert_messages_to_anthropic(messages, base_url=base_url, model=model)
    assistant = [m for m in out if m.get("role") == "assistant"][0]
    return [b for b in assistant["content"] if isinstance(b, dict) and b.get("type") == "thinking"]


def test_kimi_coding_keeps_signed_thinking():
    thinking = _thinking_on_replay(KIMI)
    assert thinking and thinking[0].get("signature") == SIG


def test_kimi_coding_keeps_unsigned_thinking():
    assert _thinking_on_replay(KIMI, signature="")


def test_moonshot_keeps_signed_thinking():
    thinking = _thinking_on_replay(MOONSHOT)
    assert thinking and thinking[0].get("signature") == SIG


def test_deepseek_still_strips_signed_thinking():
    # A DeepSeek model on the DeepSeek Anthropic endpoint must strip signed
    # thinking on replay. (The model must be a real DeepSeek slug: the bare
    # ``k3`` slug is now classified as Kimi family, and a Kimi-family MODEL
    # name deliberately preserves thinking regardless of gateway hostname —
    # the proxied-endpoint path, see _is_kimi_family_endpoint.)
    assert not _thinking_on_replay(DEEPSEEK, model="deepseek-reasoner")


def test_kimi_model_name_on_foreign_gateway_keeps_thinking():
    """A Kimi-family model slug replayed through a non-Kimi gateway hostname
    keeps its thinking blocks — upstream Kimi still enforces its replay
    semantics no matter what host fronts it (hermes-agent#13848, #17057).
    Covers both the named and bare Coding Plan slugs."""
    for model in ("kimi-k2.5", "k3"):
        assert _thinking_on_replay(DEEPSEEK, model=model), model


def test_direct_anthropic_keeps_signed_on_latest():
    assert _thinking_on_replay(None)


def test_orphan_tool_turn_demotes_and_leaks_no_internal_marker():
    """Signed thinking + parallel tool batch interrupted mid-flight (one orphan):
    the internal _thinking_signature_invalidated marker must be popped —
    never leak into the Kimi payload — while the thinking block itself
    replays as-is (Kimi does not enforce signatures)."""
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="plan both reads", signature=SIG),
            SimpleNamespace(type="tool_use", id="toolu_1", name="read_file", input={"path": "a.py"}),
            SimpleNamespace(type="tool_use", id="toolu_2", name="read_file", input={"path": "b.py"}),
        ],
        stop_reason="tool_use",
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
        {"role": "user", "content": "inspect both"},
        stored,
        {"role": "tool", "tool_call_id": "toolu_1", "content": "a.py: ok"},
        # toolu_2 interrupted: no tool result follows (orphan)
        {"role": "user", "content": "continue"},
    ]
    _sys, out = convert_messages_to_anthropic(messages, base_url=KIMI, model="k3")
    assistant = [m for m in out if m.get("role") == "assistant"][0]
    assert "_thinking_signature_invalidated" not in assistant, (
        f"internal marker leaked into Kimi payload: {assistant.keys()}"
    )
    types = [b.get("type") for b in assistant["content"] if isinstance(b, dict)]
    assert "thinking" in types, (
        "Kimi does not enforce signatures — even orphan-invalidated blocks "
        f"must replay as-is: {types}"
    )
