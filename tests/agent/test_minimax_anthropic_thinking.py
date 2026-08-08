"""MiniMax-M3 /anthropic thinking blocks must survive replay (downgraded to unsigned).

MiniMax-M3 on ``api.minimax.io/anthropic`` returns SIGNED thinking blocks.
Hermes previously routed MiniMax into the generic third-party branch of
``_manage_thinking_signatures``, which stripped ALL thinking blocks on
replay — the replayed assistant tool-call turn carried no chain-of-thought,
and interleaved reasoning died after the first tool-call turn
(hermes-agent#75725).

MiniMax accepts UNSIGNED thinking blocks on replay and continues the
chain-of-thought; a cross-context SIGNED block 400s. So the fix downgrades
signed blocks to unsigned (drop the signature, keep the text) instead of
stripping them.
"""

from types import SimpleNamespace

from agent.transports import get_transport
from agent.anthropic_adapter import convert_messages_to_anthropic

SIG = "sig-minimax"

MINIMAX = "https://api.minimax.io/anthropic"
MINIMAX_V1 = "https://api.minimax.io/anthropic/v1"
MINIMAX_CN = "https://api.minimaxi.com/anthropic"
MINIMAX_CN_V1 = "https://api.minimaxi.com/anthropic/v1"
DEEPSEEK = "https://api.deepseek.com/anthropic"
OPENAI_COMPAT = "https://api.minimax.io/v1"


def _thinking_on_replay(base_url, signature: str | None = SIG, model="MiniMax-M3"):
    """Normalize a thinking+tool_use turn, store it, convert to the next-turn
    request, and return its thinking + tool_use blocks."""
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="Let me think about this carefully", signature=signature),
            SimpleNamespace(
                type="tool_use",
                id="toolu_01",
                name="get_weather",
                input={"location": "San Francisco"},
            ),
        ],
        stop_reason="tool_use",
        usage=None,
    )
    normalized = get_transport("anthropic_messages").normalize_response(response)
    stored = {
        "role": "assistant",
        "content": normalized.content or "",
        "reasoning_details": (normalized.provider_data or {}).get("reasoning_details"),
        "anthropic_content_blocks": (normalized.provider_data or {}).get("anthropic_content_blocks"),
        "tool_calls": [
            {
                "id": "toolu_01",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "San Francisco"}',
                },
            }
        ],
    }
    messages = [
        {"role": "user", "content": "q1"},
        stored,
        {"role": "tool", "tool_call_id": "toolu_01", "content": "ok"},
    ]
    _sys, out = convert_messages_to_anthropic(messages, base_url=base_url, model=model)
    assistant = [m for m in out if m.get("role") == "assistant"][0]
    thinking = [b for b in assistant["content"] if isinstance(b, dict) and b.get("type") == "thinking"]
    tool_use = [b for b in assistant["content"] if isinstance(b, dict) and b.get("type") == "tool_use"]
    return thinking, tool_use


def test_minimax_downgrades_signed_thinking_to_unsigned():
    """Signed thinking blocks on a replayed MiniMax tool-call turn must survive
    as UNSIGNED blocks (text preserved, signature dropped)."""
    for base_url in (MINIMAX, MINIMAX_V1, MINIMAX_CN, MINIMAX_CN_V1):
        thinking, tool_use = _thinking_on_replay(base_url)
        assert thinking, f"{base_url}: thinking block must survive replay"
        assert thinking[0]["thinking"] == "Let me think about this carefully"
        assert "signature" not in thinking[0], (
            f"{base_url}: downgraded block must not carry the Anthropic signature"
        )
        assert len(tool_use) == 1, f"{base_url}: tool_use block must survive"


def test_minimax_keeps_unsigned_thinking_verbatim():
    """Already-unsigned thinking blocks pass through unchanged."""
    thinking, _ = _thinking_on_replay(MINIMAX, signature=None)
    assert thinking, "unsigned thinking block must survive replay"
    assert thinking[0]["thinking"] == "Let me think about this carefully"
    assert "signature" not in thinking[0]


def test_other_third_party_still_strips_signed():
    """Non-MiniMax third-party endpoints keep the strip-ALL behavior — the
    MiniMax branch must not change how other hosts are treated."""
    thinking, _ = _thinking_on_replay(DEEPSEEK)
    # DeepSeek keeps its own strip-signed/preserve-unsigned branch upstream;
    # with a signed block it strips. Assert we did not break that.
    assert not any(b.get("signature") for b in thinking)
