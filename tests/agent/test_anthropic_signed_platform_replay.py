"""Signed thinking reaches the wire wherever the upstream can verify it, and only there.

Anthropic documents ``signature`` values as valid across the Claude API, Bedrock and Google Cloud;
Foundry serves the same Claude; MiniMax's Anthropic API asks for its own thinking blocks back
unchanged. Unknown relays cannot verify them and keep stripping; DeepSeek and Kimi keep their own
contracts. Within the in-flight tool loop every model gets its thinking back (required by the API);
earlier turns only where the model keeps prior thinking.
"""

from __future__ import annotations

import copy

import pytest

from agent.anthropic_thinking_policy import native_anthropic_preserves_prior_thinking
from agent.anthropic_thinking_replay import tracks_rejected_thinking
from agent.message_sanitization import stale_thinking_reaches_wire
from agent.transports.anthropic import AnthropicTransport

BEDROCK = "https://bedrock-runtime.us-east-1.amazonaws.com"
VERTEX = "https://us-east5-aiplatform.googleapis.com/v1/projects/p/locations/us-east5/publishers/anthropic/models"
FOUNDRY = "https://my-res.services.ai.azure.com/anthropic"
MINIMAX = "https://api.minimax.io/anthropic"
MINIMAX_CN = "https://api.minimaxi.com/anthropic"
RELAY = "https://relay.example.com/anthropic"
ALL, LOOP, NONE = ["sig_prior", "sig_loop1", "sig_loop2"], ["sig_loop1", "sig_loop2"], []

_CASES = {
    # Claude platforms: preserving models replay every turn, last-turn-only models the whole tool loop.
    "bedrock_opus": (BEDROCK, "us.anthropic.claude-opus-4-8-v1:0", ALL, True),
    "bedrock_sonnet45": (BEDROCK, "us.anthropic.claude-sonnet-4-5-20250929-v1:0", LOOP, True),
    "bedrock_fips": ("https://bedrock-runtime-fips.us-gov-west-1.amazonaws.com", "anthropic.claude-opus-4-8", ALL, True),
    "vertex": (VERTEX, "claude-opus-4-8", ALL, True),
    "foundry": (FOUNDRY, "claude-opus-4-8", ALL, True),
    "minimax": (MINIMAX, "MiniMax-M2.7", ALL, True),
    "minimax_cn": (MINIMAX_CN, "MiniMax-M3", ALL, True),
    # Direct API: Haiku 5.5 keeps all turns; older generations keep the current tool loop.
    "native_haiku55": (None, "claude-haiku-5-5", ALL, True),
    "native_haiku45": (None, "claude-haiku-4-5", LOOP, True),
    "native_sonnet45": (None, "claude-sonnet-4-5", LOOP, True),
    "native_opus48": (None, "claude-opus-4-8", ALL, True),
    # Unchanged contracts.
    "unknown_relay": (RELAY, "claude-opus-4-8", NONE, False),
    "bedrock_lookalike_host": ("https://bedrock-runtime.us-east-1.amazonaws.com.evil.test", "claude-opus-4-8", NONE, False),
    "bedrock_lookalike_path": ("https://evil.test/bedrock-runtime.us-east-1.amazonaws.com", "claude-opus-4-8", NONE, False),
    "deepseek": ("https://api.deepseek.com/anthropic", "deepseek-v4-pro", NONE, False),
    "deepseek_on_bedrock_host": (BEDROCK, "deepseek-v4-pro", NONE, False),
    "kimi": ("https://api.kimi.com/coding", "kimi-k2.5", ALL, False),
}


def _signed(sig):
    return {"type": "thinking", "thinking": f"thought {sig}", "signature": sig}


def _loop_step(n):
    call = {"id": f"t{n}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
    return [
        {"role": "assistant", "content": "", "tool_calls": [call], "reasoning_details": [_signed(f"sig_loop{n}")],
         "anthropic_content_blocks": [_signed(f"sig_loop{n}"), {"type": "tool_use", "id": f"t{n}", "name": "read_file", "input": {}}]},
        {"role": "tool", "tool_call_id": f"t{n}", "content": "ok"},
    ]


def _history():
    return [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1", "reasoning_details": [_signed("sig_prior")]},
        {"role": "user", "content": "Q2"},
        *_loop_step(1),
        *_loop_step(2),
    ]


@pytest.mark.parametrize("case", list(_CASES))
def test_signed_thinking_replays_exactly_where_the_route_can_verify_it(case):
    """Invariant: the transport entry point sends the route's signed blocks in their captured order,
    the signature-400 self-heal covers every signed route, accounting charges prior-turn thinking
    exactly when it is on the wire, and stored history is never mutated."""
    base_url, model, expected, signed_route = _CASES[case]
    history = _history()
    before = copy.deepcopy(history)

    _system, wire = AnthropicTransport().convert_messages(history, base_url=base_url, model=model)

    blocks = [b for m in wire if m["role"] == "assistant" for b in m["content"]]
    assert [b.get("signature") for b in blocks if b.get("type") == "thinking"] == expected
    for step in (1, 2):  # interleaved order: each step's thinking precedes its tool_use
        types = [b["type"] for b in blocks if b.get("signature") == f"sig_loop{step}" or b.get("id") == f"t{step}"]
        assert types == (["thinking", "tool_use"] if f"sig_loop{step}" in expected else ["tool_use"])
    agent = type("A", (), {"api_mode": "anthropic_messages", "base_url": base_url, "model": model})()
    assert tracks_rejected_thinking(agent) is signed_route
    # Kimi replays as-is through its own reasoning_content echo; signed routes are priced by the carriers.
    assert native_anthropic_preserves_prior_thinking(base_url, model) is (signed_route and "sig_prior" in expected)
    if signed_route:
        assert stale_thinking_reaches_wire("anthropic_messages", "custom", model, base_url) is ("sig_prior" in expected)
    assert history == before


def test_a_steer_merged_into_the_tool_result_turn_keeps_the_loop_thinking():
    """/steer text lands in the tool_result user turn; it must not end the in-flight loop for
    last-turn-only models (main kept the newest assistant's thinking there)."""
    history = _history()
    history.append({"role": "user", "content": "steer: also check b"})
    _system, wire = AnthropicTransport().convert_messages(history, base_url=None, model="claude-sonnet-4-5")
    sigs = [b.get("signature") for m in wire if m["role"] == "assistant" for b in m["content"] if b.get("type") == "thinking"]
    assert sigs == LOOP
