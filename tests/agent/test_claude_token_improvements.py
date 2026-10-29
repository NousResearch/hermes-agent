"""Regression coverage for model-specific compression and signed thinking replay."""

from copy import deepcopy

from agent.context_compressor import ContextCompressor
from agent.anthropic_adapter import _manage_thinking_signatures


def test_model_absolute_cap_applies_and_does_not_leak_on_switch():
    compressor = ContextCompressor(
        model="claude-opus-5-5", threshold_percent=0.8,
        config_context_length=1_000_000, threshold_tokens_cap=220_000,
        model_token_caps={"claude": 150_000},
    )
    assert compressor.threshold_tokens == 150_000
    assert compressor.model_token_caps == {"claude": 150_000}
    assert ContextCompressor._coerce_model_token_caps({"claude": "oops", "": 8, "codex": -4, "bool": True}) == {}
    compressor.update_model("openai-codex/gpt-6-luna", 1_000_000)
    assert compressor.threshold_tokens == 220_000
    compressor.update_model("claude-sonnet", 1_000_000)
    assert compressor.threshold_tokens == 150_000


def test_native_signed_thinking_prefix_remains_append_only():
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "plan", "signature": "signed"},
            {"type": "tool_use", "id": "tool-1", "name": "x", "input": {}},
        ]},
        {"role": "user", "content": "continue"},
    ]
    before = deepcopy(history)
    replay = deepcopy(history) + [{"role": "assistant", "content": [
        {"type": "text", "text": "done"},
    ]}]
    _manage_thinking_signatures(replay, "https://api.anthropic.com", "claude-sonnet")
    assert replay[:len(before)] == before
    assert history == before


def test_third_party_still_strips_signed_thinking():
    messages = [{"role": "assistant", "content": [
        {"type": "thinking", "thinking": "private", "signature": "signed"},
        {"type": "text", "text": "ok"},
    ]}]
    _manage_thinking_signatures(messages, "https://api.deepseek.com/anthropic", "deepseek-chat")
    assert not any(b.get("type") == "thinking" for b in messages[0]["content"])
