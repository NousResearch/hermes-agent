"""Offline request-boundary checks for the Claude token optimization rollout."""
from copy import deepcopy

import pytest

from agent.anthropic_adapter import build_anthropic_kwargs, convert_messages_to_anthropic
from agent.context_compressor import ContextCompressor


def assistant(call_id, *, redacted=False):
    detail = ({"type": "redacted_thinking", "data": "fixture-opaque"}
              if redacted else {"type": "thinking", "thinking": "fixture plan", "signature": "fixture-signed"})
    return {"role": "assistant", "content": "", "reasoning_details": [detail],
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path":"fixture"}'}}]}


def result(call_id):
    return {"role": "tool", "tool_call_id": call_id, "content": "fixture result"}


@pytest.mark.parametrize("is_oauth", [False, True])
@pytest.mark.parametrize("base_url", ["https://api.anthropic.com", "https://inference-api.nousresearch.com"])
@pytest.mark.parametrize("redacted", [False, True])
def test_real_payload_builder_preserves_old_signed_prefix(is_oauth, base_url, redacted):
    history = [{"role": "user", "content": "fixture task"}, assistant("a", redacted=redacted), result("a")]
    original = deepcopy(history)
    appended = deepcopy(history) + [assistant("b"), result("b")]
    def build(messages):
        return build_anthropic_kwargs(model="claude-opus-5-5", messages=messages, tools=None,
                                      max_tokens=4096, reasoning_config={"enabled": True, "effort": "high"},
                                      is_oauth=is_oauth, base_url=base_url)
    first, second = build(history), build(appended)
    assert first["messages"] == second["messages"][:len(first["messages"])]
    signed = next(m for m in first["messages"] if m["role"] == "assistant")
    assert signed["content"][0]["type"] == ("redacted_thinking" if redacted else "thinking")
    assert history == original


@pytest.mark.parametrize("orphan", [False, True])
def test_compacted_tail_replay_preserves_only_intact_turn_signatures(orphan):
    # This fixture is the adapter boundary after compaction, not a live summarizer test.
    history = [{"role": "user", "content": "[compaction summary]"}, assistant("a")]
    if not orphan:
        history.append(result("a"))
    history += [{"role": "user", "content": "resume"}, assistant("b"), result("b")]
    original = deepcopy(history)
    _, converted = convert_messages_to_anthropic(history, base_url="https://api.anthropic.com", model="claude-opus-5-5")
    old = next(m for m in converted if m["role"] == "assistant")
    assert any(b.get("type") == "thinking" for b in old["content"]) is not orphan
    assert "_thinking_signature_invalidated" not in old
    assert history == original


def test_longest_model_cap_and_context_recalculation():
    c = ContextCompressor(model="claude-opus-5-5", config_context_length=1_000_000,
                          model_token_caps={"claude": 150_000, "claude-opus": 120_000})
    assert c.threshold_tokens == 120_000
    c.update_model("claude-opus-5-5", 200_000)
    assert c.threshold_tokens == 120_000
    c.update_model("gpt-6-luna", 1_000_000)
    baseline = ContextCompressor(model="gpt-6-luna", config_context_length=1_000_000)
    assert c.threshold_tokens == baseline.threshold_tokens
    c.update_model("claude-sonnet-4-6", 1_000_000)
    assert c.threshold_tokens == 150_000
