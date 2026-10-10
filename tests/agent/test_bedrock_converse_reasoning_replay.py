"""Claude reasons on Bedrock Converse, and each Converse model gets back only the reasoning it can take.

Converse is the bearer-token Claude path and the sticky fallback after a stream-denied AnthropicBedrock
error, so a Claude turn captured in Anthropic shape (``anthropic_content_blocks``) must keep its signed
thinking ahead of its toolUse. AWS docs: Claude needs the signature and all previous messages; DeepSeek-R1's
sample removes prior reasoning; Kimi K3 raises InternalServerException on prior-turn reasoning; non-Claude
models reject the ``reasoningText.signature`` field.
"""

from __future__ import annotations

import copy

import pytest

from agent.transports.bedrock import BedrockTransport

CLAUDE = "us.anthropic.claude-opus-4-8-v1:0"


def _anthropic_bedrock_turn(question, sig, tool_id):
    """A tool turn stored by the AnthropicBedrock path, before the session fell back to Converse."""
    signed = {"type": "thinking", "thinking": f"plan {sig}", "signature": sig}
    call = {"id": tool_id, "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a"}'}}
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": "", "tool_calls": [call], "reasoning_details": [signed],
         "anthropic_content_blocks": [signed, {"type": "tool_use", "id": tool_id, "name": "read_file", "input": {"path": "RAW"}}]},
        {"role": "tool", "tool_call_id": tool_id, "content": "ok"},
    ]


def _converse_turn(question, sig, tool_id):
    """A tool turn captured by the Converse normalizer for a non-Claude model (ordered sidecar)."""
    call = {"id": tool_id, "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a"}'}}
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": "", "tool_calls": [call], "reasoning_content": f"own {sig}",
         "bedrock_content_blocks": [{"reasoningContent": {"text": f"own {sig}"}},
                                    {"toolUse": {"toolUseId": tool_id, "name": "read_file", "input": {"path": "a"}}}]},
        {"role": "tool", "tool_call_id": tool_id, "content": "ok"},
    ]


def _converse_history():
    turn = _converse_turn("Q1", "sig_prior", "t1")
    turn[1]["bedrock_content_blocks"][0]["reasoningContent"]["signature"] = "foreign"  # e.g. minted by Claude
    return turn + [{"role": "assistant", "content": "A1"}] + _converse_turn("Q2", "sig_loop", "t2")


def _history():
    return _anthropic_bedrock_turn("Q1", "sig_prior", "t1") + [{"role": "assistant", "content": "A1"}] + _anthropic_bedrock_turn("Q2", "sig_loop", "t2")


def _reasoning(blocks):
    return [b["reasoningContent"]["reasoningText"] for b in blocks if "reasoningContent" in b]


@pytest.mark.parametrize(
    "model, prior, loop",
    [
        (CLAUDE, [{"text": "plan sig_prior", "signature": "sig_prior"}], [{"text": "plan sig_loop", "signature": "sig_loop"}]),
        # The model's own Converse capture: prior turns follow its contract, the in-flight loop replays as captured.
        ("us.deepseek.r1-v1:0", [], [{"text": "own sig_loop"}]),
        ("global.moonshotai.kimi-k3", [], [{"text": "own sig_loop"}]),
        # gpt-oss rejects reasoningText.signature: prior turns go back as readable text only.
        ("openai.gpt-oss-120b-1:0", [{"text": "own sig_prior"}], [{"text": "own sig_loop"}]),
    ],
)
def test_converse_replays_each_models_reasoning_contract(model, prior, loop):
    """Invariant: the in-flight tool loop replays its reasoning ahead of its toolUse with redacted tool
    input; earlier turns follow the model's documented contract; a Claude signature never reaches a
    non-Claude model; history is untouched."""
    history = _history() if "claude" in model else _converse_history()
    before = copy.deepcopy(history)
    kwargs = BedrockTransport().build_kwargs(model=model, messages=history, reasoning_config={"enabled": True, "effort": "high"})

    assistants = [m["content"] for m in kwargs["messages"] if m["role"] == "assistant"]
    assert _reasoning(assistants[0]) == prior
    assert _reasoning(assistants[-1]) == loop
    assert [next(iter(b)) for b in assistants[-1] if "cachePoint" not in b] == ["reasoningContent", "toolUse"]
    assert assistants[-1][1]["toolUse"]["input"] == {"path": "a"}
    assert history == before
    if "claude" not in model:  # Claude-shaped carriers from before a /model switch stay Claude's
        switched = BedrockTransport().build_kwargs(model=model, messages=_history())
        assert "signature" not in repr(switched["messages"])


def test_claude_on_converse_requests_thinking():
    """Invariant: a reasoning config reaches Claude and Nova 2 as additionalModelRequestFields; models
    without a documented switch get none."""
    msgs = [{"role": "user", "content": "hi"}]
    cfg = {"enabled": True, "effort": "high"}
    transport = BedrockTransport()

    adaptive = transport.build_kwargs(model=CLAUDE, messages=msgs, reasoning_config=cfg)
    assert adaptive["additionalModelRequestFields"] == {
        "thinking": {"type": "adaptive", "display": "summarized"}, "output_config": {"effort": "high"}}
    manual = transport.build_kwargs(model="anthropic.claude-sonnet-4-5-20250929-v1:0", messages=msgs,
                                    max_tokens=4096, reasoning_config=cfg)
    assert manual["additionalModelRequestFields"]["thinking"] == {"type": "enabled", "budget_tokens": 16000}
    assert manual["inferenceConfig"]["maxTokens"] > 16000 and manual["inferenceConfig"]["temperature"] == 1
    nova = transport.build_kwargs(model="us.amazon.nova-2-lite-v1:0", messages=msgs, max_tokens=4096, reasoning_config=cfg)
    assert nova["additionalModelRequestFields"] == {"reasoningConfig": {"type": "enabled", "maxReasoningEffort": "high"}}
    assert "maxTokens" not in nova.get("inferenceConfig", {})  # Nova: maxTokens must be unset at high
    assert "additionalModelRequestFields" not in transport.build_kwargs(
        model="meta.llama3-70b-instruct-v1:0", messages=msgs, reasoning_config=cfg)
    assert "additionalModelRequestFields" not in transport.build_kwargs(model=CLAUDE, messages=msgs)
