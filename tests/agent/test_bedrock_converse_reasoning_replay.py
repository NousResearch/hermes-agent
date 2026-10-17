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


@pytest.mark.parametrize("model, expect", [
    (CLAUDE, {"text": "plan", "signature": "SIG"}),
    ("openai.gpt-oss-120b-1:0", {"text": "plan"}),
])
def test_converse_reasoning_survives_a_reload_from_state_db(model, expect):
    """``bedrock_content_blocks`` is live-only; the persisted ``reasoning_details`` must carry the turn."""
    from agent.bedrock_adapter import convert_messages_to_converse, normalize_converse_response

    live = normalize_converse_response({"output": {"message": {"content": [
        {"reasoningContent": {"reasoningText": {"text": "plan", "signature": "SIG"}}},
        {"toolUse": {"toolUseId": "t1", "name": "read_file", "input": {"path": "a"}}},
    ]}}, "stopReason": "tool_use", "usage": {}}).choices[0].message
    reloaded = {"role": "assistant", "content": "", "reasoning_content": live.reasoning_content,
                "reasoning_details": live.reasoning_details,
                "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]}
    history = [{"role": "user", "content": "Q1"}, reloaded, {"role": "tool", "tool_call_id": "t1", "content": "ok"},
               {"role": "assistant", "content": "A1"}, {"role": "user", "content": "Q2"}]
    _, msgs = convert_messages_to_converse(history, model=model)
    reasoning = [b["reasoningContent"]["reasoningText"] for b in msgs[1]["content"] if "reasoningContent" in b]
    assert reasoning == [expect]


GPT = "global.openai.gpt-6.1-sol"
OPUS = "global.anthropic.claude-opus-5-5"


def _bedrock_turn(model, tag, tool_id, *, reloaded):
    """One tool turn as the agent stores it: produced by ``model`` through the real transport (which stamps it)."""
    transport = BedrockTransport()
    transport.build_kwargs(model=model, messages=[{"role": "user", "content": "x"}])
    normalized = transport.normalize_response({"output": {"message": {"role": "assistant", "content": [
        {"reasoningContent": {"reasoningText": {"text": f"think {tag}"}}},
        {"reasoningContent": {"redactedContent": f"rsn_{tag}".encode()}},
        {"toolUse": {"toolUseId": tool_id, "name": "read_file", "input": {"path": "a"}}},
    ]}}, "stopReason": "tool_use", "usage": {}})
    data = dict(normalized.provider_data)
    if reloaded:  # bedrock_content_blocks is live-only; a state.db reload keeps reasoning_details
        data.pop("bedrock_content_blocks")
    call = {"id": tool_id, "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a"}'}}
    return {"role": "assistant", "content": "", "tool_calls": [call], "reasoning_content": f"think {tag}", **data}


def _session(issuer, *, reloaded):
    """A prior tool turn, a final answer, then an in-flight tool loop: every assistant turn minted by ``issuer``."""
    return [
        {"role": "user", "content": "Q1"}, _bedrock_turn(issuer, "prior", "call_1", reloaded=reloaded),
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"}, {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"}, _bedrock_turn(issuer, "loop", "call_2", reloaded=reloaded),
        {"role": "tool", "tool_call_id": "call_2", "content": "ok"},
    ]


def _replayed_reasoning(kwargs):
    return [[next(iter(b["reasoningContent"])) for b in m["content"] if "reasoningContent" in b]
            for m in kwargs["messages"] if m["role"] == "assistant" and any("toolUse" in b for b in m["content"])]


@pytest.mark.parametrize("reloaded", [False, True])
@pytest.mark.parametrize("issuer, target, expect", [
    # The issuing model gets its own sealed blobs back, prior turn and in-flight loop alike.
    (GPT, GPT, [["reasoningText", "redactedContent"]] * 2),
    # Fallback or switch to Claude, including mid-loop: Claude cannot read another family's reasoning at all.
    (GPT, OPUS, [[], []]),
    # Another GPT cannot open the blob either (it hangs on it): readable text only, in the loop too.
    (GPT, "global.openai.gpt-6-sol", [["reasoningText"]] * 2),
])
def test_reasoning_another_model_minted_never_replays_sealed(issuer, target, expect, reloaded):
    """Invariant: a sealed redactedContent blob reaches only the model that minted it. Coatue field report:
    a GPT-6.1 session fell back to Opus 5.5 after two read timeouts and every call 400'd on
    ``Invalid `data` in `redacted_thinking` block`` because GPT's blobs were replayed verbatim."""
    history = _session(issuer, reloaded=reloaded)
    before = copy.deepcopy(history)
    kwargs = BedrockTransport().build_kwargs(model=target, messages=history)
    assert _replayed_reasoning(kwargs) == expect
    toolus = [b["toolUse"]["toolUseId"] for m in kwargs["messages"] for b in m["content"] if "toolUse" in b]
    assert toolus == ["call_1", "call_2"]
    assert history == before


def test_claude_minted_reasoning_still_replays_to_claude():
    """Control: the gate is per issuer family, so Claude-to-Claude replay (signed thinking and redacted
    blobs) stays verbatim, as Bedrock requires for Claude."""
    history = _session("global.anthropic.claude-sonnet-5-5", reloaded=False)
    kwargs = BedrockTransport().build_kwargs(model=OPUS, messages=history)
    assert _replayed_reasoning(kwargs) == [["reasoningText", "redactedContent"]] * 2
