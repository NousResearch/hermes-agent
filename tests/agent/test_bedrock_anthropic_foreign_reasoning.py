"""Bedrock GPT -> Claude: another model's sealed reasoning never reaches Claude on the Anthropic Messages path.

A Bedrock session on a reasoning GPT (Converse) stores GPT's sealed reasoning as ``redacted_thinking`` in
``reasoning_details`` and stamps the turn with the issuing model (``CONVERSE_ISSUER_TYPE``). #142139 made
Converse drop a non-Claude turn's reasoning when Claude answers. Claude on Bedrock runs on the
AnthropicBedrock (``anthropic_messages``) path instead, which replayed the blob; Claude rejects it with
``Invalid `data` in `redacted_thinking` block``, and that 400 used to classify as ``format_error`` and fail
over to GPT on every turn.

Two layers are pinned here: the converter drops thinking minted by a non-Claude model (stamped turns), and
the classifier routes the rejection to the existing ``thinking_signature`` recovery (the safety net for
unstamped turns from older sessions), so the turn stays on Claude.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import httpx
import pytest

from agent.anthropic_message_convert import convert_messages_to_anthropic
from agent.bedrock_adapter import CONVERSE_ISSUER_TYPE
from agent.error_classifier import FailoverReason, classify_api_error

CLAUDE = "global.anthropic.claude-opus-5-5"
GPT = "us.openai.gpt-6.1-sol"
BEDROCK_URL = "https://bedrock-runtime.us-east-1.amazonaws.com"
GPT_BLOB = "R1BULXNlYWxlZC1yZWFzb25pbmctYmxvYg=="  # base64, as _persisted_reasoning_details stores it
REJECTION = "messages.3.content.10: Invalid `data` in `redacted_thinking` block"


def _gpt_turn(answer: str, *, stamped: bool = True, tool_call: dict | None = None) -> dict:
    details = [
        {"type": "thinking", "thinking": "gpt readable summary"},
        {"type": "redacted_thinking", "data": GPT_BLOB},
    ]
    if stamped:
        details.append({"type": CONVERSE_ISSUER_TYPE, "model": GPT})
    turn = {"role": "assistant", "content": answer, "reasoning_details": details}
    if tool_call:
        turn["tool_calls"] = [tool_call]
    return turn


def _claude_turn(answer: str, sig: str, *, issuer: str | None = None) -> dict:
    details = [{"type": "thinking", "thinking": "claude chain", "signature": sig}]
    if issuer:
        details.append({"type": CONVERSE_ISSUER_TYPE, "model": issuer})
    return {"role": "assistant", "content": answer, "reasoning_details": details}


def _wire_thinking(wire: list[dict]) -> list[dict]:
    return [b for m in wire if m["role"] == "assistant" and isinstance(m["content"], list)
            for b in m["content"] if b.get("type") in ("thinking", "redacted_thinking")]


def _convert(history: list[dict], model: str = CLAUDE, base_url: str = BEDROCK_URL) -> list[dict]:
    _, wire = convert_messages_to_anthropic(history, base_url=base_url, model=model)
    return wire


# --- Root: the converter never replays a non-Claude model's reasoning ---------------------------------------


@pytest.mark.parametrize("base_url", [BEDROCK_URL, None], ids=["bedrock", "anthropic-native"])
def test_gpt_stamped_reasoning_never_reaches_claude_and_claude_keeps_its_own(base_url):
    history = [
        {"role": "user", "content": "q1"}, _gpt_turn("a1"),
        {"role": "user", "content": "q2"}, _claude_turn("a2", "claude-sig-2"),
        {"role": "user", "content": "q3"},
    ]
    wire = _convert(history, base_url=base_url)

    assert GPT_BLOB not in json.dumps(wire)
    assert "gpt readable summary" not in json.dumps(wire)
    assert {"type": "thinking", "thinking": "claude chain", "signature": "claude-sig-2"} in _wire_thinking(wire)
    assert wire[1]["content"] == [{"type": "text", "text": "a1"}]


def test_fallback_mid_tool_loop_drops_gpt_reasoning_from_the_in_flight_loop():
    call = {"id": "call_1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}
    history = [
        {"role": "user", "content": "run it"}, _gpt_turn("", tool_call=call),
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]
    wire = _convert(history)

    assert GPT_BLOB not in json.dumps(wire)
    assert _wire_thinking(wire) == []
    assert [b["type"] for b in wire[1]["content"]] == ["tool_use"]


def test_claude_stamped_converse_turn_keeps_its_signed_thinking_and_redacted_blob():
    """A Claude turn captured on Converse (sticky stream-denied fallback) is still Claude's: it replays."""
    turn = _claude_turn("a1", "claude-sig-1", issuer="us.anthropic.claude-sonnet-4-6")
    turn["reasoning_details"].insert(1, {"type": "redacted_thinking", "data": "Q2xhdWRlLXJlZGFjdGVk"})
    wire = _convert([{"role": "user", "content": "q1"}, turn, {"role": "user", "content": "q2"}])

    assert _wire_thinking(wire) == [
        {"type": "thinking", "thinking": "claude chain", "signature": "claude-sig-1"},
        {"type": "redacted_thinking", "data": "Q2xhdWRlLXJlZGFjdGVk"},
    ]


def test_unstamped_redacted_thinking_still_replays():
    """Unstamped turns keep replaying: Claude's own AnthropicBedrock turns carry no issuer record and Claude
    mints redacted_thinking itself, so "unstamped" cannot mean "foreign". A pre-stamp GPT blob is healed by
    the one-shot thinking_signature recovery instead (see the conversation-loop test below)."""
    wire = _convert([{"role": "user", "content": "q1"}, _gpt_turn("a1", stamped=False), {"role": "user", "content": "q2"}])

    assert {"type": "redacted_thinking", "data": GPT_BLOB} in _wire_thinking(wire)


PROFILE_ARN = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123"


def _converse_assistant_reasoning(wire: list[dict]) -> list[dict]:
    return [b for m in wire if m["role"] == "assistant" for b in m["content"] if "reasoningContent" in b]


@pytest.mark.parametrize("wire", ["anthropic", "converse"])
def test_unresolved_profile_arn_stamp_keeps_claudes_own_thinking(monkeypatch, wire):
    """A Claude configured by application-profile ARN stamps its Converse turns with the ARN. When
    GetInferenceProfile is not granted the ARN stays unresolved: that issuer is unknown, not foreign, so a
    switch to a plain Claude id keeps the turn's signed thinking on both Bedrock wires."""
    import agent.bedrock_adapter as bedrock
    from agent.bedrock_adapter import convert_messages_to_converse

    monkeypatch.setitem(bedrock._inference_profile_model_cache, PROFILE_ARN, PROFILE_ARN)
    history = [{"role": "user", "content": "q1"}, _claude_turn("a1", "claude-sig-1", issuer=PROFILE_ARN),
               {"role": "user", "content": "q2"}]
    if wire == "anthropic":
        assert {"type": "thinking", "thinking": "claude chain", "signature": "claude-sig-1"} in _wire_thinking(_convert(history))
    else:
        _system, msgs = convert_messages_to_converse(history, model=CLAUDE)
        assert _converse_assistant_reasoning(msgs)


@pytest.mark.parametrize("wire", ["anthropic", "converse"])
def test_profile_arn_resolving_to_gpt_still_drops_its_reasoning(monkeypatch, wire):
    import agent.bedrock_adapter as bedrock
    from agent.bedrock_adapter import convert_messages_to_converse

    monkeypatch.setitem(bedrock._inference_profile_model_cache, PROFILE_ARN,
                        "arn:aws:bedrock:us-east-1::foundation-model/openai.gpt-6.1-sol")
    turn = _gpt_turn("a1", stamped=False)
    turn["reasoning_details"].append({"type": CONVERSE_ISSUER_TYPE, "model": PROFILE_ARN})
    history = [{"role": "user", "content": "q1"}, turn, {"role": "user", "content": "q2"}]
    if wire == "anthropic":
        assert GPT_BLOB not in json.dumps(_convert(history))
    else:
        _system, msgs = convert_messages_to_converse(history, model=CLAUDE)
        assert _converse_assistant_reasoning(msgs) == []


# --- Safety net: the rejection reaches the thinking_signature recovery ---------------------------------------


def _anthropic_400(message: str, body: dict | None = None):
    import anthropic

    body = body if body is not None else {"message": message}
    response = httpx.Response(400, request=httpx.Request("POST", f"{BEDROCK_URL}/model/{CLAUDE}/invoke"))
    return anthropic.BadRequestError(f"Error code: 400 - {body}", response=response, body=body)


@pytest.mark.parametrize("body", [
    None,  # Bedrock's flat {"message": ...}
    {"type": "error", "error": {"type": "invalid_request_error", "message": REJECTION}},  # direct API
], ids=["bedrock-body", "anthropic-body"])
def test_invalid_redacted_thinking_data_classifies_as_thinking_signature(body):
    result = classify_api_error(_anthropic_400(REJECTION, body), provider="bedrock", model=CLAUDE)

    assert result.reason == FailoverReason.thinking_signature
    assert result.retryable is True


@pytest.mark.parametrize(("message", "expected"), [
    # Existing Anthropic signature wording keeps its class.
    ("messages.1.content.0: Invalid `signature` in `thinking` block", FailoverReason.thinking_signature),
    # OpenAI Responses stale encrypted reasoning: still beats the Anthropic rule (#70595).
    ("thinking_signature_invalid: redacted_thinking invalid", FailoverReason.invalid_encrypted_content),
    # Other invalid-block 400s stay format errors.
    ("messages.1.content.0: Invalid `data` in `image` block", FailoverReason.format_error),
])
def test_neighbouring_400s_keep_their_class(message, expected):
    assert classify_api_error(_anthropic_400(message), provider="bedrock", model=CLAUDE).reason == expected


# --- Composed: real AIAgent turn loop, AnthropicBedrock client faked -----------------------------------------


class _FakeMessages:
    """Claude on Bedrock: rejects any request replaying GPT's sealed blob, else answers with signed thinking."""

    def __init__(self, log):
        self.log = log
        self.always_reject = False

    def create(self, **kwargs):
        sent_blob = GPT_BLOB in json.dumps(kwargs.get("messages"), default=str)
        signatures = [b.get("signature") for b in _wire_thinking(kwargs.get("messages") or []) if b.get("signature")]
        self.log.append({"model": kwargs.get("model"), "sent_gpt_blob": sent_blob, "claude_signatures": signatures})
        if sent_blob or self.always_reject:
            raise _anthropic_400(REJECTION)
        n = len(self.log)
        return SimpleNamespace(
            content=[SimpleNamespace(type="thinking", thinking=f"claude chain {n}", signature=f"claude-sig-{n}"),
                     SimpleNamespace(type="text", text=f"claude answer {n}")],
            stop_reason="end_turn", model=CLAUDE,
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )


class _FakeConverse:
    """Converse client: a GPT answer here means the turn failed over (setup probes also land here)."""

    def __init__(self, log):
        self.log = log

    def converse(self, **kwargs):
        self.log.append(kwargs.get("modelId"))
        return {"output": {"message": {"role": "assistant", "content": [{"text": "gpt answer"}]}},
                "stopReason": "end_turn", "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}}

    def converse_stream(self, **kwargs):
        raise AssertionError("streaming is disabled in this test")


@pytest.fixture()
def bedrock_claude_agent(monkeypatch):
    import run_agent

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIATESTTESTTESTTEST")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **kwargs: [{
        "type": "function",
        "function": {"name": "t", "description": "t", "parameters": {"type": "object", "properties": {}}},
    }])
    monkeypatch.setattr("model_tools.check_toolset_requirements", dict)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    log: list[dict] = []
    converse_models: list[str] = []
    client = SimpleNamespace(messages=_FakeMessages(log), close=lambda: None)
    monkeypatch.setattr("agent.anthropic_adapter.build_anthropic_bedrock_client", lambda *a, **k: client)
    monkeypatch.setattr("agent.bedrock_adapter._get_bedrock_runtime_client",
                        lambda *a, **k: _FakeConverse(converse_models))

    agent = run_agent.AIAgent(
        model=CLAUDE, provider="bedrock", base_url=BEDROCK_URL, api_mode="anthropic_messages",
        fallback_model=[{"provider": "bedrock", "model": GPT}],
        quiet_mode=True, skip_context_files=True, skip_memory=True, max_iterations=4,
    )
    agent._disable_streaming = True
    agent._persist_session = lambda *a, **k: None
    agent._save_trajectory = lambda *a, **k: None
    return agent, log, converse_models


def _gpt_history(*, stamped: bool) -> list[dict]:
    return [
        {"role": "user", "content": "q1"}, _gpt_turn("a1", stamped=stamped),
        {"role": "user", "content": "q2"}, _gpt_turn("a2", stamped=stamped),
    ]


def test_switch_to_claude_never_sends_stamped_gpt_reasoning(bedrock_claude_agent, caplog):
    agent, log, converse_models = bedrock_claude_agent
    with caplog.at_level(logging.INFO):
        result = agent.run_conversation("q3", conversation_history=_gpt_history(stamped=True))

    assert result["final_response"] == "claude answer 1"
    assert log == [{"model": CLAUDE, "sent_gpt_blob": False, "claude_signatures": []}]
    assert GPT not in converse_models
    assert "Fallback activated" not in caplog.text


def test_rejected_unstamped_gpt_blob_is_stripped_and_the_turn_stays_on_claude(bedrock_claude_agent, caplog):
    agent, log, converse_models = bedrock_claude_agent
    with caplog.at_level(logging.INFO):
        first = agent.run_conversation("q3", conversation_history=_gpt_history(stamped=False))
        second = agent.run_conversation("q4", conversation_history=first["messages"])

    assert "Fallback activated" not in caplog.text
    assert GPT not in converse_models
    assert {entry["model"] for entry in log} == {CLAUDE}
    # One rejection, one stripped retry on Claude; the next turn keeps the blob out and replays
    # Claude's own signed thinking from the turn that succeeded.
    assert [entry["sent_gpt_blob"] for entry in log] == [True, False, False]
    assert first["final_response"] == "claude answer 2"
    assert second["final_response"] == "claude answer 3"
    assert "claude-sig-2" in log[2]["claude_signatures"]


def test_a_rejection_the_strip_cannot_heal_stays_bounded(bedrock_claude_agent, caplog):
    """The strip runs once per turn (thinking_sig_retry_attempted); a repeat rejection takes the
    ordinary retryable budget and then the fallback chain, never an endless strip-and-resend."""
    agent, log, converse_models = bedrock_claude_agent
    agent._anthropic_client.messages.always_reject = True
    with caplog.at_level(logging.INFO):
        result = agent.run_conversation("q3", conversation_history=_gpt_history(stamped=False))

    assert caplog.text.count("Thinking block signature recovery") == 1
    assert 2 <= len(log) <= 1 + agent._api_max_retries
    assert GPT in converse_models and result["final_response"] == "gpt answer"
