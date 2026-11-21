"""Signature-400 recovery on Claude over Bedrock Converse.

Converse replays an assistant turn from its ordered ``bedrock_content_blocks`` sidecar ahead of
``reasoning_details``. A repair that only pops ``reasoning_details`` resends the rejected signed block on
every retry (``bound to a different conversation`` after compression rebuilt the tool list). The
rejection must suppress the block on every carrier, persist, and survive a rebuild from canonical history.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

from agent.anthropic_thinking_replay import apply_rejected_thinking_suppression, tracks_rejected_thinking
from agent.bedrock_adapter import convert_messages_to_converse
from agent.error_classifier import FailoverReason
from agent.turn_recovery import _recover_format_errors
from agent.turn_retry_state import TurnRetryState

CLAUDE = "global.anthropic.claude-opus-5-5"
BASE = "https://bedrock-runtime.eu-central-1.amazonaws.com"


class _ModelConfigDB:
    def __init__(self):
        self.cfg = {}

    def get_session_model_config_value(self, session_id, key, default):
        return self.cfg.get((session_id, key), default)

    def patch_session_model_config(self, session_id, patch):
        for key, value in patch.items():
            self.cfg[(session_id, key)] = value


def _agent(model: str | None = CLAUDE, db=None, session_id="s1"):
    agent = SimpleNamespace(
        api_mode="bedrock_converse", model=model, base_url=BASE, provider="bedrock",
        session_id=session_id, _session_db=db or _ModelConfigDB(), _persist_disabled=False,
        log_prefix="", _usage_anchor=None, context_compressor=None,
    )
    agent._vprint = lambda *args, **kwargs: None
    return agent


def _converse_turn(question, answer, sig, tool_id):
    """A live Converse turn: persisted ``reasoning_details`` plus the ordered sidecar the adapter replays first."""
    return [
        {"role": "user", "content": question},
        {
            "role": "assistant", "content": answer,
            "tool_calls": [{"id": tool_id, "type": "function", "function": {"name": "read_file", "arguments": "{}"}}],
            "reasoning_details": [{"type": "thinking", "thinking": f"why {sig}", "signature": sig}],
            "bedrock_content_blocks": [
                {"reasoningContent": {"text": f"why {sig}", "signature": sig}},
                {"text": answer},
                {"toolUse": {"toolUseId": tool_id, "name": "read_file", "input": {}}},
            ],
        },
        {"role": "tool", "tool_call_id": tool_id, "content": "ok"},
    ]


def _signatures(messages, model=CLAUDE):
    _system, wire = convert_messages_to_converse(messages, model=model)
    return [
        block["reasoningContent"]["reasoningText"].get("signature")
        for message in wire if message["role"] == "assistant"
        for block in message["content"]
        if isinstance(block.get("reasoningContent"), dict) and "reasoningText" in block["reasoningContent"]
    ]


def _reject(agent, history):
    before, request, retry = copy.deepcopy(history), copy.deepcopy(history), TurnRetryState()
    assert _recover_format_errors(
        agent, RuntimeError("Invalid `signature` in `thinking` block"),
        SimpleNamespace(reason=FailoverReason.thinking_signature), retry, history, request,
    )
    assert retry.thinking_sig_retry_attempted and history == before
    return request


def test_claude_on_converse_tracks_rejections_and_other_converse_models_do_not():
    assert tracks_rejected_thinking(_agent(CLAUDE)) is True
    assert tracks_rejected_thinking(_agent("eu.anthropic.claude-sonnet-5-5")) is True
    assert tracks_rejected_thinking(_agent("global.moonshotai.kimi-k3")) is False
    assert tracks_rejected_thinking(_agent("deepseek.r1-v1:0")) is False
    assert tracks_rejected_thinking(_agent(None)) is False


def test_rejected_signature_leaves_every_converse_carrier_and_never_returns():
    """Invariant: after one signature 400 the retry and every rebuilt request carry none of the rejected
    signed reasoning (sidecar included), visible answers and tool calls still replay, canonical history is
    untouched, and reasoning minted after the rejection replays normally."""
    db = _ModelConfigDB()
    history = _converse_turn("Q1", "A1", "sig_head", "t1") + _converse_turn("Q2", "A2", "sig_loop", "t2")
    assert _signatures(history) == ["sig_head", "sig_loop"]

    retry_request = _reject(_agent(db=db), history)
    assert _signatures(retry_request) == []
    _system, wire = convert_messages_to_converse(retry_request, model=CLAUDE)
    assert sum("toolUse" in block for message in wire for block in message["content"]) == 2
    assert "why sig_head" not in repr(wire)

    later = history + _converse_turn("Q3", "A3", "sig_new", "t3")
    fresh = _agent(db=db)  # new process, same session: suppression comes from the persisted fingerprints
    rebuilt = copy.deepcopy(later)
    apply_rejected_thinking_suppression(fresh, rebuilt)
    assert _signatures(rebuilt) == ["sig_new"]


def test_non_claude_converse_keeps_the_one_request_reasoning_details_repair():
    history = _converse_turn("Q1", "A1", "sig_kimi", "t1")
    agent = _agent("global.moonshotai.kimi-k3")
    request = _reject(agent, history)
    assert all("reasoning_details" not in message for message in request)
    assert agent._session_db.cfg == {}
