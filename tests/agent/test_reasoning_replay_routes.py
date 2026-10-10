"""Stored reasoning is replayed to every chat-completions route that can read it, on the
carriers that route reads; strict schemas never receive a reasoning key; a route that still
rejects one is learned per (provider, host, model) and stripped on a single retry."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from agent.message_sanitization import (
    apply_reasoning_content_policy, reapply_reasoning_echo, reasoning_replay_route, rejected_reasoning_carriers,
)
from agent.transports.chat_completions import ChatCompletionsTransport

_DETAILS = [{"type": "reasoning.text", "text": "thought"}]
_TURN = {"role": "assistant", "content": "", "reasoning": "thought", "reasoning_content": "thought",
         "reasoning_details": _DETAILS,
         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"},
                         "extra_content": {"google": {"thought_signature": "sig"}}}]}
_BARE = {"role": "assistant", "content": "", "tool_calls": [{"id": "c2", "type": "function",
                                                             "function": {"name": "t", "arguments": "{}"}}]}


def _wire(provider, base_url, model="m", rejected=()):
    route = reasoning_replay_route("chat_completions", provider, model, base_url, rejected=rejected)
    out = []
    for source in (_TURN, _BARE):
        api_msg = deepcopy(source)
        apply_reasoning_content_policy(source, api_msg, route.pad, route.carriers)
        out.append(api_msg)
    return ChatCompletionsTransport().convert_messages(out, model=model, base_url=base_url, provider=provider)


@pytest.mark.parametrize("provider,base_url,model,expected", [
    # strict schemas: nothing, not even the must-echo pad (#45655, #70233)
    ("mistral", "https://api.mistral.ai/v1", "deepseek-r1", set()),
    ("custom", "https://api.groq.com/openai/v1", "m", set()),
    ("custom", "https://api.cerebras.ai/v1", "m", set()),
    ("fireworks", "https://api.fireworks.ai/inference/v1", "m", {"reasoning_content"}),
    # OpenRouter-format gateways, incl. the Portal (#118182 does not reproduce)
    ("nous", "https://inference-api.nousresearch.com/v1", "z-ai/glm-5.3",
     {"reasoning_content", "reasoning", "reasoning_details"}),
    ("kilocode", "https://api.kilo.ai/api/gateway", "m", {"reasoning_content", "reasoning", "reasoning_details"}),
    ("novita", "https://api.novita.ai/openai/v1", "m", {"reasoning_content", "reasoning_details"}),
    ("ollama-cloud", "https://ollama.com/v1", "m", {"reasoning_content", "reasoning"}),
    ("custom", "http://127.0.0.1:8080/v1", "m", {"reasoning_content", "reasoning"}),
    # everyone else: the de-facto standard alias
    ("zai", "https://api.z.ai/api/paas/v4", "glm-5.3", {"reasoning_content"}),
    ("openai-api", "https://api.openai.com/v1", "gpt-5", {"reasoning_content"}),
])
def test_each_route_gets_exactly_the_carriers_it_reads(provider, base_url, model, expected):
    turn, bare = _wire(provider, base_url, model)
    assert {k for k in ("reasoning_content", "reasoning", "reasoning_details") if k in turn} == expected
    if "reasoning_content" in expected:
        assert turn["reasoning_content"] == "thought"
    assert "extra_content" not in turn["tool_calls"][0]  # Gemini routes only
    # A reasoning-less turn is never padded outside the must-echo tier.
    assert "reasoning_content" not in bare
    assert _TURN["reasoning_details"] is _DETAILS and "reasoning" in _TURN  # history untouched


def test_must_echo_pads_and_aliased_gemini_keeps_its_signature():
    _, bare = _wire("deepseek", "https://api.deepseek.com/v1", "deepseek-v4-flash")
    assert bare["reasoning_content"] == " "  # DeepSeek 400s without it; "" is rejected too
    gemini_alias, _ = _wire("gemini", "https://generativelanguage.googleapis.com/v1beta/openai", "my-alias")
    assert gemini_alias["tool_calls"][0]["extra_content"] == {"google": {"thought_signature": "sig"}}


class _Rejection(Exception):
    status_code = 400

    def __init__(self, body):
        super().__init__(body)
        self.body = body


def _agent(model, db=None):
    return SimpleNamespace(
        provider="nous", model=model, base_url="https://inference-api.nousresearch.com/v1",
        api_mode="chat_completions", _force_ascii_payload=False, _image_rejecting_models=set(),
        log_prefix="", _vprint=lambda *a, **k: None, session_id="s1" if db else None, _session_db=db,
    )


def _recover(agent, body, sent):
    from agent.turn_recovery import recover_before_classification

    return recover_before_classification(
        agent, _Rejection(body), messages=[], api_messages=sent,
        api_kwargs={"messages": sent}, active_system_prompt="",
    )[0]


def test_named_field_rejection_strips_once_per_model_and_never_loops(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="cli", model="z-ai/glm-5.3-flash:US")
    lane = _agent("z-ai/glm-5.3-flash:US", db)
    sent = [{"role": "user", "content": "hi"}, deepcopy(_TURN)]
    sent[1].pop("tool_calls")
    body = "Extra inputs are not permitted, field: 'messages[1].reasoning_details'"
    # A generic upstream error (content filter behind "Provider returned error") is not a rejection.
    assert _recover(lane, "Provider returned error", sent) is False
    assert not rejected_reasoning_carriers(lane)

    assert _recover(lane, body, sent) is True
    rejected = lane._reasoning_rejecting_routes[("nous", "inference-api.nousresearch.com", "z-ai/glm-5.3-flash:US")]
    assert rejected == {"reasoning", "reasoning_details"}
    # The retry re-shapes the same api_messages for the narrowed route: reasoning_content survives.
    route = reasoning_replay_route("chat_completions", "nous", lane.model, lane.base_url, rejected=rejected)
    reapply_reasoning_echo(sent, route.pad, route.carriers)
    assert sent[1]["reasoning_content"] == "thought"
    assert "reasoning" not in sent[1] and "reasoning_details" not in sent[1]
    # Same rejection again: nothing new to strip -> normal error path, no loop.
    assert _recover(lane, body, [{"role": "assistant", "reasoning_details": _DETAILS}]) is False
    # A resumed process (fresh agent, same session) never re-learns it with another 400.
    assert rejected_reasoning_carriers(_agent(lane.model, db)) == {"reasoning", "reasoning_details"}
    # A sibling model on the same gateway keeps every carrier.
    sibling = reasoning_replay_route("chat_completions", "nous", "moonshotai/kimi-k3", lane.base_url)
    assert "reasoning_details" in sibling.carriers
