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


def test_route_shaping_keeps_private_native_carrier_for_the_transport():
    """A resumed Gemini/Copilot row carries its signature only as a private reasoning_details record;
    route shaping must leave it for ``shape_wire_carriers`` even on routes that never read the array."""
    from agent.message_sanitization import apply_reasoning_content_policy, reasoning_replay_route
    from agent.reasoning_carriers import CARRIER_TYPE, shape_wire_carriers

    base = "https://generativelanguage.googleapis.com/v1beta/openai"
    record = {"type": CARRIER_TYPE, "extra_content": {"google": {"thought_signature": "SIG"}}}
    row = {"role": "assistant", "content": "391", "reasoning_details": [{"type": "reasoning.text", "text": "t"}, record]}
    route = reasoning_replay_route("chat_completions", "gemini", "gemini-3.5-pro", base)
    api = dict(row)
    apply_reasoning_content_policy(row, api, route.pad, route.carriers)
    wire = shape_wire_carriers([api], model="gemini-3.5-pro", base_url=base)[0]
    assert wire["extra_content"] == record["extra_content"]
    assert "reasoning_details" not in wire


def test_value_complaint_never_strips_reasoning_content_from_a_must_echo_route():
    from types import SimpleNamespace
    from agent.message_sanitization import record_reasoning_field_rejection

    agent = SimpleNamespace(provider="deepseek", model="deepseek-v4-flash", base_url="https://api.deepseek.com/v1",
                            api_mode="chat_completions", _reasoning_rejecting_routes={}, session_id=None)
    sent = [{"role": "assistant", "content": "", "reasoning_content": " ", "tool_calls": [{"id": "c"}]}]
    for body in ("reasoning_content is not allowed to be empty",
                 "Extra inputs are not permitted, field: 'messages[1].reasoning_content'"):
        assert record_reasoning_field_rejection(agent, body, sent) == frozenset()
    assert agent._reasoning_rejecting_routes == {}


# A model name alone never makes a self-hosted server must-echo: vLLM / SGLang / llama.cpp render the
# model's chat template, which turns the " " pad into prompt text (DeepSeek-V4.1: ``<think> </think>``
# where the model wrote ``<think></think>``), and that breaks the server's prefix cache at the turn.
_STORED_PAD = {"role": "assistant", "content": "", "reasoning_content": " ",
               "tool_calls": [{"id": "c3", "type": "function", "function": {"name": "t", "arguments": "{}"}}]}


@pytest.mark.parametrize("provider,base_url,model", [
    ("custom", "http://127.0.0.1:8888/v1", "deepseek-v4.1-flash"),
    ("custom", "http://192.168.1.20:8000/v1", "DeepSeek-V4-Flash"),
    ("custom", "http://vllm:8000/v1", "deepseek-v4-flash"),
    ("lmstudio", "http://localhost:1234/v1", "deepseek-r2-distill"),
    ("custom", "http://100.101.102.103:8080/v1", "MiMo-V2-Flash"),
])
def test_self_hosted_server_gets_what_the_model_wrote(provider, base_url, model):
    route = reasoning_replay_route("chat_completions", provider, model, base_url)
    assert route.pad is False
    turn, bare = _wire(provider, base_url, model)
    assert turn["reasoning_content"] == "thought"  # real reasoning is still replayed
    assert "reasoning_content" not in bare  # a reasoning-less tool call stays reasoning-less
    # A pad stored by an older build (or under a must-echo fallback) is not replayed to it either.
    stored = deepcopy(_STORED_PAD)
    apply_reasoning_content_policy(_STORED_PAD, stored, route.pad, route.carriers)
    assert "reasoning_content" not in stored


@pytest.mark.parametrize("provider,base_url,model", [
    ("deepseek", "http://127.0.0.1:4000/v1", "deepseek-v4-pro"),  # the provider says DeepSeek: pad anywhere
    ("openrouter", "https://openrouter.ai/api/v1", "deepseek/deepseek-v4-flash"),
    ("custom", "https://api.deepseek.com/v1", "deepseek-v4-flash"),
])
def test_hosted_deepseek_routes_still_pad(provider, base_url, model):
    _, bare = _wire(provider, base_url, model)
    assert bare["reasoning_content"] == " "


_MUST_ECHO_400 = ("Error code: 400 - {'error': {'message': 'The `reasoning_content` in the thinking mode must be "
                  "passed back to the API.', 'type': 'invalid_request_error'}}")


def _proxy_agent(model, db=None, base_url="http://127.0.0.1:4000/v1"):
    """A LiteLLM-style proxy on localhost in front of api.deepseek.com: no family rule matches it."""
    agent = _agent(model, db)
    agent.provider, agent.base_url = "custom", base_url
    return agent


def test_self_hosted_proxy_to_a_must_echo_api_is_learned_once(tmp_path):
    from hermes_state import SessionDB
    from agent.message_sanitization import learned_reasoning_echo

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="cli", model="deepseek-v4-pro")
    proxy = _proxy_agent("deepseek-v4-pro", db)
    sent = [{"role": "user", "content": "hi"}, deepcopy(_BARE), {"role": "tool", "tool_call_id": "c2", "content": "ok"}]
    assert reasoning_replay_route("chat_completions", "custom", proxy.model, proxy.base_url).pad is False
    assert _recover(proxy, "Provider returned error", sent) is False  # not the must-echo wording
    assert learned_reasoning_echo(proxy) is False

    assert _recover(proxy, _MUST_ECHO_400, sent) is True
    assert learned_reasoning_echo(proxy) is True
    # The retry re-shapes the same api_messages for the now must-echo route: the tool-call turn is padded.
    route = reasoning_replay_route("chat_completions", "custom", proxy.model, proxy.base_url, echo_opt_in=True)
    reapply_reasoning_echo(sent, route.pad, route.carriers)
    assert sent[1]["reasoning_content"] == " "
    # The same 400 again (the route pads now: a value complaint) -> normal error path, no loop.
    assert _recover(proxy, _MUST_ECHO_400, sent) is False
    # A resumed process (fresh agent, same session) pads from its first request.
    assert learned_reasoning_echo(_proxy_agent("deepseek-v4-pro", db)) is True
    # A sibling model behind the same proxy is not affected.
    assert learned_reasoning_echo(_proxy_agent("qwen3-coder", db)) is False


def test_a_route_that_already_pads_never_learns_from_the_value_complaint():
    from agent.message_sanitization import learned_reasoning_echo, record_reasoning_echo_requirement

    hosted = _proxy_agent("deepseek-v4-flash", base_url="https://api.deepseek.com/v1")
    assert record_reasoning_echo_requirement(hosted, _MUST_ECHO_400) is False
    opted_in = _proxy_agent("deepseek-v4-pro")
    opted_in._reasoning_echo_flag = True  # model.reasoning_echo: true
    assert record_reasoning_echo_requirement(opted_in, _MUST_ECHO_400) is False
    responses = _proxy_agent("deepseek-v4-pro")
    responses.api_mode = "codex_responses"
    assert record_reasoning_echo_requirement(responses, _MUST_ECHO_400) is False
    assert not learned_reasoning_echo(hosted) and not learned_reasoning_echo(responses)
