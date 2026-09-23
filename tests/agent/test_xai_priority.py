"""Priority follows the requested route, never prompt state or the preceding response."""

import asyncio
import copy
import json
from types import SimpleNamespace

import httpx
import openai
import pytest
import yaml

from agent import fast_mode
from agent.auxiliary_client import _CodexCompletionsAdapter, _AsyncCompletionsAdapter
from agent.codex_runtime import _consume_codex_event_stream
from agent.transports.codex import ResponsesApiTransport
from hermes_cli.models import resolve_fast_mode_overrides

URL = "https://api.x.ai/v1"
MESSAGES = [{"role": "system", "content": "Stable instructions"}, {"role": "user", "content": "Hi"}]
TOOLS = [{"type": "function", "function": {"name": "constant", "description": "Return a constant",
          "parameters": {"type": "object", "properties": {}}}}]


@pytest.mark.parametrize("model,provider,url,eligible", [
    ("grok-4.6", "xai", URL, True),
    ("x-ai/grok-4.6-latest", "xai", URL, True),
    ("grok-4.7", "xai", URL, True),
    ("XAI/GROK_4.7", "grok", URL + "/", True),
    ("grok-4.70", "xai", URL, False),
    ("grok-4.7-fast", "xai", URL, False),
    ("grok-4.8", "xai", URL, False),
    ("grok-4.5", "xai", URL, False),
    ("grok-4.7", "xai-oauth", URL, False),
    ("grok-4.7", "openrouter", URL, False),
    ("grok-4.7", "custom", URL, False),
    ("grok-4.7", "xai", "https://us-east-1.api.x.ai/v1", False),
    ("grok-4.7", "xai", "https://proxy.example/v1", False),
    ("grok-4.7", "xai", "https://api.x.ai/proxy/v1", False),
    ("grok-4.7", "xai", "http://api.x.ai/v1", False),
])
def test_selector_main_and_aux_share_route_gate(model, provider, url, eligible):
    overrides = {"service_tier": "priority"}
    assert bool(resolve_fast_mode_overrides(model, provider=provider, base_url=url)) is eligible
    main = ResponsesApiTransport().build_kwargs(
        model, MESSAGES, TOOLS, provider=provider, base_url=url, is_xai_responses=True,
        request_overrides=overrides)
    aux, _, _ = _CodexCompletionsAdapter(SimpleNamespace(
        base_url=url, _hermes_aux_effective_provider=provider), model)._build_responses_kwargs(
            dict(messages=MESSAGES, tools=TOOLS, extra_body=overrides))
    assert (main.get("service_tier") == "priority") is eligible
    assert (aux.get("service_tier") == "priority") is eligible


@pytest.mark.parametrize("mode", [None, "priority", "auto", "cold"])
def test_windows_route_switch_and_prompt_invariants(mode, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(fast_mode.time, "monotonic", lambda: now[0])
    agent = SimpleNamespace(model="grok-4.7", provider="xai", base_url=URL, service_tier=mode,
                            request_overrides={"service_tier": "priority"} if mode == "priority" else {},
                            fast_auto_seconds=60)
    original = copy.deepcopy((MESSAGES, TOOLS, agent.request_overrides))
    fast_mode.begin_turn(agent, [])
    def request():
        return ResponsesApiTransport().build_kwargs(
            agent.model, MESSAGES, TOOLS, provider=agent.provider, base_url=agent.base_url,
            is_xai_responses=True, session_id="stable-session",
            request_overrides=fast_mode.effective_request_overrides(agent))
    first = request()
    assert (first.get("service_tier") == "priority") is (mode is not None)
    now[0] += 61
    expired = request()
    assert (expired.get("service_tier") == "priority") is (mode == "priority")
    fast_mode.begin_turn(agent, [{"role": "user", "content": "prior"}])
    next_turn = request()
    assert (next_turn.get("service_tier") == "priority") is (mode in ("auto", "priority"))
    agent.provider = "xai-oauth"
    switched = request()
    assert "service_tier" not in switched
    for payload in (expired, next_turn, switched):
        assert {k: v for k, v in first.items() if k != "service_tier"} == {
            k: v for k, v in payload.items() if k != "service_tier"}
    assert (MESSAGES, TOOLS, agent.request_overrides) == original


@pytest.mark.parametrize("tier", ["priority", "default", None])
@pytest.mark.parametrize("async_mode", [False, True])
def test_aux_sync_async_stream_preserves_actual_tier(tier, async_mode):
    wire = []
    def respond(request):
        wire.append(json.loads(request.content))
        terminal = {"id": "r", "status": "completed", "usage": {"input_tokens": 1, "output_tokens": 1}}
        if tier is not None:
            terminal["service_tier"] = tier
        events = [
            {"type": "response.output_text.delta", "delta": "OK"},
            {"type": "response.completed", "response": terminal},
        ]
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text="".join("data: " + json.dumps(e) + "\n\n" for e in events))
    with openai.OpenAI(api_key="test-key", base_url=URL,
                       http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
        client._hermes_aux_effective_provider = "xai"
        adapter = _CodexCompletionsAdapter(client, "grok-4.7")
        kwargs = dict(messages=MESSAGES, extra_body={"service_tier": "priority"})
        result = (asyncio.run(_AsyncCompletionsAdapter(adapter).create(**kwargs))
                  if async_mode else adapter.create(**kwargs))
        assert result.service_tier == tier
        assert result.choices[0].message.content == "OK"
        assert wire[0]["service_tier"] == "priority"


def test_extra_body_is_filtered_without_mutating_caller():
    extra = {"service_tier": "priority", "keep": 1}
    kwargs = ResponsesApiTransport().build_kwargs(
        "grok-4.7", MESSAGES, provider="xai-oauth", base_url=URL, is_xai_responses=True,
        request_overrides={"service_tier": "priority", "extra_body": extra})
    assert "service_tier" not in kwargs
    assert kwargs["extra_body"] == {"keep": 1}
    assert extra["service_tier"] == "priority"


@pytest.mark.parametrize("event_type", ["response.completed", "response.incomplete", "response.failed"])
def test_stream_terminal_tier_is_authoritative(event_type):
    for tier in ("priority", "default", None):
        response = {} if tier is None else {"service_tier": tier}
        result = _consume_codex_event_stream([
            {"type": "response.created", "response": {"service_tier": "priority"}},
            {"type": event_type, "response": response}], model="grok-4.7")
        assert result.service_tier == tier


def test_aux_profiles_a_b_a_use_real_config_and_do_not_inherit_main_priority(tmp_path, monkeypatch):
    from agent import auxiliary_client as aux
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    wire = []
    clients = []
    def respond(request):
        body = json.loads(request.content)
        wire.append(body)
        events = [{"type": "response.output_text.delta", "delta": "OK"},
                  {"type": "response.completed", "response": {"id": "r", "status": "completed"}}]
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text="".join("data: " + json.dumps(e) + "\n\n" for e in events))
    def create_client(**kwargs):
        client = openai.OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond)))
        clients.append(client)
        return client
    monkeypatch.setattr(aux, "_create_openai_client", create_client)
    homes = [tmp_path / "A", tmp_path / "B"]
    for home, priority in zip(homes, (True, False)):
        home.mkdir()
        task = dict(provider="xai", model="grok-4.7", base_url=URL, api_key="test-key", api_mode="codex_responses")
        if priority:
            task["extra_body"] = {"service_tier": "priority"}
        (home / "config.yaml").write_text(yaml.safe_dump({"agent": {"service_tier": "fast"},
            "model": {"provider": "xai", "model": "grok-4.7"}, "auxiliary": {"compression": task}}), encoding="utf-8")
    for home, expected in ((homes[0], "priority"), (homes[1], None), (homes[0], "priority")):
        token = set_hermes_home_override(home)
        try:
            response = aux.call_llm(task="compression", messages=MESSAGES, timeout=10)
            assert response.service_tier is None
            assert wire[-1].get("service_tier") == expected, (
                aux._get_task_extra_body("compression"),
                [(str(c.base_url), getattr(c, "_hermes_aux_effective_provider", None)) for c in clients],
                aux._resolve_task_provider_model("compression"))
        finally:
            reset_hermes_home_override(token)
