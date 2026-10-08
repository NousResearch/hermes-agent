"""Recovery rebuilds retain narrowed request controls and task progress budgets."""
from types import SimpleNamespace

import pytest

from agent import auxiliary_client as aux


@pytest.mark.parametrize("async_mode", [False, True])
def test_rebuilt_retry_keeps_stripped_fields_and_progress_budget(monkeypatch, async_mode):
    client = SimpleNamespace(api_key="test", base_url="https://chatgpt.com/backend-api/codex")
    wrapped = aux.CodexAuxiliaryClient(client, "gpt-5.6-sol")
    if async_mode:
        wrapped = aux.AsyncCodexAuxiliaryClient(wrapped)
    monkeypatch.setattr(aux, "_get_cached_client", lambda *a, **kw: (wrapped, "gpt-5.6-sol"))
    monkeypatch.setattr(aux, "_get_task_no_progress_timeout", lambda task: 7.0)
    narrowed = {"model": "gpt-5.6-sol", "messages": [], "timeout": 30,
                "no_progress_timeout": 7.0}
    _, sent = aux._prepare_same_provider_retry(
        task="compression", resolved_provider="openai-codex", resolved_model="gpt-5.6-sol",
        resolved_base_url=None, resolved_api_key=None, resolved_api_mode="codex_responses",
        main_runtime=None, final_model="gpt-5.6-sol", messages=[], temperature=0.3,
        max_tokens=64, tools=None, effective_timeout=30, effective_extra_body={},
        reasoning_config=None, async_mode=async_mode, narrowed_kwargs=narrowed)
    assert sent["no_progress_timeout"] == 7.0
    assert "temperature" not in sent and "max_tokens" not in sent



def test_parameter_strip_survives_following_auth_refresh(monkeypatch):
    class BadRequest(Exception):
        status_code = 400

    class Unauthorized(Exception):
        status_code = 401

    client = SimpleNamespace(api_key="stale", base_url="https://openrouter.ai/api/v1")
    monkeypatch.setattr(aux, "_auth_refresh_provider_for_route", lambda *a, **kw: "openrouter")
    monkeypatch.setattr(aux, "_refresh_provider_credentials", lambda *a, **kw: True)
    monkeypatch.setattr(aux, "_recoverable_pool_provider", lambda *a, **kw: None)
    ladder = aux._aux_recovery_ladder(
        BadRequest("Unsupported value: 'temperature' does not support 0.3"),
        client=client, kwargs={"model": "m", "messages": [], "temperature": 0.3},
        task="title_generation", async_mode=False, base_info=str(client.base_url),
        resolved_provider="openrouter", resolved_model="m", resolved_base_url=None,
        resolved_api_key=None, resolved_api_mode=None, final_model="m", max_tokens=None,
        main_runtime=None, route_info=None)
    sent = []

    def perform(step):
        if step.kind == "call":
            sent.append(step.args[1])
            raise Unauthorized("401 Unauthorized")
        assert step.kind == "retry_same_provider"
        kind, args, request = aux._ladder_step_call(step, None, {}, {})
        assert kind == "retry" and args == ()
        sent.append(request["narrowed_kwargs"])
        return "ok"

    assert aux._drive_ladder(ladder, perform) == "ok"
    assert len(sent) == 2 and all("temperature" not in kwargs for kwargs in sent)



@pytest.mark.parametrize("async_mode", [False, True])
def test_codex_fallback_keeps_task_progress_budget(monkeypatch, async_mode):
    leaf = SimpleNamespace(api_key="test", base_url="https://chatgpt.com/backend-api/codex")
    client = aux.CodexAuxiliaryClient(leaf, "m")
    monkeypatch.setattr(aux, "_get_task_no_progress_timeout", lambda task: 7.0)
    if async_mode:
        client = aux.AsyncCodexAuxiliaryClient(client)
    _, kwargs, rebuild = aux._plan_fallback_candidate(
        client, "m", "fallback_chain[0](openai-codex)", task="compression",
        effective_timeout=30, apply_fast_lane=False, messages=[], tools=None,
        temperature=None, max_tokens=None, effective_extra_body={}, reasoning_config=None)
    assert kwargs["no_progress_timeout"] == 7.0
    _, refreshed = rebuild("openai-codex", client, "m")
    assert refreshed["no_progress_timeout"] == 7.0



@pytest.mark.parametrize("async_mode", [False, True])
def test_stream_only_fallback_forces_destination_stream(monkeypatch, async_mode):
    import asyncio

    client = SimpleNamespace(base_url="https://copilot.tencent.com/v1")
    seen = []
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])
    monkeypatch.setattr(aux, "_plan_fallback_candidate", lambda *a, **kw: (
        aux._FallbackDestination("tencent", client.base_url, None, "m"),
        {"model": "m", "messages": []}, None))

    def send(c, kw, task=None, **opts):
        seen.append(opts.get("force_stream", False))
        return response

    async def asend(c, kw, task=None, **opts):
        return send(c, kw, task, **opts)

    monkeypatch.setattr(aux, "_create_with_progress", send)
    monkeypatch.setattr(aux, "_acreate_with_progress", asend)
    kwargs = dict(task="compression", messages=[], temperature=None, max_tokens=None,
                  tools=None, effective_timeout=30, effective_extra_body={}, reasoning_config=None)
    if async_mode:
        result = asyncio.run(aux._call_fallback_candidate_async(client, "m", "tencent", **kwargs))
    else:
        result = aux._call_fallback_candidate_sync(client, "m", "tencent", **kwargs)
    assert result is response
    assert seen == [True]
