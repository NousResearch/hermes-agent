"""Plugin facade route policy over the real auxiliary chain (PR #106486).

Only external HTTP transport and registry data are substituted; request records,
not result attribution, establish that strict calls never reach another route.
"""

import json

import httpx
import openai
import pytest
import hermes_yaml as yaml

from agent import auxiliary_client as aux
from agent.plugin_llm import PluginLlmTrustError, _TrustPolicy, make_plugin_llm_for_test


ENTRIES = ("complete", "acomplete", "complete_structured", "acomplete_structured")
CHAT_ENTRIES = ENTRIES[:2]
SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}


async def invoke(llm, entry, **kwargs):
    if "structured" in entry:
        kwargs.update(instructions="Return an ok flag", input=[{"type": "text", "text": "fixture"}],
                      json_schema=SCHEMA)
    else:
        kwargs["messages"] = [{"role": "user", "content": "fixture"}]
    kwargs.setdefault("task", "demo_task")
    result = getattr(llm, entry)(**kwargs)
    return await result if entry.startswith("a") else result


@pytest.fixture
def route_wire(tmp_path, monkeypatch):
    from hermes_cli import plugins

    primary = dict(provider="custom", model="fixture-model", base_url="https://primary.invalid/v1",
                   api_key="fixture-key")
    helper = dict(provider="custom", model="helper-model", base_url="https://helper.invalid/v1",
                  api_key="fixture-key")
    config = {"model": dict(provider="custom", default="main-model", base_url="https://main.invalid/v1",
                            api_key="fixture-key"),
              "auxiliary": {"transient_retries": 1, "demo_task": dict(primary, fallback_chain=[helper])}}
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(plugins, "get_plugin_auxiliary_tasks", lambda: [
        {"key": "demo_task", "plugin": "demo", "defaults": {}}])
    monkeypatch.setattr(openai._base_client.BaseClient, "_should_retry", lambda *a, **k: False)
    monkeypatch.setattr(aux.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(aux, "_is_provider_unhealthy", lambda *a, **k: False)
    requests = []
    outcome = ["payment"]

    def respond(request):
        body = json.loads(request.content)
        requests.append((request.url.host, body, request.extensions.get("timeout")))
        assert request.url.host in {"primary.invalid", "helper.invalid", "main.invalid"}
        if request.url.host == "primary.invalid" and outcome[0] == "payment":
            return httpx.Response(402, request=request, json={"error": {"message": "Payment Required"}})
        if request.url.host == "primary.invalid":
            if outcome[0] == "connection" or (outcome[0] == "transient" and len(requests) == 1):
                raise httpx.RemoteProtocolError("peer closed connection", request=request)
            if outcome[0] == "timeout":
                raise httpx.ReadTimeout("request timed out", request=request)
            if outcome[0] == "repair" and "response_format" in body:
                return httpx.Response(400, request=request, json={
                    "error": {"message": "response_format is not supported"}})
        return httpx.Response(200, request=request, json={
            "id": "fixture-response", "object": "chat.completion", "created": 1,
            "model": body["model"], "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": '{"ok": true}'}}]})

    monkeypatch.setattr(httpx.Client, "send", lambda _self, request, **_kw: respond(request))

    async def asend(_self, request, **_kw):
        return respond(request)

    monkeypatch.setattr(httpx.AsyncClient, "send", asend)

    def configure():
        (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    aux.shutdown_cached_clients()
    aux.clear_runtime_main()
    configure()
    llm = make_plugin_llm_for_test(plugin_id="demo", policy=_TrustPolicy(plugin_id="demo"))
    try:
        yield llm, requests, config, outcome, configure
    finally:
        aux.shutdown_cached_clients()
        aux.clear_runtime_main()


def routes(requests):
    return [(host, body["model"]) for host, body, _timeout in requests]


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("policy", [None, True, False], ids=["default", "fallback", "strict"])
async def test_payment_route_policy(route_wire, entry, policy):
    llm, requests, _config, _outcome, _configure = route_wire
    kwargs = {} if policy is None else {"allow_fallback": policy}
    if policy is False:
        with pytest.raises(openai.APIStatusError) as caught:
            await invoke(llm, entry, **kwargs)
        assert caught.value.status_code == 402
        assert type(caught.value) is openai.APIStatusError
        assert routes(requests) == [("primary.invalid", "fixture-model")]
    else:
        result = await invoke(llm, entry, **kwargs)
        assert result.text == '{"ok": true}'
        assert routes(requests) == [("primary.invalid", "fixture-model"), ("helper.invalid", "helper-model")]


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", CHAT_ENTRIES)
@pytest.mark.parametrize("strict", [False, True], ids=["default", "strict"])
async def test_empty_chain_main_model_rescue(route_wire, entry, strict):
    llm, requests, config, outcome, configure = route_wire
    config["auxiliary"]["demo_task"]["fallback_chain"] = []
    outcome[0] = "connection"
    configure()
    if strict:
        with pytest.raises(openai.APIConnectionError):
            await invoke(llm, entry, allow_fallback=False)
        assert len(requests) >= 2
        assert set(routes(requests)) == {("primary.invalid", "fixture-model")}
    else:
        assert (await invoke(llm, entry)).text == '{"ok": true}'
        assert routes(requests)[-1] == ("main.invalid", "main-model")
        assert routes(requests)[:-1] == [("primary.invalid", "fixture-model")] * (len(requests) - 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", CHAT_ENTRIES)
@pytest.mark.parametrize("strict", [False, True], ids=["default", "strict"])
async def test_unavailable_task_provider(route_wire, entry, strict):
    llm, requests, config, _outcome, configure = route_wire
    route = config["auxiliary"]["demo_task"]
    route["provider"] = "fixture-unavailable"
    route.pop("base_url")
    route.pop("api_key")
    configure()
    if strict:
        with pytest.raises(RuntimeError):
            await invoke(llm, entry, allow_fallback=False)
        assert requests == []
    else:
        assert (await invoke(llm, entry)).text == '{"ok": true}'
        assert routes(requests) == [("helper.invalid", "helper-model")]


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", CHAT_ENTRIES)
async def test_strict_same_route_transient_retry(route_wire, entry):
    llm, requests, _config, outcome, _configure = route_wire
    outcome[0] = "transient"
    assert (await invoke(llm, entry, allow_fallback=False)).text == '{"ok": true}'
    assert routes(requests) == [("primary.invalid", "fixture-model")] * 2


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", CHAT_ENTRIES)
async def test_default_auto_cache_cannot_become_strict_primary(route_wire, entry):
    llm, requests, config, _outcome, configure = route_wire
    config["model"]["provider"] = "fixture-unavailable"
    config["auxiliary"]["demo_task"] = {
        "provider": "auto", "fallback_chain": config["auxiliary"]["demo_task"]["fallback_chain"]}
    configure()
    assert (await invoke(llm, entry)).text == '{"ok": true}'
    assert routes(requests) == [("helper.invalid", "helper-model")]
    with pytest.raises(RuntimeError):
        await invoke(llm, entry, allow_fallback=False)
    assert routes(requests) == [("helper.invalid", "helper-model")]


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", CHAT_ENTRIES)
@pytest.mark.parametrize("strict", [False, True], ids=["default", "strict"])
async def test_timeout_policy_keeps_requested_bound(route_wire, entry, strict):
    llm, requests, _config, outcome, _configure = route_wire
    outcome[0] = "timeout"
    if strict:
        with pytest.raises(openai.APITimeoutError):
            await invoke(llm, entry, allow_fallback=False, timeout=7)
        assert len(requests) >= 2
        assert set(routes(requests)) == {("primary.invalid", "fixture-model")}
    else:
        assert (await invoke(llm, entry, timeout=7)).text == '{"ok": true}'
        assert routes(requests)[-1] == ("helper.invalid", "helper-model")
    assert all(timeout["read"] == 7 for _host, _body, timeout in requests)


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ENTRIES[2:])
async def test_strict_structured_parameter_repair(route_wire, entry):
    llm, requests, _config, outcome, _configure = route_wire
    outcome[0] = "repair"
    result = await invoke(llm, entry, allow_fallback=False)
    assert result.parsed == {"ok": True}
    assert result.content_type == "json"
    assert routes(requests) == [("primary.invalid", "fixture-model")] * 2
    assert "response_format" in requests[0][1]
    assert "response_format" not in requests[1][1]


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("overrides", [{"provider": "custom"}, {"model": "other-model"},
                                      {"task": "other_plugin_task"}])
async def test_strict_does_not_relax_trust(route_wire, entry, overrides):
    llm, requests, _config, _outcome, _configure = route_wire
    with pytest.raises(PluginLlmTrustError):
        await invoke(llm, entry, allow_fallback=False, **overrides)
    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("value", ["false", 0, None])
async def test_fallback_policy_requires_bool_before_trust(route_wire, entry, value):
    llm, requests, _config, _outcome, _configure = route_wire
    # A provider override would fail trust if it were evaluated before the bool.
    with pytest.raises(TypeError, match="allow_fallback must be a bool"):
        await invoke(llm, entry, allow_fallback=value, provider="custom")
    assert requests == []
