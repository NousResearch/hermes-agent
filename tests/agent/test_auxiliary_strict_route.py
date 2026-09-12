"""Public auxiliary route contracts; SDK requests never leave the process."""

import io
import json
import logging
from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest
import yaml

from agent import auxiliary_client as aux


@pytest.fixture(autouse=True)
def clean_auxiliary_state():
    aux.shutdown_cached_clients()
    aux.clear_runtime_main()
    yield
    aux.shutdown_cached_clients()
    aux.clear_runtime_main()


@pytest.fixture
def nous_wire(tmp_path, monkeypatch):
    from hermes_cli import models

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(aux, "_read_nous_auth", lambda: None)
    monkeypatch.setattr(models, "_nous_recommended_cache", {})
    catalog_requests = []

    def catalog(request, **_kwargs):
        assert request.full_url.endswith(models.NOUS_RECOMMENDED_MODELS_PATH)
        catalog_requests.append(request.full_url)
        return io.BytesIO(json.dumps({
            f"{tier}RecommendedCompactionModel": {"modelName": "recommended-model"}
            for tier in ("free", "paid")
        }).encode())

    monkeypatch.setattr(models, "_urlopen_model_catalog_request", catalog)

    def install(respond):
        requests = []
        clients = []

        def transport_for_url(client, _url):
            def handle(request):
                requests.append(request)
                clients.append(client)
                return respond(request)
            return httpx.MockTransport(handle)

        monkeypatch.setattr(httpx.Client, "_transport_for_url", transport_for_url)
        monkeypatch.setattr(httpx.AsyncClient, "_transport_for_url", transport_for_url)
        return requests, clients, catalog_requests

    return install


def _completion(request):
    return httpx.Response(200, json={
        "id": "fixture-response", "object": "chat.completion", "created": 1,
        "model": json.loads(request.content)["model"],
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "fixture result"}}],
    })


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("provider", ["nous", "auto"])
async def test_strict_nous_refresh_changes_destination(nous_wire, monkeypatch, asynchronous, provider):
    old_base = "https://inference-api.nousresearch.com/v1"
    fresh_base = "https://refreshed-nous.invalid/inference/v2"
    credentials = MagicMock(side_effect=[
        ("stale-fixture-key", old_base), ("fresh-fixture-key", fresh_base),
    ])
    monkeypatch.setattr(aux, "_resolve_nous_runtime_api", credentials)

    def respond(request):
        if request.url.host == "inference-api.nousresearch.com":
            return httpx.Response(401, json={"error": {"message": "stale credential"}})
        assert request.url.host == "refreshed-nous.invalid"
        return _completion(request)

    requests, clients, _catalog = nous_wire(respond)
    kwargs = dict(
        task="researcher", provider=provider, model="selected-model", allow_fallback=False,
        main_runtime={"provider": "nous", "model": "selected-model"},
        messages=[{"role": "user", "content": "fixture prompt"}],
    )
    for _ in range(2):
        response = await aux.async_call_llm(**kwargs) if asynchronous else aux.call_llm(**kwargs)
        assert response.choices[0].message.content == "fixture result"
        assert response.model == "selected-model"

    assert [(str(r.url), r.headers["authorization"], json.loads(r.content)["model"])
            for r in requests] == [
        (old_base + "/chat/completions", "Bearer stale-fixture-key", "selected-model"),
        (fresh_base + "/chat/completions", "Bearer fresh-fixture-key", "selected-model"),
        (fresh_base + "/chat/completions", "Bearer fresh-fixture-key", "selected-model"),
    ]
    assert clients[0] is not clients[1]
    assert clients[1] is clients[2]
    assert [call.kwargs["force_refresh"] for call in credentials.call_args_list] == [False, True]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("provider", ["nous", "auto"])
@pytest.mark.parametrize("strict", [False, True], ids=["default", "strict"])
async def test_public_model_not_found_diagnostic(
    nous_wire, monkeypatch, caplog, asynchronous, provider, strict,
):
    base = "https://inference-api.nousresearch.com/v1"
    credentials = MagicMock(return_value=("fixture-secret-key", base))
    monkeypatch.setattr(aux, "_resolve_nous_runtime_api", credentials)
    details = {"message": "model does not exist; endpoint=https://fixture.invalid/v1?token=private-query",
               "code": "model_not_found", "param": "model", "detail": "original provider detail"}

    def respond(request):
        assert request.url.host == "inference-api.nousresearch.com"
        if json.loads(request.content)["model"] == "selected-model":
            return httpx.Response(404, json={"error": details}, headers={"x-request-id": "fixture-request"})
        return _completion(request)

    requests, _clients, catalog = nous_wire(respond)
    # Observe the SDK-created exception without replacing its construction or propagation.
    sdk = openai.AsyncOpenAI if asynchronous else openai.OpenAI
    make_error = sdk._make_status_error
    errors = []

    def record_error(self, *args, **kwargs):
        error = make_error(self, *args, **kwargs)
        errors.append((error, str(error)))
        return error

    monkeypatch.setattr(sdk, "_make_status_error", record_error)
    kwargs = dict(
        task="researcher", provider=provider, model="selected-model",
        main_runtime={"provider": "nous", "model": "selected-model"},
        messages=[{"role": "user", "content": "fixture prompt"}],
    )
    if strict:
        kwargs["allow_fallback"] = False
    with caplog.at_level(logging.WARNING, logger=aux.__name__):
        if strict:
            with pytest.raises(openai.NotFoundError) as caught:
                if asynchronous:
                    await aux.async_call_llm(**kwargs)
                else:
                    aux.call_llm(**kwargs)
            assert len(errors) == 1
            assert caught.value is errors[0][0]
            assert type(caught.value) is openai.NotFoundError
            assert caught.value.status_code == 404
            assert str(caught.value) == errors[0][1]
            assert caught.value.body == details
            assert caught.value.response.json() == {"error": details}
            assert caught.value.request_id == "fixture-request"
        else:
            response = await aux.async_call_llm(**kwargs) if asynchronous else aux.call_llm(**kwargs)
            assert response.model == "recommended-model"
            assert response.choices[0].message.content == "fixture result"

    assert [json.loads(r.content)["model"] for r in requests] == (
        ["selected-model"] if strict else ["selected-model", "recommended-model"])
    assert len(catalog) == (1 if strict else 2)  # Initial discovery; only default heals.
    credentials.assert_called_once_with(force_refresh=False)
    warnings = [r.getMessage() for r in caplog.records
                if r.name == aux.__name__ and r.levelno == logging.WARNING]
    strict_warnings = [message for message in warnings if "allow_fallback=False" in message]
    if strict:
        assert len(strict_warnings) == 1
        hint = strict_warnings[0]
        assert "selected-model" in hint
        assert "catalog" in hint
        assert "Select an available model" in hint
        assert "allow_fallback=True" in hint
        assert "fixture-secret-key" not in hint
        assert "private-query" not in hint
        assert "https://" not in hint
    else:
        assert not strict_warnings


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("selection", ["explicit", "auto", "vision-auto", "vision-explicit"])
@pytest.mark.parametrize("outcome", ["payment", "unavailable", "transient"])
@pytest.mark.parametrize("policy", [None, True, False], ids=["default", "fallback", "strict"])
async def test_public_route_contract(tmp_path, monkeypatch, asynchronous, selection, outcome, policy):
    task = "vision" if selection.startswith("vision-") else "researcher"
    primary = {
        "provider": "custom", "model": "fixture-model",
        "base_url": "https://researcher.invalid/v1", "api_key": "fixture-key",
    }
    if outcome == "unavailable":
        primary = {"provider": "fixture-unavailable", "model": "fixture-model"}
    alternative = {
        "provider": "custom", "model": "helper-model",
        "base_url": "https://helper.invalid/v1", "api_key": "fixture-key",
    }
    route = dict(primary if selection.endswith("explicit") else {"provider": "auto"})
    route["fallback_chain"] = [alternative]
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"auxiliary": {task: route}}), encoding="utf-8"
    )
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append((request.url.host, body["model"]))
        if request.url.host == "researcher.invalid":
            if outcome == "payment":
                return httpx.Response(402, request=request, json={"error": {"message": "Payment Required"}})
            if outcome == "transient" and len(requests) == 1:
                raise httpx.RemoteProtocolError("peer closed connection", request=request)
        return httpx.Response(200, request=request, json={
            "id": "fixture-response", "object": "chat.completion", "created": 1,
            "model": body["model"], "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "fixture result"}}],
        })

    def send(_self, request, **_kwargs):
        return respond(request)

    async def asend(_self, request, **_kwargs):
        return respond(request)

    monkeypatch.setattr(httpx.Client, "send", send)
    monkeypatch.setattr(httpx.AsyncClient, "send", asend)
    # Let Hermes, rather than the SDK, exercise the transient retry.
    monkeypatch.setattr(openai._base_client.BaseClient, "_should_retry", lambda *a, **k: False)
    monkeypatch.setattr(aux.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(aux, "_is_provider_unhealthy", lambda *a, **k: False)
    # Vision discovery uses a fixed backend list; supply its first candidate locally.
    if selection.startswith("vision-"):
        def vision_backend(*_args, **_kwargs):
            return aux.resolve_provider_client(
                "custom", "helper-model", explicit_base_url=alternative["base_url"],
                explicit_api_key="fixture-key")
        monkeypatch.setattr(aux, "_resolve_strict_vision_backend", vision_backend)
    kwargs = dict(task=task, messages=[{"role": "user", "content": "fixture prompt"}],
                  main_runtime=primary)
    if policy is not None:
        kwargs["allow_fallback"] = policy

    async def invoke():
        return await aux.async_call_llm(**kwargs) if asynchronous else aux.call_llm(**kwargs)

    if policy is False and outcome in {"payment", "unavailable"}:
        with pytest.raises((openai.APIStatusError, RuntimeError)):
            await invoke()
        assert all(host == "researcher.invalid" and model == "fixture-model"
                   for host, model in requests)
        assert bool(requests) == (outcome == "payment")
    else:
        response = await invoke()
        assert response.choices[0].message.content == "fixture result"
        if outcome == "transient":
            assert requests == [("researcher.invalid", "fixture-model")] * 2
        else:
            # Existing vision discovery retains the explicitly requested model
            # while replacing an unavailable backend. Keep default behavior.
            expected_model = (
                "fixture-model"
                if selection == "vision-explicit" and outcome == "unavailable"
                else "helper-model"
            )
            assert requests[-1] == ("helper.invalid", expected_model)
            # A cached auto fallback must not become a strict call's primary route.
            if selection == "auto" and outcome == "unavailable":
                kwargs["allow_fallback"] = False
                count = len(requests)
                with pytest.raises(RuntimeError):
                    await invoke()
                assert len(requests) == count


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("provider", ["nous", "auto"])
@pytest.mark.parametrize("status", [401, 404, 429])
async def test_strict_model_and_credential_recovery(monkeypatch, tmp_path, asynchronous, provider, status):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("{}", encoding="utf-8")
    error = Exception({401: "stale credential", 404: "model does not exist", 429: "rate limit"}[status])
    error.status_code = status
    client = MagicMock()
    client.base_url = "https://inference-api.nousresearch.com/v1"
    client.api_key = "stale-fixture-key"
    client.chat.completions.create = (AsyncMock if asynchronous else MagicMock)(side_effect=error)
    if status == 429:
        client.chat.completions.create.side_effect = [error, error, {"ok": True}]
    fresh = MagicMock()
    fresh.base_url = client.base_url
    fresh.chat.completions.create = (AsyncMock if asynchronous else MagicMock)(return_value={"ok": True})
    monkeypatch.setattr(aux, "_try_nous", lambda **k: (client, "fixture-model"))
    monkeypatch.setattr(aux, "_to_async_client", lambda c, m, **k: (c, m))
    monkeypatch.setattr(aux, "_validate_llm_response", lambda response, *a, **k: response)
    monkeypatch.setattr(aux, "_is_provider_unhealthy", lambda *a, **k: False)
    refresh = MagicMock(return_value=("fresh-fixture-key", str(client.base_url)))
    monkeypatch.setattr(aux, "_resolve_nous_runtime_api", refresh)
    monkeypatch.setattr(aux, "_create_openai_client", lambda **k: fresh)
    heal = MagicMock(return_value="alternative-model")
    monkeypatch.setattr(aux, "_refresh_nous_recommended_model", heal)
    monkeypatch.setattr(aux, "_recoverable_pool_provider", lambda *a, **k: "nous" if status == 429 else None)
    rotate = MagicMock(return_value=True)
    monkeypatch.setattr(aux, "_recover_provider_pool", rotate)
    kwargs = dict(provider=provider, model="fixture-model", allow_fallback=False,
                  main_runtime={"provider": "nous", "model": "fixture-model"},
                  messages=[{"role": "user", "content": "fixture prompt"}])
    if status in {401, 429}:
        result = await aux.async_call_llm(**kwargs) if asynchronous else aux.call_llm(**kwargs)
        assert result == {"ok": True}
        if status == 401:
            assert fresh.chat.completions.create.call_args.kwargs["model"] == "fixture-model"
            refresh.assert_called_once()
            assert not any(entry[0] is client for entry in aux._client_cache.values())
        else:
            rotate.assert_called_once()
            assert [call.kwargs["model"] for call in client.chat.completions.create.call_args_list] == ["fixture-model"] * 3
    else:
        with pytest.raises(Exception, match="model does not exist"):
            if asynchronous:
                await aux.async_call_llm(**kwargs)
            else:
                aux.call_llm(**kwargs)
        refresh.assert_not_called()
    heal.assert_not_called()
