"""LM Studio capability-aware reasoning projection for auxiliary requests."""

import asyncio
import json
import urllib.error

import pytest


MODEL = "qwen/qwen3.8-27b"
BASE_URL = "http://127.0.0.1:1234/v1"


class _JsonResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self._body


def _prepare_real_lmstudio_request(async_mode, reasoning_config):
    import agent.auxiliary_client as aux

    request_args = dict(
        task="test", provider="lmstudio", model=MODEL, base_url=BASE_URL, api_key="lm-key",
        main_runtime={}, messages=[{"role": "user", "content": "hello"}], temperature=None,
        max_tokens=None, tools=None, timeout=None, extra_body=None,
        reasoning_config=reasoning_config, extra_headers=None, api_mode=None,
        route_info=None, async_mode=async_mode,
    )
    if not async_mode:
        return aux._prepare_aux_request(**request_args)

    async def prepare():
        return aux._prepare_aux_request(**request_args)

    return asyncio.run(prepare())


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    from agent.auxiliary_lmstudio import clear_lmstudio_probe_cache
    clear_lmstudio_probe_cache()
    yield
    clear_lmstudio_probe_cache()


def test_lmstudio_auxiliary_reasoning_uses_published_options(monkeypatch):
    """Auxiliary LM Studio calls must use the same top-level wire as main calls."""
    import agent.auxiliary_client as aux

    monkeypatch.setattr(
        "hermes_cli.models_local.lmstudio_model_reasoning_options",
        lambda *_args, **_kwargs: ["off", "low", "medium"],
    )

    kwargs = aux._build_call_kwargs(
        provider="lmstudio", model="qwen/qwen3.8-27b",
        messages=[{"role": "user", "content": "hello"}],
        base_url="http://127.0.0.1:1234/v1",
        reasoning_config={"enabled": True, "effort": "low"},
    )

    assert kwargs["reasoning_effort"] == "low"
    assert "extra_body" not in kwargs or "reasoning" not in kwargs["extra_body"]


def test_lmstudio_auxiliary_probe_failure_omits_reasoning_wire(monkeypatch):
    """Unknown LM Studio capability must fail closed, not emit generic nested JSON."""
    import agent.auxiliary_client as aux

    monkeypatch.setattr(
        "hermes_cli.models_local.lmstudio_model_reasoning_options",
        lambda *_args, **_kwargs: [],
    )

    kwargs = aux._build_call_kwargs(
        provider="lmstudio", model="qwen/qwen3.8-27b",
        messages=[{"role": "user", "content": "hello"}],
        base_url="http://127.0.0.1:1234/v1",
        reasoning_config={"enabled": True, "effort": "low"},
    )

    assert "reasoning_effort" not in kwargs
    assert "reasoning" not in kwargs.get("extra_body", {})


def test_lmstudio_auxiliary_probe_forwards_api_key(monkeypatch):
    import agent.auxiliary_client as aux

    seen = {}

    def fake_options(model, base_url, api_key=None):
        seen.update(model=model, base_url=base_url, api_key=api_key)
        return ["off", "low"]

    monkeypatch.setattr("hermes_cli.models_local.lmstudio_model_reasoning_options", fake_options)
    kwargs = aux._build_call_kwargs(
        provider="lmstudio", model="qwen/qwen3.8-27b",
        messages=[{"role": "user", "content": "hello"}],
        base_url="http://127.0.0.1:1234/v1", api_key="lm-secret",
        reasoning_config={"enabled": True, "effort": "low"},
    )

    assert seen == {
        "model": "qwen/qwen3.8-27b",
        "base_url": "http://127.0.0.1:1234/v1",
        "api_key": "lm-secret",
    }
    assert kwargs["reasoning_effort"] == "low"


def test_lmstudio_probe_key_comes_from_the_actual_client():
    from types import SimpleNamespace
    from agent.auxiliary_lmstudio import lmstudio_probe_api_key

    client = SimpleNamespace(api_key="fallback-secret")
    assert lmstudio_probe_api_key(client) == "fallback-secret"
    assert lmstudio_probe_api_key(client, "primary-secret") == "primary-secret"


def test_disabled_reasoning_probes_lmstudio_capabilities(monkeypatch):
    import agent.auxiliary_client as aux
    calls = []
    monkeypatch.setattr(
        "hermes_cli.models_local.lmstudio_model_reasoning_options",
        lambda *args, **kwargs: calls.append((args, kwargs)) or ["off", "low"],
    )

    for _ in range(2):
        aux._build_call_kwargs(
            provider="lmstudio", model="qwen/qwen3.8-27b",
            messages=[{"role": "user", "content": "hello"}],
            base_url="http://127.0.0.1:1234/v1",
            reasoning_config={"enabled": False},
        )

    assert len(calls) == 1


@pytest.mark.parametrize("allowed_options", [["off", "low"], ["off"]])
def test_disabled_reasoning_projects_explicit_off_when_catalog_allows_it(monkeypatch, allowed_options):
    import agent.auxiliary_client as aux

    monkeypatch.setattr(
        "hermes_cli.models_local.lmstudio_model_reasoning_options",
        lambda *_args, **_kwargs: allowed_options,
    )

    kwargs = aux._build_call_kwargs(
        provider="lmstudio", model="qwen/qwen3.8-27b",
        messages=[{"role": "user", "content": "hello"}],
        base_url="http://127.0.0.1:1234/v1",
        reasoning_config={"enabled": False},
    )

    assert kwargs["reasoning_effort"] == "none"
    assert "reasoning" not in kwargs.get("extra_body", {})


def test_disabled_reasoning_omits_off_when_catalog_does_not_allow_it(monkeypatch):
    import agent.auxiliary_client as aux

    monkeypatch.setattr(
        "hermes_cli.models_local.lmstudio_model_reasoning_options",
        lambda *_args, **_kwargs: ["low"],
    )

    kwargs = aux._build_call_kwargs(
        provider="lmstudio", model="qwen/qwen3.8-27b",
        messages=[{"role": "user", "content": "hello"}],
        base_url="http://127.0.0.1:1234/v1",
        reasoning_config={"enabled": False},
    )

    assert "reasoning_effort" not in kwargs
    assert "reasoning" not in kwargs.get("extra_body", {})


@pytest.mark.parametrize("failure", ["unavailable", "auth"])
@pytest.mark.parametrize("async_mode", [False, True])
def test_real_lmstudio_probe_failure_is_not_cached_and_restores_reasoning(
    monkeypatch, failure, async_mode,
):
    """Transport and auth failures must not turn into a cached, false empty capability result."""
    from hermes_cli import models

    calls = []

    def fake_open(request, *, timeout):
        calls.append((request, timeout))
        if len(calls) == 1:
            if failure == "unavailable":
                raise urllib.error.URLError("LM Studio is unavailable")
            raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, None)
        return _JsonResponse({
            "models": [{
                "key": MODEL,
                "capabilities": {"reasoning": {"allowed_options": ["off", "low"]}},
            }],
        })

    monkeypatch.setattr(models, "_urlopen_model_catalog_request", fake_open)
    first = _prepare_real_lmstudio_request(async_mode, {"enabled": False})
    second = _prepare_real_lmstudio_request(async_mode, {"enabled": False})

    assert "reasoning_effort" not in first.kwargs
    assert second.kwargs["reasoning_effort"] == "none"
    assert len(calls) == 2


@pytest.mark.parametrize("async_mode", [False, True])
def test_real_lmstudio_empty_reasoning_options_are_cached(monkeypatch, async_mode):
    """A valid catalog without reasoning remains cacheable, preventing needless HTTP probes."""
    from hermes_cli import models

    calls = []

    def fake_open(request, *, timeout):
        calls.append((request, timeout))
        return _JsonResponse({"models": [{"key": MODEL, "capabilities": {}}]})

    monkeypatch.setattr(models, "_urlopen_model_catalog_request", fake_open)
    first = _prepare_real_lmstudio_request(async_mode, {"enabled": True, "effort": "low"})
    second = _prepare_real_lmstudio_request(async_mode, {"enabled": True, "effort": "low"})

    assert "reasoning_effort" not in first.kwargs
    assert "reasoning_effort" not in second.kwargs
    assert len(calls) == 1


def test_enabled_reasoning_probe_is_bounded_cached(monkeypatch):
    import agent.auxiliary_client as aux
    calls = []
    monkeypatch.setattr(
        "hermes_cli.models_local.lmstudio_model_reasoning_options",
        lambda *args, **kwargs: calls.append((args, kwargs)) or ["off", "low"],
    )

    for _ in range(2):
        aux._build_call_kwargs(
            provider="lmstudio", model="qwen/qwen3.8-27b",
            messages=[{"role": "user", "content": "hello"}],
            base_url="http://127.0.0.1:1234/v1",
            reasoning_config={"enabled": True, "effort": "low"},
        )

    assert len(calls) == 1


def test_probe_cache_is_route_scoped_and_secret_safe(monkeypatch):
    import agent.auxiliary_lmstudio as lmstudio

    calls = []
    monkeypatch.setattr(
        "hermes_cli.models_local.lmstudio_model_reasoning_options",
        lambda model, base_url, api_key=None: calls.append((model, base_url, api_key)) or ["off", "low"],
    )

    for route in (
        ("model-a", "http://127.0.0.1:1234/v1", "secret-a"),
        ("model-b", "http://127.0.0.1:1234/v1", "secret-a"),
        ("model-a", "http://127.0.0.1:2345/v1", "secret-a"),
        ("model-a", "http://127.0.0.1:1234/v1", "secret-b"),
    ):
        lmstudio.lmstudio_reasoning_projection("lmstudio", *route)

    assert len(calls) == 4
    with lmstudio._probe_cache_lock:
        cache_keys = list(lmstudio._probe_cache)
    assert len(cache_keys) == 4
    assert all("secret-" not in repr(cache_key) for cache_key in cache_keys)


def test_probe_cache_evicts_oldest_entry_at_bound(monkeypatch):
    import agent.auxiliary_lmstudio as lmstudio

    monkeypatch.setattr(
        "hermes_cli.models_local.lmstudio_model_reasoning_options",
        lambda *_args, **_kwargs: ["off", "low"],
    )

    for index in range(lmstudio._PROBE_CACHE_MAX_ENTRIES + 1):
        lmstudio.lmstudio_reasoning_projection(
            "lmstudio", f"model-{index}", "http://127.0.0.1:1234/v1", "secret"
        )

    with lmstudio._probe_cache_lock:
        assert len(lmstudio._probe_cache) == lmstudio._PROBE_CACHE_MAX_ENTRIES


def _prepare_fallback_request(monkeypatch, *, async_mode):
    import agent.auxiliary_client as aux
    from types import SimpleNamespace

    fallback = SimpleNamespace(
        api_key="fallback-secret",
        base_url="http://127.0.0.1:1234/v1",
        _hermes_aux_effective_provider="lmstudio",
    )
    seen = []
    monkeypatch.setattr(aux, "_get_cached_client", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(
        aux, "_try_configured_fallback_for_unavailable_client",
        lambda *_args: (fallback, "qwen/qwen3.8-27b", "fallback_chain[0](lmstudio)"),
    )
    monkeypatch.setattr(aux, "_resolve_task_provider_model", lambda *args: (
        "qwen-oauth", "qwen3.8-27b", None, "primary-secret", None))
    monkeypatch.setattr(aux, "_get_task_extra_body", lambda *_args: {})
    monkeypatch.setattr(
        "hermes_cli.models_local.lmstudio_model_reasoning_options",
        lambda model, base_url, api_key=None: seen.append(api_key) or ["off", "low"],
    )
    if async_mode:
        monkeypatch.setattr(aux, "_to_async_client", lambda client, model, is_vision=False: (client, model))

    request = aux._prepare_aux_request(
        "test", provider="qwen-oauth", model="qwen3.8-27b", base_url=None,
        api_key="primary-secret", main_runtime={}, messages=[{"role": "user", "content": "hello"}],
        temperature=None, max_tokens=None, tools=None, timeout=None, extra_body=None,
        reasoning_config={"enabled": True, "effort": "low"}, extra_headers=None, api_mode=None,
        route_info=None, async_mode=async_mode,
    )
    return request, seen


def test_unavailable_primary_uses_fallback_provider_credential_sync(monkeypatch):
    request, seen = _prepare_fallback_request(monkeypatch, async_mode=False)
    assert request.request_provider == "lmstudio"
    assert seen == ["fallback-secret"]


def test_unavailable_primary_uses_fallback_provider_credential_async(monkeypatch):
    request, seen = _prepare_fallback_request(monkeypatch, async_mode=True)
    assert request.request_provider == "lmstudio"
    assert seen == ["fallback-secret"]


@pytest.mark.parametrize("async_mode", [False, True])
def test_fallback_route_identity_survives_same_provider_retry(monkeypatch, async_mode):
    import agent.auxiliary_client as aux
    from types import SimpleNamespace

    fallback = SimpleNamespace(
        api_key="fallback-secret",
        base_url="http://127.0.0.1:1234/v1",
        _hermes_aux_effective_provider="lmstudio",
    )
    retry_client = SimpleNamespace(
        api_key="fallback-secret",
        base_url="http://127.0.0.1:1234/v1",
    )
    cached_calls = []

    def cached_client(provider, model=None, **kwargs):
        cached_calls.append((provider, model, kwargs))
        if len(cached_calls) == 1:
            return None, None
        return retry_client, "qwen/qwen3.8-27b"

    seen = []
    monkeypatch.setattr(aux, "_get_cached_client", cached_client)
    monkeypatch.setattr(
        aux, "_try_configured_fallback_for_unavailable_client",
        lambda *_args: (fallback, "qwen/qwen3.8-27b", "fallback_chain[0](lmstudio)"),
    )
    monkeypatch.setattr(
        aux, "_fallback_destination",
        lambda *_args: aux._FallbackDestination(
            "lmstudio", fallback.base_url, "chat_completions", "qwen/qwen3.8-27b"
        ),
    )
    monkeypatch.setattr(aux, "_fallback_entry_api_key", lambda _entry: "fallback-secret")
    monkeypatch.setattr(aux, "_resolve_task_provider_model", lambda *args: (
        "qwen-oauth", "qwen3.8-27b", None, "primary-secret", None))
    monkeypatch.setattr(aux, "_get_task_extra_body", lambda *_args: {})
    monkeypatch.setattr(
        "hermes_cli.models_local.lmstudio_model_reasoning_options",
        lambda _model, _base_url, api_key=None: seen.append(api_key) or ["off", "low"],
    )
    if async_mode:
        monkeypatch.setattr(aux, "_to_async_client", lambda client, model, is_vision=False: (client, model))

    request = aux._prepare_aux_request(
        "test", provider="qwen-oauth", model="qwen3.8-27b", base_url=None,
        api_key="primary-secret", main_runtime={}, messages=[{"role": "user", "content": "hello"}],
        temperature=None, max_tokens=None, tools=None, timeout=None, extra_body=None,
        reasoning_config={"enabled": True, "effort": "low"}, extra_headers=None, api_mode=None,
        route_info=None, async_mode=async_mode,
    )

    assert request.resolved_provider == "lmstudio"
    assert request.resolved_model == "qwen/qwen3.8-27b"
    assert request.resolved_base_url == fallback.base_url
    assert request.resolved_api_key == "fallback-secret"
    assert seen == ["fallback-secret"]

    from agent.auxiliary_lmstudio import clear_lmstudio_probe_cache
    clear_lmstudio_probe_cache()
    aux._prepare_same_provider_retry(
        task="test", resolved_provider=request.resolved_provider,
        resolved_model=request.resolved_model, resolved_base_url=request.resolved_base_url,
        resolved_api_key=request.resolved_api_key, resolved_api_mode=request.resolved_api_mode,
        main_runtime={}, final_model=request.final_model,
        messages=[{"role": "user", "content": "hello"}], temperature=None, max_tokens=None,
        tools=None, effective_timeout=request.effective_timeout,
        effective_extra_body=request.effective_extra_body,
        reasoning_config={"enabled": True, "effort": "low"}, async_mode=async_mode,
    )

    assert seen == ["fallback-secret", "fallback-secret"]
    assert cached_calls[1][0] == "lmstudio"
    assert cached_calls[1][2]["base_url"] == fallback.base_url
    assert cached_calls[1][2]["api_key"] == "fallback-secret"


def test_auto_route_identity_survives_recovery(monkeypatch):
    import agent.auxiliary_client as aux
    from types import SimpleNamespace

    selected = SimpleNamespace(
        api_key="auto-secret",
        base_url="http://127.0.0.1:1234/v1",
        _hermes_aux_effective_provider="lmstudio",
    )
    cached_calls = []

    def cached_client(*args, **kwargs):
        cached_calls.append((args, kwargs))
        return (None, None) if len(cached_calls) == 1 else (selected, "qwen/qwen3.8-27b")

    monkeypatch.setattr(aux, "_get_cached_client", cached_client)
    monkeypatch.setattr(aux, "_resolve_task_provider_model", lambda *args: (
        "auto", None, None, None, None))
    monkeypatch.setattr(aux, "_get_task_extra_body", lambda *_args: {})
    monkeypatch.setattr(
        "hermes_cli.models_local.lmstudio_model_reasoning_options",
        lambda _model, _base_url, api_key=None: ["off", "low"],
    )

    request = aux._prepare_aux_request(
        "test", provider="auto", model=None, base_url=None, api_key=None, main_runtime={},
        messages=[{"role": "user", "content": "hello"}], temperature=None, max_tokens=None,
        tools=None, timeout=None, extra_body=None,
        reasoning_config={"enabled": True, "effort": "low"}, extra_headers=None, api_mode=None,
        route_info=None, async_mode=False,
    )

    assert request.resolved_provider == "auto"
    assert request.request_provider == "lmstudio"
    assert request.resolved_model == "qwen/qwen3.8-27b"
    assert request.resolved_base_url == selected.base_url
    assert request.resolved_api_key == "auto-secret"


def test_pool_rotation_retry_drops_failed_credential(monkeypatch):
    import agent.auxiliary_client as aux
    from types import SimpleNamespace

    class AuthError(Exception):
        status_code = 401

    route = aux._LadderRoute(
        SimpleNamespace(api_key="failed-secret", base_url="https://openrouter.ai/api/v1"),
        "test", "", False, "https://openrouter.ai/api/v1", "auto", "model",
        "https://openrouter.ai/api/v1", "failed-secret", None, "model", {}, None, 30.0,
    )
    monkeypatch.setattr(aux, "_recoverable_pool_provider", lambda *args, **kwargs: "openrouter")
    monkeypatch.setattr(aux, "_recover_provider_pool", lambda *args, **kwargs: True)

    step = next(aux._ladder_credential_rungs(AuthError("expired"), route, {}, False))

    assert step.kind == "retry_same_provider"
    assert step.args == ("openrouter", "model", None)
    _, _, retry_kwargs = aux._ladder_step_call(
        step, SimpleNamespace(resolved_provider="auto", resolved_api_mode=None),
        {"resolved_api_key": "failed-secret"}, {},
    )
    assert retry_kwargs["resolved_api_key"] is None


def test_vision_route_identity_updates_after_provider_switch(monkeypatch):
    import agent.auxiliary_client as aux
    from types import SimpleNamespace

    fallback = SimpleNamespace(
        api_key="vision-secret",
        base_url="http://127.0.0.1:1234/v1",
        _hermes_aux_effective_provider="lmstudio",
    )
    monkeypatch.setattr(
        aux, "resolve_vision_provider_client",
        lambda **kwargs: ("lmstudio", fallback, "qwen/qwen3.8-27b"),
    )
    monkeypatch.setattr(
        "hermes_cli.models_local.lmstudio_model_reasoning_options",
        lambda _model, _base_url, api_key=None: ["off", "low"],
    )

    request = aux._prepare_aux_request(
        "vision", provider="qwen-oauth", model="qwen3.8-27b", base_url=None,
        api_key="primary-secret", main_runtime={}, messages=[{"role": "user", "content": "hello"}],
        temperature=None, max_tokens=None, tools=None, timeout=None, extra_body=None,
        reasoning_config={"enabled": True, "effort": "low"}, extra_headers=None, api_mode=None,
        route_info=None, async_mode=False,
    )

    assert request.request_provider == "lmstudio"
    assert request.resolved_model == "qwen/qwen3.8-27b"
    assert request.resolved_base_url == fallback.base_url
    assert request.resolved_api_key == "vision-secret"
    assert request.kwargs["reasoning_effort"] == "low"


def test_provider_switched_vision_route_does_not_inherit_primary_key(monkeypatch):
    import agent.auxiliary_client as aux
    from types import SimpleNamespace

    fallback = SimpleNamespace(
        api_key="",
        base_url="http://127.0.0.1:1234/v1",
        _hermes_aux_effective_provider="lmstudio",
    )
    monkeypatch.setattr(
        aux, "resolve_vision_provider_client",
        lambda **kwargs: ("lmstudio", fallback, "qwen/qwen3.8-27b"),
    )

    route = aux._resolve_call_client(
        "vision", provider="primary", model="primary-model", base_url=None,
        api_key="primary-secret", resolved_provider="primary", resolved_model="primary-model",
        resolved_base_url=None, resolved_api_key="primary-secret", resolved_api_mode=None,
        main_runtime=None, async_mode=False,
    )

    assert route.resolved_provider == "lmstudio"
    assert route.api_key is None


@pytest.mark.parametrize("async_mode", [False, True])
def test_same_provider_zai_vision_route_uses_selected_transport_and_key(monkeypatch, async_mode):
    """ZAI's forced OpenAI vision client must replace stale config transport on retries too."""
    import asyncio
    from types import SimpleNamespace

    import agent.auxiliary_client as aux

    client = SimpleNamespace(
        api_key="zai-secret",
        base_url="https://api.z.ai/api/paas/v4",
    )
    monkeypatch.setattr(
        aux, "_resolve_task_provider_model",
        lambda *_args: ("zai", "glm-5.3-flash", None, "primary-secret", "anthropic_messages"),
    )
    monkeypatch.setattr(
        aux,
        "resolve_vision_provider_client",
        lambda **_kwargs: ("zai", client, "glm-5.3-flash"),
    )
    monkeypatch.setattr(
        aux,
        "_build_call_kwargs",
        lambda _provider, _model, _messages, **kwargs: {"api_key": kwargs["api_key"]},
    )
    monkeypatch.setattr(aux, "_validate_llm_response", lambda response, _task: response)

    request = aux._prepare_aux_request(
        "vision", provider="zai", model="glm-5.3-flash", base_url=None,
        api_key="primary-secret", main_runtime={}, messages=[], temperature=None,
        max_tokens=None, tools=None, timeout=None, extra_body=None, reasoning_config=None,
        extra_headers=None, api_mode=None, route_info=None, async_mode=async_mode,
    )

    assert request.resolved_api_mode == "chat_completions"
    assert request.resolved_api_key == "zai-secret"

    seen = []
    if async_mode:
        async def relay(_client, kwargs, **relay_kwargs):
            seen.append((relay_kwargs["api_mode"], kwargs["api_key"]))
            return "ok"

        monkeypatch.setattr(aux, "_relay_async_completion", relay)
        asyncio.run(aux._relay_async_completion(
            request.client, request.kwargs, provider=request.request_provider,
            api_mode=request.resolved_api_mode,
        ))
        asyncio.run(aux._retry_same_provider_async(
            resolved_provider=request.resolved_provider, resolved_api_mode=request.resolved_api_mode,
            task="vision", resolved_model=request.resolved_model,
            resolved_base_url=request.resolved_base_url, resolved_api_key=request.resolved_api_key,
            main_runtime={}, final_model=request.final_model, messages=[], temperature=None,
            max_tokens=None, tools=None, effective_timeout=request.effective_timeout,
            effective_extra_body={}, reasoning_config=None,
        ))
    else:
        monkeypatch.setattr(
            aux, "_relay_sync_completion",
            lambda _client, kwargs, **relay_kwargs: seen.append(
                (relay_kwargs["api_mode"], kwargs["api_key"])
            ) or "ok",
        )
        aux._relay_sync_completion(
            request.client, request.kwargs, provider=request.request_provider,
            api_mode=request.resolved_api_mode,
        )
        aux._retry_same_provider_sync(
            resolved_provider=request.resolved_provider, resolved_api_mode=request.resolved_api_mode,
            task="vision", resolved_model=request.resolved_model,
            resolved_base_url=request.resolved_base_url, resolved_api_key=request.resolved_api_key,
            main_runtime={}, final_model=request.final_model, messages=[], temperature=None,
            max_tokens=None, tools=None, effective_timeout=request.effective_timeout,
            effective_extra_body={}, reasoning_config=None,
        )

    assert seen == [
        ("chat_completions", "zai-secret"),
        ("chat_completions", "zai-secret"),
    ]


@pytest.mark.parametrize("async_mode", [False, True])
def test_provider_switched_vision_route_uses_selected_transport_for_retry(monkeypatch, async_mode):
    """A vision provider switch must replace the primary wire mode on every retry."""
    import asyncio
    import agent.auxiliary_client as aux
    from types import SimpleNamespace

    fallback = SimpleNamespace(
        api_key="",
        base_url="http://127.0.0.1:1234/v1",
        _hermes_aux_effective_provider="lmstudio",
    )
    monkeypatch.setattr(
        aux,
        "resolve_vision_provider_client",
        lambda **kwargs: ("lmstudio", fallback, "qwen/qwen3.8-27b"),
    )

    route = aux._resolve_call_client(
        "vision", provider="primary", model="primary-model", base_url=None,
        api_key="primary-secret", resolved_provider="primary", resolved_model="primary-model",
        resolved_base_url=None, resolved_api_key="primary-secret",
        resolved_api_mode="anthropic_messages", main_runtime=None, async_mode=async_mode,
    )

    assert route.resolved_provider == "lmstudio"
    assert route.resolved_api_mode == "chat_completions"
    assert route.resolved_api_key is None

    with aux._relay_aux_call_scope(("vision",), {}):
        aux._set_relay_auxiliary_route(route.effective_provider, route.final_model, route.resolved_api_mode)
        assert aux._RELAY_AUX_CALL_CONTEXT.get()["api_mode"] == "chat_completions"

    seen = []
    if async_mode:
        async def relay(*args, **kwargs):
            seen.append(kwargs["api_mode"])
            return "ok"

        monkeypatch.setattr(aux, "_relay_async_completion", relay)
        monkeypatch.setattr(aux, "_validate_llm_response", lambda response, task: response)
        asyncio.run(aux._retry_same_provider_async(
            resolved_provider=route.resolved_provider, resolved_api_mode=route.resolved_api_mode,
            task="vision", resolved_model=route.resolved_model,
            resolved_base_url=route.resolved_base_url, resolved_api_key=route.resolved_api_key,
            main_runtime=None, final_model=route.final_model,
            messages=[], temperature=None, max_tokens=None, tools=None, effective_timeout=30.0,
            effective_extra_body={}, reasoning_config=None,
        ))
    else:
        monkeypatch.setattr(
            aux, "_relay_sync_completion",
            lambda *args, **kwargs: seen.append(kwargs["api_mode"]) or "ok",
        )
        monkeypatch.setattr(aux, "_validate_llm_response", lambda response, task: response)
        aux._retry_same_provider_sync(
            resolved_provider=route.resolved_provider, resolved_api_mode=route.resolved_api_mode,
            task="vision", resolved_model=route.resolved_model,
            resolved_base_url=route.resolved_base_url, resolved_api_key=route.resolved_api_key,
            main_runtime=None, final_model=route.final_model,
            messages=[], temperature=None, max_tokens=None, tools=None, effective_timeout=30.0,
            effective_extra_body={}, reasoning_config=None,
        )

    assert seen == ["chat_completions"]
