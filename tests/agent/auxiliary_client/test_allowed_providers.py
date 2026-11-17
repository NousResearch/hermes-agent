"""``auxiliary.allowed_providers``: an opt-in allow-list that keeps every auxiliary call on the
listed providers.

The fixture config is a single-provider deployment: one named provider entry (``hub``) as the main
model provider, an OpenRouter key in the environment (so the built-in discovery chain WOULD find
it), and the allow-list naming only ``hub``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest

import hermes_yaml as yaml
from agent import auxiliary_client as aux
from agent.auxiliary_health import (
    AuxiliaryProviderNotAllowed, _allowed_labels, aux_client_route, aux_route_allowed, require_aux_client_allowed,
    require_aux_route_allowed,
)

HUB_URL = "https://hub.example.com/v1"
ELSEWHERE = "https://elsewhere.example.com/v1"
MSGS = [{"role": "user", "content": "hi"}]


def _config(allowed=None, **auxiliary):
    cfg = {
        "model": {"provider": "hub", "default": "hub-model"},
        "providers": {"hub": {"base_url": HUB_URL, "api_key": "test-key", "api_mode": "chat_completions"}},
        "auxiliary": {"transient_retries": 0, **auxiliary},
    }
    if allowed is not None:
        cfg["auxiliary"]["allowed_providers"] = allowed
    return cfg


@pytest.fixture
def home(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")  # what an unrestricted chain would pick up
    with aux._client_cache_lock:
        aux._client_cache.clear()
    aux._reset_aux_unhealthy_cache()

    def write(cfg):
        (hermes_home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    yield write
    with aux._client_cache_lock:
        aux._client_cache.clear()
    aux._reset_aux_unhealthy_cache()


def _labels():
    return [label for label, _ in aux._get_provider_chain()]


def _completion(content):
    return {"id": "c-1", "object": "chat.completion", "created": 0, "model": "hub-model",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


@pytest.fixture
def wire(monkeypatch):
    """Answer every HTTP request in-process (sync and async) and record its URL."""
    seen, reply = [], {"content": "ok"}

    def send(self, request, *args, **kwargs):
        seen.append(str(request.url))
        return httpx.Response(200, json=_completion(reply["content"]), request=request)

    async def asend(self, request, *args, **kwargs):
        return send(self, request)

    monkeypatch.setattr(httpx.Client, "send", send)
    monkeypatch.setattr(httpx.AsyncClient, "send", asend)
    return seen, reply


def _live_runtime(model="hub-model"):
    """The runtime a turn records for a profile whose ``model.provider`` is the named entry ``hub``."""
    from hermes_cli.runtime_provider import resolve_runtime_provider

    rt = resolve_runtime_provider(requested="hub")
    return {"provider": rt["provider"], "requested_provider": rt.get("requested_provider") or "",
            "model": model, "base_url": rt["base_url"], "api_key": rt["api_key"],
            "api_mode": rt.get("api_mode") or ""}


@pytest.fixture
def live_turn(home):
    """Bind the main runtime the way a turn binds it, with the allow-list on."""
    tokens = []

    def bind(**auxiliary):
        home(_config(["hub"], **auxiliary))
        rt = _live_runtime()
        tokens.append(aux.set_runtime_main(
            rt["provider"], rt["model"], requested_provider=rt["requested_provider"],
            base_url=rt["base_url"], api_key=rt["api_key"], api_mode=rt["api_mode"], session_id="s-1"))
        return rt

    yield bind
    for token in tokens:
        aux.reset_runtime_main(token)


# ── configuration ───────────────────────────────────────────────────────────


def test_unset_or_empty_keeps_the_discovery_chain(home):
    for allowed in (None, []):
        home(_config(allowed))
        assert _allowed_labels(_config(allowed)) is None
        assert _labels() == ["openrouter", "nous", "local/custom", "api-key"]
        assert aux_route_allowed("openrouter")


def test_listed_providers_only_and_identity_normalisation(home):
    home(_config(["custom:hub", "codex"]))
    assert _allowed_labels(_config(["custom:hub", "codex"])) == frozenset({"hub", "openai-codex"})
    assert _labels() == ["api-key"]  # the lane stays; it checks each provider it would try
    for name in ("hub", "custom:hub", "fallback_chain[0](hub)", "main-agent(hub)", "openai-codex"):
        assert aux_route_allowed(name), name
    for name in ("openrouter", "nous", "auto"):
        assert not aux_route_allowed(name), name
    # A route that carries its own base_url is "custom", whatever its name says.
    assert not aux_route_allowed("hub", base_url=ELSEWHERE)


def test_route_labels_in_the_list_name_no_provider(home):
    home(_config(["hub", "auto", "api-key"]))
    assert _allowed_labels(_config(["hub", "auto", "api-key"])) == frozenset({"hub"})
    assert not aux_route_allowed("auto") and not aux_route_allowed("api-key")


def test_json_string_value_is_parsed(home):
    assert _allowed_labels(_config('["hub"]')) == frozenset({"hub"})


def test_set_but_unparseable_value_allows_nothing(home):
    home(_config({"hub": True}))
    assert not aux_route_allowed("hub")
    assert _labels() == ["api-key"]
    assert aux._resolve_api_key_provider() == (None, None)


def test_api_key_lane_tries_only_listed_providers(home, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-test-key")
    monkeypatch.setenv("ZAI_API_KEY", "z-test-key")
    home(_config(["zai"]))
    tried, real = [], aux._select_pool_entry
    monkeypatch.setattr(aux, "_select_pool_entry", lambda pid, *a, **k: tried.append(pid) or real(pid, *a, **k))
    client, _model = aux._resolve_api_key_provider()
    assert client is not None and client._hermes_aux_backend_provider == "zai"
    assert tried == ["zai"]


def test_auto_with_its_own_base_url_is_checked_before_resolution(home):
    home(_config(["hub"]))
    require_aux_route_allowed("auto", "compression")  # a bare auto is checked once resolved
    with pytest.raises(AuxiliaryProviderNotAllowed):
        require_aux_route_allowed("auto", "compression", ELSEWHERE)


# ── text routing ────────────────────────────────────────────────────────────


def test_auto_routes_to_the_listed_main_provider(home):
    home(_config(["hub"]))
    client, model, provider = aux._resolve_auto_route(main_runtime={}, task="title_generation")
    assert provider == "hub" and model == "hub-model"
    assert str(client.base_url).rstrip("/") == HUB_URL


def test_auto_never_falls_through_to_discovery_when_main_is_not_listed(home):
    cfg = _config(["hub"])
    cfg["model"] = {"provider": "openrouter", "default": "some/model"}
    home(cfg)
    with patch.object(aux, "_try_openrouter") as try_or, patch.object(aux, "_try_nous") as try_nous:
        assert aux._resolve_auto_route(main_runtime={}, task="compression")[0] is None
    try_or.assert_not_called()
    try_nous.assert_not_called()


def test_main_runtime_on_an_unlisted_provider_is_not_used(home):
    """The main agent may have failed over at runtime; auxiliary calls stay on the list."""
    home(_config(["hub"]))
    client, _model, _provider = aux._resolve_auto_route(
        main_runtime={"provider": "openrouter", "model": "x/y"}, task="compression")
    assert client is None


def test_fallback_entries_outside_the_list_are_never_resolved(home):
    cfg = _config(["hub"], compression={"fallback_chain": [
        {"provider": "openrouter", "model": "a/b"},
        {"provider": "hub", "model": "m", "base_url": ELSEWHERE},
    ]})
    cfg["fallback_providers"] = [{"provider": "openrouter", "model": "a/b"}]
    home(cfg)
    with patch.object(aux, "resolve_provider_client") as rpc:
        assert aux._try_configured_fallback_chain("compression", "hub")[0] is None
        assert aux._try_main_fallback_chain("compression", "hub")[0] is None
    rpc.assert_not_called()


def test_main_agent_fallback_requires_a_listed_main_provider(home):
    cfg = _config(["hub"])
    cfg["model"] = {"provider": "openrouter", "default": "some/model"}
    home(cfg)
    with patch.object(aux, "resolve_provider_client") as rpc:
        assert aux._try_main_agent_model_fallback("hub", task="compression")[0] is None
    rpc.assert_not_called()


def test_main_agent_auto_resolution_stays_on_the_list(home):
    cfg = _config(["hub"])
    cfg["model"] = {"provider": "auto", "default": "x/y"}
    home(cfg)
    assert aux.resolve_provider_client("auto", model="x/y", raw_codex=True)[0] is None
    cfg["auxiliary"]["allowed_providers"] = ["openrouter"]
    home(cfg)
    client, _model = aux.resolve_provider_client("auto", model="x/y", raw_codex=True)
    assert client._hermes_aux_effective_provider == "openrouter"


def test_text_client_getter_refuses_an_unlisted_pin(home):
    home(_config(["hub"], skills_hub={"provider": "openrouter", "model": "a/b"}))
    assert aux.get_text_auxiliary_client("skills_hub") == (None, None)


# ── vision ──────────────────────────────────────────────────────────────────


def test_vision_auto_reaches_aggregators_without_the_list(home):
    """Control: with no allow-list a text-only main model sends vision to OpenRouter."""
    home(_config(None))
    with patch.object(aux, "_main_model_supports_vision", return_value=False):
        provider, client, _model = aux.resolve_vision_provider_client(main_runtime={})
    assert provider == "openrouter" and client is not None


def test_vision_auto_never_reaches_aggregators_with_the_list(home):
    home(_config(["hub"]))
    with patch.object(aux, "_main_model_supports_vision", return_value=False), \
            patch.object(aux, "_resolve_strict_vision_backend") as strict:
        _provider, client, _model = aux.resolve_vision_provider_client(main_runtime={})
        assert aux.get_available_vision_backends() == ["hub"]
    assert client is None
    strict.assert_not_called()


def test_vision_auto_uses_the_listed_main_provider(home):
    home(_config(["hub"]))
    provider, client, _model = aux.resolve_vision_provider_client(main_runtime={})
    assert provider == "hub" and str(client.base_url).rstrip("/") == HUB_URL


def test_explicit_vision_pin_outside_the_list_resolves_nothing(home):
    home(_config(["hub"], vision={"provider": "openrouter"}))
    with patch.object(aux, "_try_openrouter") as try_or:
        provider, client, _model = aux.resolve_vision_provider_client(main_runtime={})
    assert (provider, client) == ("openrouter", None)
    try_or.assert_not_called()


# ── call_llm / async_call_llm ───────────────────────────────────────────────


def test_call_llm_refuses_an_explicit_pin_outside_the_list(home):
    home(_config(["hub"], title_generation={"provider": "openrouter", "model": "a/b"}))
    with patch.object(aux, "OpenAI") as openai_cls, pytest.raises(AuxiliaryProviderNotAllowed):
        aux.call_llm(task="title_generation", messages=MSGS)
    openai_cls.assert_not_called()


def test_call_llm_refuses_a_per_task_base_url(home):
    home(_config(["hub"], compression={"base_url": ELSEWHERE, "api_key": "x"}))
    with patch.object(aux, "OpenAI") as openai_cls, pytest.raises(AuxiliaryProviderNotAllowed):
        aux.call_llm(task="compression", messages=MSGS)
    openai_cls.assert_not_called()


def test_async_call_llm_refuses_an_explicit_vision_pin(home):
    home(_config(["hub"], vision={"provider": "nous"}))
    with pytest.raises(AuxiliaryProviderNotAllowed):
        asyncio.run(aux.async_call_llm(task="vision", messages=MSGS))


def test_a_listed_pin_is_allowed(home, wire):
    seen, _reply = wire
    home(_config(["hub"], title_generation={"provider": "hub", "model": "small-model"}))
    aux.call_llm(task="title_generation", messages=MSGS)
    assert seen and all(u.startswith(HUB_URL + "/") for u in seen), seen


def test_dispatch_check_refuses_a_route_the_filters_missed(home):
    """Defense in depth: if resolution ever hands back an unlisted client, it is not called."""
    home(_config(["hub"]))
    leaked = MagicMock(name="openrouter-client")
    leaked.base_url = "https://openrouter.ai/api/v1"
    leaked._hermes_aux_effective_provider = "openrouter"
    with patch.object(aux, "_get_cached_client", return_value=(leaked, "a/b")), \
            pytest.raises(AuxiliaryProviderNotAllowed):
        aux.call_llm(task="compression", messages=MSGS)
    leaked.chat.completions.create.assert_not_called()


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_dispatch_check_refuses_a_custom_client_off_the_list(home, mode):
    home(_config(["hub"]))
    leaked = MagicMock(name="custom-client")
    leaked.base_url = ELSEWHERE + "/"
    leaked._hermes_aux_effective_provider = "custom"
    with patch.object(aux, "_get_cached_client", return_value=(leaked, "m")), \
            pytest.raises(AuxiliaryProviderNotAllowed):
        if mode == "sync":
            aux.call_llm(task="compression", messages=MSGS)
        else:
            asyncio.run(aux.async_call_llm(task="compression", messages=MSGS))
    leaked.chat.completions.create.assert_not_called()


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_dispatch_check_refuses_a_route_that_names_no_provider(home, mode):
    """A resolved client that carries no provider identity fails closed, even at the listed endpoint."""
    home(_config(["hub"]))
    unnamed = MagicMock(name="unnamed-client", spec=["base_url", "chat"])
    unnamed.base_url = HUB_URL
    with patch.object(aux, "_get_cached_client", return_value=(unnamed, "m")), \
            pytest.raises(AuxiliaryProviderNotAllowed):
        if mode == "sync":
            aux.call_llm(task="compression", messages=MSGS)
        else:
            asyncio.run(aux.async_call_llm(task="compression", messages=MSGS))
    unnamed.chat.completions.create.assert_not_called()


def test_dispatch_check_names_the_provider_behind_the_api_key_lane(home):
    home(_config(["zai"]))
    lane = MagicMock(name="api-key-client")
    lane.base_url = "https://api.example.com/v1"
    lane._hermes_aux_effective_provider = "api-key"
    lane._hermes_aux_backend_provider = "zai"
    assert aux_client_route(lane, "api-key") == "zai"
    require_aux_client_allowed(lane, "api-key", "compression")
    lane._hermes_aux_backend_provider = "gemini"
    with patch.object(aux, "_get_cached_client", return_value=(lane, "m")), \
            pytest.raises(AuxiliaryProviderNotAllowed):
        aux.call_llm(task="compression", messages=MSGS)
    lane.chat.completions.create.assert_not_called()


def test_text_client_getter_checks_an_auto_override_before_resolving(home):
    home(_config(["hub"], compression={"provider": "auto", "base_url": ELSEWHERE, "api_key": "x"}))
    with patch.object(aux, "resolve_provider_client") as rpc:
        assert aux.get_text_auxiliary_client("compression") == (None, None)
    rpc.assert_not_called()


def test_text_client_getter_unchanged_without_the_list(home):
    home(_config(None))
    with patch.object(aux, "resolve_provider_client", return_value=(None, None)):
        assert aux.get_text_auxiliary_client("compression") == (None, None)
    other = MagicMock(name="openrouter-client")
    with patch.object(aux, "resolve_provider_client", return_value=(other, "a/b")):
        assert aux.get_text_auxiliary_client("compression") == (other, "a/b")


def test_text_client_getter_checks_the_resolved_route(home):
    home(_config(["hub"]))
    leaked = MagicMock(name="openrouter-client")
    leaked.base_url = "https://openrouter.ai/api/v1"
    leaked._hermes_aux_effective_provider = "openrouter"
    with patch.object(aux, "resolve_provider_client", return_value=(leaked, "a/b")):
        assert aux.get_text_auxiliary_client("compression") == (None, None)
    listed = MagicMock(name="hub-client")
    listed.base_url = HUB_URL
    listed._hermes_aux_effective_provider = "hub"
    with patch.object(aux, "resolve_provider_client", return_value=(listed, "hub-model")):
        assert aux.get_text_auxiliary_client("compression") == (listed, "hub-model")


def test_dispatch_check_refuses_a_named_entry_client_without_an_endpoint(home):
    home(_config(["hub"]))
    bare = MagicMock(name="hub-client-without-base-url", spec=["chat"])
    with patch.object(aux, "_get_cached_client", return_value=(bare, "m")), \
            patch.object(aux, "_effective_provider_for_client", return_value="hub"), \
            pytest.raises(AuxiliaryProviderNotAllowed):
        aux.call_llm(task="compression", messages=MSGS)
    bare.chat.completions.create.assert_not_called()


# ── a listed route that resolves somewhere else is dropped at every handoff ──

BACKUP_URL = "https://backup.example.com/v1"


def _two_entry_config(**auxiliary):
    cfg = _config(["hub", "backup"], **auxiliary)
    cfg["providers"]["backup"] = {"base_url": BACKUP_URL, "api_key": "backup-key", "api_mode": "chat_completions"}
    return cfg


def _resolver_sending(provider_name, url):
    """The real router, except that ``provider_name`` resolves to a client at ``url``."""
    real = aux.resolve_provider_client

    def resolve(provider, *args, **kwargs):
        if provider == provider_name:
            return aux.OpenAI(api_key="x", base_url=url), "m"
        return real(provider, *args, **kwargs)
    return resolve


def test_listed_fallback_entry_resolving_elsewhere_is_dropped(home):
    home(_two_entry_config(compression={"fallback_chain": [{"provider": "backup", "model": "m"}]}))
    with patch.object(aux, "resolve_provider_client", side_effect=_resolver_sending("backup", ELSEWHERE)):
        assert aux._resolve_fallback_entry({"provider": "backup", "model": "m"}) == (None, None)
        assert aux._try_configured_fallback_chain("compression", "hub")[0] is None
    with patch.object(aux, "resolve_provider_client", side_effect=_resolver_sending("backup", BACKUP_URL)):
        client, _model = aux._resolve_fallback_entry({"provider": "backup", "model": "m"})
    assert str(client.base_url).rstrip("/") == BACKUP_URL


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_recovery_never_calls_a_listed_fallback_that_resolved_elsewhere(home, monkeypatch, mode):
    home(_two_entry_config(compression={"fallback_chain": [{"provider": "backup", "model": "m"}]}))
    seen = []

    def send(self, request, *args, **kwargs):
        seen.append(str(request.url))
        if "hub.example.com" in str(request.url):
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json=_completion("leak"), request=request)

    async def asend(self, request, *args, **kwargs):
        return send(self, request)

    monkeypatch.setattr(httpx.Client, "send", send)
    monkeypatch.setattr(httpx.AsyncClient, "send", asend)
    rt = _live_runtime()
    token = aux.set_runtime_main(rt["provider"], rt["model"], requested_provider=rt["requested_provider"],
                                 base_url=rt["base_url"], api_key=rt["api_key"], api_mode=rt["api_mode"],
                                 session_id="s-1")
    try:
        with patch.object(aux, "resolve_provider_client", side_effect=_resolver_sending("backup", ELSEWHERE)), \
                pytest.raises(openai.APIConnectionError) as excinfo:  # the listed provider's own error
            if mode == "sync":
                aux.call_llm(task="compression", messages=MSGS)
            else:
                asyncio.run(aux.async_call_llm(task="compression", messages=MSGS))
    finally:
        aux.reset_runtime_main(token)
    assert str(excinfo.value.request.url).startswith(HUB_URL + "/")
    assert seen and all(u.startswith(HUB_URL + "/") for u in seen), seen


def test_main_agent_fallback_resolving_elsewhere_is_dropped(home):
    home(_config(["hub"]))
    with patch.object(aux, "resolve_provider_client", side_effect=_resolver_sending("hub", ELSEWHERE)):
        assert aux._try_main_agent_model_fallback("openrouter", task="compression")[0] is None


def test_vision_auto_moves_on_after_a_main_client_that_resolved_elsewhere(home):
    home(_config(["hub", "openrouter"]))
    stray = aux.OpenAI(api_key="x", base_url=ELSEWHERE)
    with patch.object(aux, "_vision_main_provider_client", return_value=(stray, "m")):
        provider, client, _model = aux.resolve_vision_provider_client(main_runtime={})
    assert provider == "openrouter" and "openrouter.ai" in str(client.base_url)


def test_unset_list_never_inspects_the_resolved_client(home):
    class Touchy:
        @property
        def base_url(self):
            raise RuntimeError("inspected")

    home(_config(None))
    touchy = Touchy()
    with patch.object(aux, "resolve_provider_client", return_value=(touchy, "m")):
        assert aux.get_text_auxiliary_client("compression") == (touchy, "m")
        assert aux._resolve_fallback_entry({"provider": "openrouter", "model": "m"})[0] is touchy


def test_direct_vision_caller_gets_no_client_resolving_elsewhere(home):
    home(_config(["hub"], vision={"provider": "hub"}))
    stray = aux.OpenAI(api_key="x", base_url=ELSEWHERE)
    with patch.object(aux, "_get_cached_client", return_value=(stray, "m")):
        provider, client, _model = aux.resolve_vision_provider_client(main_runtime={})
    assert (provider, client) == ("hub", None)


# ── the live runtime of a named provider: "custom" + that entry's base_url ──


def test_the_live_named_provider_runtime_is_custom_plus_the_entry_url(home):
    home(_config(["hub"]))
    rt = _live_runtime()
    assert (rt["provider"], rt["requested_provider"]) == ("custom", "hub")
    assert rt["base_url"].rstrip("/") == HUB_URL


def test_auto_step1_uses_the_live_runtime_of_the_listed_entry(home):
    home(_config(["hub"]))
    client, model, _provider = aux._resolve_auto_route(main_runtime=_live_runtime(), task=None)
    assert client is not None and model == "hub-model"
    assert str(client.base_url).rstrip("/") == HUB_URL


@pytest.mark.parametrize("endpoint,allowed", [
    (HUB_URL, True), (HUB_URL + "/", True),
    ("HTTPS://HUB.EXAMPLE.COM/v1", True),                  # scheme and host are case-insensitive
    ("https://hub.example.com:443/v1", True),              # the default port is the same endpoint
    ("https://hub.example.com/V1", False),                 # the path is not
    ("https://hub.example.com:8443/v1", False),
    ("http://hub.example.com/v1", False),
    ("https://user@hub.example.com/v1", False),
    ("https://hub.example.com@evil.example.com/v1", False),
    ("https://hub.example.com/v1?x=1", False),
    ("https://hub.example.com/v1/extra", False),
    ("https://hub.example.com.evil.com/v1", False),
    ("https://evil.com/https://hub.example.com/v1", False),
    (ELSEWHERE, False), ("", False), (None, False),
])
def test_custom_is_identified_only_by_a_listed_entrys_own_endpoint(home, endpoint, allowed):
    home(_config(["hub"]))
    assert aux_route_allowed("custom", endpoint=endpoint) is allowed


def test_endpoint_identity_never_relabels_a_named_route_or_an_override(home):
    cfg = _config(["hub"])
    cfg["providers"]["other"] = {"base_url": "https://other.example.com/v1", "api_key": "x"}
    home(cfg)
    assert not aux_route_allowed("custom", endpoint="https://other.example.com/v1")  # entry not listed
    assert not aux_route_allowed("hub", base_url=HUB_URL, endpoint=HUB_URL)  # an override is always custom
    assert not aux_route_allowed("openrouter", endpoint=HUB_URL)  # a named route is its name
    home(_config(None))
    assert aux_route_allowed("custom", endpoint=ELSEWHERE)  # no list, no restriction


def test_a_custom_runtime_pointing_elsewhere_is_refused(home):
    home(_config(["hub"]))
    rt = dict(_live_runtime(), base_url=ELSEWHERE)
    with patch.object(aux, "resolve_provider_client") as rpc, patch.object(aux, "_try_openrouter") as try_or:
        assert aux._resolve_auto_route(main_runtime=rt, task="compression")[0] is None
    rpc.assert_not_called()
    try_or.assert_not_called()


def test_vision_step1_refuses_a_custom_runtime_elsewhere(home, wire):
    seen, _reply = wire
    home(_config(["hub"]))
    rt = dict(_live_runtime(), base_url=ELSEWHERE)
    with patch.object(aux, "_main_model_supports_vision", return_value=True):
        assert aux.resolve_vision_provider_client(provider="auto", main_runtime=rt)[1] is None
        client = aux.resolve_vision_provider_client(provider="auto", main_runtime=_live_runtime())[1]
        assert str(client.base_url).rstrip("/") == HUB_URL
    assert not seen


# ── end to end: real calls, the HTTP send answered in-process ──────────────


def test_plugin_llm_call_reaches_the_listed_provider_in_a_turn(live_turn, wire):
    """A plugin's structured call (PluginLlm -> async_call_llm, no task, no provider)."""
    from agent.plugin_llm import PluginLlm

    seen, reply = wire
    live_turn()
    reply["content"] = '{"verdict": "PASS"}'
    result = asyncio.run(PluginLlm(plugin_id="example").acomplete_structured(
        instructions="Review.", input=[{"type": "text", "text": "diff"}], json_mode=True))
    assert result.parsed == {"verdict": "PASS"}
    assert seen and all(u.startswith(HUB_URL + "/") for u in seen), seen


@pytest.mark.parametrize("task,auxiliary,model", [
    ("title_generation", {"title_generation": {"model": "small-model"}}, "small-model"),
    ("title_generation", {"title_generation": {"provider": "hub", "model": "small-model"}}, "small-model"),
    ("compression", {}, "hub-model"),
    (None, {}, "hub-model"),
])
def test_in_turn_auxiliary_tasks_reach_the_listed_provider(live_turn, wire, task, auxiliary, model):
    seen, _reply = wire
    live_turn(**auxiliary)
    sent = []
    orig = httpx.Client.send

    def capture(self, request, *args, **kwargs):
        sent.append(json.loads(request.content or b"{}").get("model"))
        return orig(self, request, *args, **kwargs)

    with patch.object(httpx.Client, "send", capture):
        aux.call_llm(task=task, messages=MSGS)
    assert seen and all(u.startswith(HUB_URL + "/") for u in seen), seen
    assert sent and sent[-1] == model


LISTED_PROVIDER_ERROR = {"conn": openai.APIConnectionError, "429": openai.RateLimitError,
                         "402": openai.APIStatusError, "401": openai.AuthenticationError}


@pytest.mark.parametrize("err", ["conn", "429", "402", "401"])
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("task", [None, "compression"])
def test_failures_never_fall_back_off_the_list(home, monkeypatch, err, mode, task):
    """The listed provider fails while every fallback layer points elsewhere (task fallback_chain, a
    custom base_url entry, top-level fallback_providers, OpenRouter/Nous keys): nothing else is called.
    Without the allow-list the same call is answered by a fallback (see the control below)."""
    cfg = _config(["hub"], compression={"fallback_chain": [
        {"provider": "openrouter", "model": "a/b"},
        {"provider": "custom", "base_url": "https://evil.example.com/v1", "model": "m"}]})
    cfg["fallback_providers"] = [{"provider": "nous", "model": "c/d"}]
    home(cfg)
    monkeypatch.setenv("NOUS_API_KEY", "nous-test-key")
    seen = []

    def send(self, request, *args, **kwargs):
        seen.append(str(request.url))
        if "hub.example.com" in str(request.url):
            if err == "conn":
                raise httpx.ConnectError("refused", request=request)
            return httpx.Response(int(err), json={"error": {"message": "nope", "type": "x"}}, request=request)
        return httpx.Response(200, json=_completion("leak"), request=request)

    async def asend(self, request, *args, **kwargs):
        return send(self, request)

    monkeypatch.setattr(httpx.Client, "send", send)
    monkeypatch.setattr(httpx.AsyncClient, "send", asend)
    rt = _live_runtime()
    token = aux.set_runtime_main(rt["provider"], rt["model"], requested_provider=rt["requested_provider"],
                                 base_url=rt["base_url"], api_key=rt["api_key"], api_mode=rt["api_mode"],
                                 session_id="s-1")
    try:
        with pytest.raises(LISTED_PROVIDER_ERROR[err]) as excinfo:  # the listed provider's own error
            if mode == "sync":
                aux.call_llm(task=task, messages=MSGS)
            else:
                asyncio.run(aux.async_call_llm(task=task, messages=MSGS))
    finally:
        aux.reset_runtime_main(token)
    assert str(excinfo.value.request.url).startswith(HUB_URL + "/")
    assert err == "conn" or excinfo.value.status_code == int(err)
    assert seen and all(u.startswith(HUB_URL + "/") for u in seen), seen


def test_control_without_the_list_the_same_failure_falls_back(home, monkeypatch):
    """Control for the test above: with the allow-list unset, a fallback outside it answers."""
    cfg = _config(None, compression={"fallback_chain": [{"provider": "openrouter", "model": "a/b"}]})
    home(cfg)
    seen = []

    def send(self, request, *args, **kwargs):
        seen.append(str(request.url))
        if "hub.example.com" in str(request.url):
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json=_completion("fallback"), request=request)

    monkeypatch.setattr(httpx.Client, "send", send)
    rt = _live_runtime()
    token = aux.set_runtime_main(rt["provider"], rt["model"], requested_provider=rt["requested_provider"],
                                 base_url=rt["base_url"], api_key=rt["api_key"], api_mode=rt["api_mode"],
                                 session_id="s-1")
    try:
        aux.call_llm(task="compression", messages=MSGS)
    finally:
        aux.reset_runtime_main(token)
    assert any(not u.startswith(HUB_URL + "/") for u in seen), seen


@pytest.mark.parametrize("pin", [{"provider": "custom", "base_url": HUB_URL, "api_key": "x"},
                                 {"provider": "hub", "base_url": HUB_URL + "/", "api_key": "x"}])
def test_a_pin_with_its_own_base_url_is_refused_even_at_the_listed_endpoint(live_turn, wire, pin):
    seen, _reply = wire
    live_turn(compression=pin)
    with pytest.raises(AuxiliaryProviderNotAllowed):
        aux.call_llm(task="compression", messages=MSGS)
    assert not seen


@pytest.mark.parametrize("base_url", [ELSEWHERE, HUB_URL])
def test_a_caller_base_url_is_refused_in_a_turn(live_turn, wire, base_url):
    seen, _reply = wire
    live_turn()
    with pytest.raises(AuxiliaryProviderNotAllowed):
        aux.call_llm(task=None, base_url=base_url, messages=MSGS)
    assert not seen


def test_a_named_runtime_with_a_foreign_url_never_calls_it(home, wire):
    seen, _reply = wire
    home(_config(["hub"]))
    rt = dict(_live_runtime(), provider="hub", base_url=ELSEWHERE)
    with contextlib.suppress(Exception):
        aux.call_llm(task="compression", main_runtime=rt, messages=MSGS)
    assert all(u.startswith(HUB_URL + "/") for u in seen), seen
    assert not aux_route_allowed("hub", endpoint=ELSEWHERE)  # a listed entry only at its own endpoint
    assert aux_route_allowed("hub", endpoint=HUB_URL + "/")
