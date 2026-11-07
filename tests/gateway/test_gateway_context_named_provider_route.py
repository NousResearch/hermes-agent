"""A session /model-switched to ``providers.<name>`` keeps that provider's context window when its
persisted route carries only the provider name (no base_url) — /status and the reset banner must not
look the window up against the profile's default endpoint and its context pin."""
from types import SimpleNamespace

import pytest

import gateway.run as gateway_run
from gateway.run_turn import GatewayTurnMixin

LOCAL = "http://127.0.0.1:8031/v1"
PROXY = "http://proxy.example:8080/v1"
WINDOWS = {LOCAL: 1_048_576, PROXY: 524_288}


@pytest.fixture
def routes(monkeypatch):
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {
        "model": {"default": "proxy-model", "provider": "custom", "base_url": PROXY}})
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs",
                        lambda: {"provider": "custom", "base_url": PROXY, "api_key": "p"})
    seen = []

    def for_provider(provider, target_model=None):
        seen.append(provider)
        if provider == "local-ds4":
            return {"provider": "custom", "base_url": LOCAL, "api_key": "l"}
        raise RuntimeError("unknown provider")
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs_for_provider", for_provider)
    import agent.model_metadata as mm
    monkeypatch.setattr(mm, "get_model_context_length",
                        lambda model, base_url="", **_kw: WINDOWS.get(base_url, 256_000))
    return seen


def test_provider_only_route_resolves_named_provider_endpoint(routes):
    ctx = gateway_run._resolve_gateway_model_context("local-model", {"provider": "local-ds4"})
    assert (ctx.base_url, ctx.context_length, ctx.provider) == (LOCAL, 1_048_576, "local-ds4")
    assert routes == ["local-ds4"]


def test_route_with_base_url_is_unchanged(routes):
    ctx = gateway_run._resolve_gateway_model_context("local-model", {"provider": "local-ds4", "base_url": LOCAL})
    assert ctx.context_length == 1_048_576 and routes == []


@pytest.mark.parametrize("route", [None, {"provider": "custom"}, {"provider": "auto"}, {"provider": ""}])
def test_default_and_generic_routes_keep_default_runtime(routes, route):
    ctx = gateway_run._resolve_gateway_model_context(None, route)
    assert (ctx.base_url, ctx.context_length) == (PROXY, 524_288) and routes == []


def test_unresolvable_named_provider_falls_back_to_default_runtime(routes):
    ctx = gateway_run._resolve_gateway_model_context("x", {"provider": "gone"})
    assert ctx.base_url == PROXY and routes == ["gone"]


def test_reset_banner_reports_the_sessions_switched_model(routes, monkeypatch):
    runner = SimpleNamespace(
        _session_key_for_source=lambda src: "k",
        _session_model_override=lambda key: {"model": "local-model", "provider": "local-ds4", "base_url": None},
    )
    captured = {}

    def fake_resolve(model=None, route=None):
        captured.update(model=model, route=route)
        return gateway_run._GatewayModelContext(model=model or "proxy-model", provider="local-ds4",
                                                base_url=LOCAL, context_length=1_048_576, context_source="config")
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model_context", fake_resolve)
    GatewayTurnMixin._format_session_info(runner, runner._session_model_override("k"))
    assert captured["model"] == "local-model" and captured["route"]["provider"] == "local-ds4"
