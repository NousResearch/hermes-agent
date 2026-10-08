"""Refreshed Nous wire mode, router shims, and caller-pinned pool credentials."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent import auxiliary_client as aux


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("wire", ["native", "chat"])
def test_nous_refresh_keeps_configured_wire(monkeypatch, async_mode, wire):
    from hermes_cli import config
    from agent import anthropic_adapter

    monkeypatch.setattr(config, "load_config_readonly", lambda: {"nous": {"anthropic_wire": wire}})
    monkeypatch.setattr(aux, "_resolve_nous_runtime_api", lambda **kw: ("fresh-fake", "https://nous.invalid/v1"))
    plain = SimpleNamespace(api_key="fresh-fake", base_url="https://nous.invalid/v1")
    monkeypatch.setattr(aux, "_create_openai_client", lambda **kw: plain)
    monkeypatch.setattr(anthropic_adapter, "build_anthropic_client", lambda *a: plain)
    monkeypatch.setattr(aux, "_store_cached_client", lambda *a, **kw: None)
    monkeypatch.setattr(aux, "_current_event_loop", lambda: None)
    # Observe the sync transport passed into async conversion without constructing HTTP clients.
    monkeypatch.setattr(aux, "_to_async_client", lambda client, model, **kw: (client, model))
    client, model = aux._refresh_nous_auxiliary_client(
        cache_provider="nous", model="anthropic/fake", async_mode=async_mode)
    assert isinstance(client, aux.AnthropicAuxiliaryClient) == (wire == "native")
    assert model == "anthropic/fake"


def test_router_timeout_shim_enters_provider_fallback(monkeypatch):
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content="Connect timeout, please try again later.", tool_calls=None))], usage=None)
    with pytest.raises(RuntimeError) as caught:
        aux._validate_llm_response(response, "title_generation")
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda task: {})
    monkeypatch.setattr(aux, "_try_configured_fallback_chain", lambda *a, **kw: (
        SimpleNamespace(), "fallback-model", "fallback_chain[0](openrouter)"))
    route = aux._LadderRoute(
        SimpleNamespace(), "title_generation", "", False, "https://router.invalid",
        "custom", "fake", None, None, None, "fake", None, None, 30)
    ladder = aux._ladder_provider_fallback(caught.value, route)
    step = next(ladder, None)
    assert step is not None and step.kind == "fallback"
    ladder.close()


@pytest.mark.parametrize("pinned", [None, "caller-pinned-fake"])
def test_explicit_key_prevents_pool_rotation(monkeypatch, pinned):
    class PaymentError(Exception):
        status_code = 402

    monkeypatch.setattr(aux, "_auth_refresh_provider_for_route", lambda *a: "openrouter")
    monkeypatch.setattr(aux, "_recoverable_pool_provider", lambda *a, **kw: "openrouter")
    rotate = Mock(return_value=True)
    monkeypatch.setattr(aux, "_recover_provider_pool", rotate)
    route = aux._LadderRoute(
        SimpleNamespace(api_key=pinned or "pool-fake"), "title_generation", "", False,
        "https://router.invalid", "openrouter", "fake", None, pinned, None,
        "fake", None, None, 30)
    ladder = aux._ladder_credential_rungs(PaymentError("payment required"), route, {}, False)
    step = next(ladder, None)
    assert rotate.call_count == (0 if pinned else 1)
    assert (step is None) == bool(pinned)
    ladder.close()
