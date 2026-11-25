"""An explicit API provider must not depend on default-provider credentials (#159980)."""

from typing import Any
import json
from pathlib import Path

import pytest

from gateway.platforms.api_server import _ProviderAuthResolutionError
from tests.gateway.test_api_server import _make_routing_adapter, _patch_create_agent_runtime


@pytest.mark.parametrize("selection", ["request", "lock", "route", "override"])
def test_selected_provider_does_not_resolve_default_auth(monkeypatch, selection):
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    _patch_create_agent_runtime(monkeypatch, captured, FakeAgent)

    def unavailable_default():
        pytest.fail("the unselected default provider must not resolve credentials")

    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs", unavailable_default)
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {
        "model": {"provider": "anthropic", "default": "default-model"}})
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs_for_provider", lambda provider, target_model=None: {
        "provider": provider, "api_key": "selected-only", "base_url": "http://127.0.0.1/v1",
        "api_mode": "chat_completions"})
    adapter = _make_routing_adapter({})
    monkeypatch.setattr(adapter, "_ensure_store", lambda: None)
    override = {"provider": "custom:selected", "model": "selected-model"} if selection == "override" else None
    monkeypatch.setattr(adapter, "_session_model_override_for", lambda *_: override)
    kwargs: dict[str, Any] = {"session_id": "selected-session"}
    if selection in {"request", "lock"}:
        kwargs.update(requested_provider="custom:selected", requested_model="selected-model",
                      confirmed_runtime_lock=selection == "lock")
    elif selection == "route":
        kwargs["route"] = {"provider": "custom:selected", "model": "selected-model"}

    adapter._create_agent(**kwargs)

    assert captured["provider"] == "custom:selected"
    assert captured["model"] == "selected-model"
    assert captured["api_key"] == "selected-only"


def test_real_selected_provider_resolves_with_unconfigured_default(tmp_path, monkeypatch):
    """Resolve real profile config and credentials, replacing only the final agent constructor."""
    import gateway.run as gateway_run
    from hermes_cli.runtime_provider import resolve_runtime_provider

    home = tmp_path / "profile"
    home.mkdir()
    config = {
        "model": {"provider": "anthropic", "default": "default-model"},
        "providers": {"selected": {
            "base_url": "http://127.0.0.1:18991/v1", "api_key": "selected-only",
            "model": "selected-model"}}}
    (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Confirm the independent default is genuinely unavailable in this isolated home.
    with pytest.raises(RuntimeError):
        gateway_run._resolve_runtime_agent_kwargs()
    selected = resolve_runtime_provider(requested="custom:selected", target_model="selected-model")
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
    monkeypatch.setattr("gateway.run.GatewayRunner._load_reasoning_config", staticmethod(lambda model="": {}))
    monkeypatch.setattr("gateway.run.GatewayRunner._load_fallback_model", staticmethod(lambda: None))
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_: set())
    adapter = _make_routing_adapter({})
    monkeypatch.setattr(adapter, "_ensure_store", lambda: None)
    monkeypatch.setattr(adapter, "_session_model_override_for", lambda *_: None)

    adapter._create_agent(session_id="real-profile", requested_provider="custom:selected",
                          requested_model="selected-model")

    assert captured["model"] == "selected-model"
    assert captured["api_key"] == selected["api_key"] == "selected-only"
    assert captured["base_url"] == selected["base_url"]


@pytest.mark.parametrize("selection", ["request", "lock", "route", "override"])
def test_selected_provider_auth_failure_never_uses_default(monkeypatch, selection):
    from types import SimpleNamespace

    _patch_create_agent_runtime(monkeypatch, {}, lambda **kwargs: SimpleNamespace())
    adapter = _make_routing_adapter({})
    monkeypatch.setattr(adapter, "_ensure_store", lambda: None)
    override = {"provider": "custom:selected", "model": "selected-model"} if selection == "override" else None
    monkeypatch.setattr(adapter, "_session_model_override_for", lambda *_: override)

    def unavailable_selected(*args, **kwargs):
        raise RuntimeError("selected credentials unavailable")

    monkeypatch.setattr("gateway.platforms.api_server._resolve_request_runtime_agent_kwargs", unavailable_selected)
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs_for_provider", unavailable_selected)
    kwargs: dict[str, Any] = {"session_id": "selected-session"}
    if selection in {"request", "lock"}:
        kwargs.update(requested_provider="custom:selected", requested_model="selected-model",
                      confirmed_runtime_lock=selection == "lock")
    elif selection == "route":
        kwargs["route"] = {"provider": "custom:selected", "model": "selected-model"}

    with pytest.raises(_ProviderAuthResolutionError, match="selected credentials unavailable"):
        adapter._create_agent(**kwargs)


def test_unselected_turn_preserves_default_auth_failure(monkeypatch):
    _patch_create_agent_runtime(monkeypatch, {}, lambda **kwargs: None)

    def unavailable_default():
        raise RuntimeError("default credentials unavailable")

    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs", unavailable_default)
    adapter = _make_routing_adapter({})
    monkeypatch.setattr(adapter, "_session_model_override_for", lambda *_: None)

    with pytest.raises(_ProviderAuthResolutionError, match="default credentials unavailable"):
        adapter._create_agent(session_id="default-session")
