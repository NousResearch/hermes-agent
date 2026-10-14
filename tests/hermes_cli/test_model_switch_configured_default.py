"""A declared default absent from discovery remains selectable on its own route (#138261)."""

import json
import secrets
from unittest.mock import Mock

import pytest

from hermes_cli import model_switch as switch
from hermes_cli.config import get_config_path
from hermes_cli.models_validate import validate_requested_model

ROUTER = "accounts/fireworks/routers/deepseek-v4p1-flash-us"
ENDPOINT = "https://api.fireworks.ai/inference/v1"
# Placeholder credential for the mocked-discovery path; generated, never a real secret.
API_KEY = secrets.token_hex(8)


@pytest.fixture
def listing(monkeypatch):
    # Only discovery and metadata I/O are replaced; validation and config loading are real.
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda *a, **kw: {})
    monkeypatch.setattr("hermes_cli.models_validate._profile_catalog", lambda *a: ([], False))
    monkeypatch.setattr("hermes_cli.models._model_in_provider_catalog", lambda *a: False)
    monkeypatch.setattr("hermes_cli.model_catalog.get_catalog", lambda *a, **kw: {})
    monkeypatch.setattr(switch, "get_model_info", lambda *a, **kw: None)
    monkeypatch.setattr(switch, "get_model_capabilities", lambda *a, **kw: None)
    fetch = Mock(return_value=["accounts/fireworks/models/deepseek-v4p1-flash"])
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", fetch)
    return fetch


def _config(data):
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize("config", [
    {"model": {"default": ROUTER, "provider": "fireworks", "base_url": ENDPOINT}},
    {"model": {"default": ROUTER, "provider": "fireworks"}},
    {"model": {"default": ROUTER, "provider": "fw", "base_url": ENDPOINT + "/"}},
    {"model": {"name": ROUTER, "provider": "fireworks", "api_base": ENDPOINT}},
    {"model": {"default": {"model": ROUTER, "provider": "fireworks"}, "base_url": ENDPOINT}},
    {"model": {"default": ROUTER, "provider": "fireworks", "base_url": "HTTPS://API.FIREWORKS.AI/inference/v1"}},
    {"model": {"default": "router", "provider": "openrouter", "base_url": "https://old.example/v1"},
     "model_aliases": {"router": {"model": ROUTER, "provider": "fireworks", "base_url": ENDPOINT}}},
    {"model": {"default": "router"},
     "model_aliases": {"router": {"model": ROUTER, "provider": "fireworks", "base_url": ENDPOINT}}},
    {"model": {"default": "router", "provider": "auto"},
     "model_aliases": {"router": {"model": ROUTER, "provider": "fireworks", "base_url": ENDPOINT}}},
    {"model": {"default": "router", "provider": "fw", "base_url": ENDPOINT,
               "aliases": {"router": {"model": ROUTER, "provider": "fireworks"}}}},
])
def test_configured_default_can_be_reselected_after_live_listing_miss(config, listing, monkeypatch):
    _config(config)
    rejected = validate_requested_model(ROUTER, "fireworks", api_key=API_KEY, base_url=ENDPOINT)
    assert not rejected["accepted"]
    assert "not found" in rejected["message"]
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **kw: {
        "provider": "fireworks", "base_url": ENDPOINT, "api_key": API_KEY,
        "api_mode": "chat_completions",
    })
    result = switch.switch_model(ROUTER, "fireworks", ROUTER, ENDPOINT, API_KEY,
                                 explicit_provider="fireworks")
    assert result.success, result.error_message
    assert (result.new_model, result.target_provider, result.base_url) == (ROUTER, "fireworks", ENDPOINT)
    assert result.warning_message == rejected["message"]
    assert listing.called


@pytest.mark.parametrize("model_cfg,requested,endpoint", [
    ({"default": ROUTER, "provider": "fireworks", "base_url": ENDPOINT}, ROUTER + "-typo", ENDPOINT),
    ({"default": ROUTER, "provider": "openrouter", "base_url": ENDPOINT}, ROUTER, ENDPOINT),
    ({"default": ROUTER, "provider": "fireworks", "base_url": ENDPOINT}, ROUTER, "https://us.api.fireworks.ai/inference/v1"),
    ({"default": ROUTER, "provider": "fireworks", "base_url": ENDPOINT}, ROUTER, ENDPOINT + "/other"),
    ({"default": ROUTER, "provider": "fireworks", "base_url": ENDPOINT}, ROUTER, "http://api.fireworks.ai/inference/v1"),
    ({"default": ROUTER, "provider": "fireworks", "base_url": ENDPOINT}, ROUTER, "https://api.fireworks.ai:8443/inference/v1"),
    ({"default": ROUTER, "provider": "fireworks", "base_url": "https://relay.example/TeamA/v1"}, ROUTER, "https://relay.example/teama/v1"),
    ({"default": ROUTER, "provider": "fireworks", "base_url": "https://relay.example/v1?tenant=A"}, ROUTER, "https://relay.example/v1?tenant=a"),
    ({"default": ROUTER, "provider": "fireworks", "base_url": "https://relay.example/v1?tenant=A/"}, ROUTER, "https://relay.example/v1?tenant=A"),
    ({"default": ROUTER, "provider": "fireworks", "base_url": "https://[broken/v1"}, ROUTER, ENDPOINT),
    ({"default": ROUTER, "provider": "fireworks", "base_url": ENDPOINT}, ROUTER, "https://[broken/v1"),
    ({"default": ROUTER, "provider": "fireworks", "base_url": "https://[broken/v1"}, ROUTER, "https://[also-broken/v1"),
    ({"default": ROUTER, "provider": "fireworks"}, ROUTER, "https://other.example/v1"),
    ({"default": ROUTER}, ROUTER, ENDPOINT),
    ({"default": ROUTER, "provider": "auto"}, ROUTER, ENDPOINT),
    ({"provider": "fireworks"}, ROUTER, ENDPOINT),
    (ROUTER, ROUTER, ENDPOINT),
])
def test_default_declaration_does_not_accept_another_model_or_route(model_cfg, requested, endpoint, listing):
    _config({"model": model_cfg})
    st = switch._Switch("", "fireworks", "old", "", "", False, "", {}, [],
                        new_model=requested, target_provider="fireworks", base_url=endpoint, api_key=API_KEY)
    failed = switch._validate_switch(st)
    assert failed is not None and not failed.success
    assert "not found" in failed.error_message


@pytest.mark.parametrize("provider,endpoint,providers,custom,expected", [
    ("relay", ENDPOINT, {"relay": {"base_url": ENDPOINT}}, [], True),
    ("other", ENDPOINT, {"relay": {"base_url": ENDPOINT}, "other": {"base_url": ENDPOINT}}, [], False),
    ("custom:relay", ENDPOINT, {}, [{"name": "relay", "base_url": ENDPOINT}], True),
    ("custom:other", ENDPOINT, {}, [{"name": "relay", "base_url": ENDPOINT},
                                 {"name": "other", "base_url": ENDPOINT}], False),
])
def test_named_provider_default_keeps_its_owner_even_on_shared_endpoint(provider, endpoint, providers, custom, expected, listing):
    owner = "relay"
    _config({"model": {"default": ROUTER, "provider": owner},
             "providers": providers, "custom_providers": custom})
    assert switch._config_declares_model(ROUTER, provider, endpoint, providers, custom) is expected


@pytest.mark.parametrize("providers", [{}, {"custom": {"timeout": 20}}])
def test_bare_custom_default_does_not_borrow_first_named_endpoint(providers, listing):
    custom = [{"name": "relay", "base_url": ENDPOINT}]
    _config({"model": {"default": ROUTER, "provider": "custom"}, "custom_providers": custom,
             "providers": providers})
    assert not switch._config_declares_model(ROUTER, "custom", ENDPOINT, providers, custom)


def test_default_declaration_is_profile_scoped_and_unreadable_config_fails_closed(tmp_path, monkeypatch, listing):
    for profile, declared in [("first", True), ("second", False)]:
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / profile))
        _config({"model": {"default": ROUTER if declared else "another-router", "provider": "fireworks"}})
        assert switch._config_declares_model(ROUTER, "fireworks", ENDPOINT, {}, []) is declared
    # Even after caching a matching default, a broken file cannot authorize a new override.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "first"))
    get_config_path().write_text("model: [", encoding="utf-8")
    assert not switch._config_declares_model(ROUTER, "fireworks", ENDPOINT, {}, [])
