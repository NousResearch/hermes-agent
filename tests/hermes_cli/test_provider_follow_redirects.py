"""Redirect policy is tri-state, model-specific, and scoped to the active provider route."""

import copy
import logging

import pytest

from hermes_cli.config import atomic_config_replace, get_compatible_custom_providers, load_config_readonly
from hermes_cli.timeouts import get_provider_follow_redirects


@pytest.fixture
def configure(tmp_path, monkeypatch):
    """Exercise the real readonly loader against an isolated profile, without loader mocks."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def write(config):
        atomic_config_replace(tmp_path / "config.yaml", config)

    return write


@pytest.mark.parametrize("value, expected", [
    (True, True), (False, False),
    (" TRUE ", True), ("yes", True), ("on", True), ("1", True),
    (" FALSE ", False), ("no", False), ("off", False), ("0", False),
    (None, None), ("", None), ("garbage", None),
    (1, None), (0, None), (2, None), (-1, None),
    ([], None), ({"enabled": True}, None),
])
def test_only_explicit_boolean_values_change_policy(configure, value, expected):
    configure({"providers": {"redirect-provider": {"follow_redirects": value}}})
    assert get_provider_follow_redirects("redirect-provider") is expected


def test_shared_endpoint_uses_configured_identity_not_first_entry(configure):
    url = "https://proxy.example/v1"
    configure({"providers": {
        "first": {"base_url": url, "follow_redirects": True},
        "second": {"base_url": url, "name": "Second Proxy", "follow_redirects": False},
    }})
    assert get_provider_follow_redirects("custom", base_url=url, requested_provider="custom:second") is False
    assert get_provider_follow_redirects("custom", base_url=url, requested_provider="custom:first") is True
    assert get_provider_follow_redirects("custom", base_url=url, requested_provider="custom:unknown") is None


@pytest.mark.parametrize("providers", [{}, None, [], ["not-a-provider"], {"redirect-provider": []}])
def test_absent_or_malformed_provider_returns_unset(configure, providers):
    configure({"providers": providers})
    assert get_provider_follow_redirects("redirect-provider") is None
    assert get_provider_follow_redirects("") is None


def test_model_override_precedence_and_exact_model_identity(configure):
    configure({"providers": {"redirect-provider": {
        "follow_redirects": True,
        "models": {
            "disabled": {"follow_redirects": False},
            "invalid": {"follow_redirects": "garbage"},
            "unset": {},
            "bad-shape": [],
        },
    }}})
    assert get_provider_follow_redirects("redirect-provider", "disabled") is False
    assert get_provider_follow_redirects("redirect-provider", "invalid") is None
    for model in (None, "", "unknown", "unset", "bad-shape", "Disabled"):
        assert get_provider_follow_redirects("redirect-provider", model) is True


def test_model_can_enable_redirects_when_provider_disables_them(configure):
    configure({"providers": {"redirect-provider": {
        "follow_redirects": False,
        "models": {"enabled": {"follow_redirects": True}},
    }}})
    assert get_provider_follow_redirects("redirect-provider", "enabled") is True
    assert get_provider_follow_redirects("redirect-provider", "other") is False


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("models", [
    {"disabled": {"follow_redirects": False}},
    [{"id": "disabled", "follow_redirects": False}],
])
def test_named_custom_identity_and_normalized_model_lists(configure, legacy, models):
    entry = {
        "name": "Display Endpoint", "base_url": "https://redirect.example/v1",
        "follow_redirects": True, "models": models,
    }
    configure({"custom_providers": [entry]} if legacy else {"providers": {"durable-key": entry}})
    identity = "custom:display-endpoint" if legacy else "custom:durable-key"
    assert get_provider_follow_redirects(identity) is True
    assert get_provider_follow_redirects("custom", requested_provider=identity) is True
    assert get_provider_follow_redirects("Display Endpoint") is True
    assert get_provider_follow_redirects(identity, "disabled") is False
    assert get_provider_follow_redirects("custom", "disabled", base_url="https://redirect.example/v1/") is False


@pytest.mark.parametrize("runtime_provider", ["custom", "custom:old-route", "anthropic"])
def test_current_route_trumps_stale_requested_identity(configure, runtime_provider):
    configure({"providers": {
        "old-route": {"base_url": "https://old.example/v1", "follow_redirects": True},
        "new-route": {"base_url": "https://new.example/v1", "follow_redirects": False,
                      "models": {"enabled": {"follow_redirects": True}}},
    }})
    kwargs = {"base_url": "https://NEW.example:443/v1/", "requested_provider": "custom:old-route"}
    assert get_provider_follow_redirects(runtime_provider, **kwargs) is False
    assert get_provider_follow_redirects(runtime_provider, "enabled", **kwargs) is True


@pytest.mark.parametrize("runtime_provider", ["custom", "custom:old-route"])
def test_unknown_rotated_custom_route_does_not_inherit_opt_in(configure, runtime_provider):
    configure({"providers": {"old-route": {
        "base_url": "https://old.example/v1", "follow_redirects": True,
    }}})
    assert get_provider_follow_redirects(
        runtime_provider, base_url="https://unknown.example/v1", requested_provider="custom:old-route",
    ) is None
    # Path identity is case-sensitive: sharing a host does not share a redirect policy.
    assert get_provider_follow_redirects(
        runtime_provider, base_url="https://old.example/V1", requested_provider="custom:old-route",
    ) is None


def test_builtin_policy_does_not_follow_stale_requested_provider(configure):
    configure({"providers": {"anthropic": {"follow_redirects": False}, "old-route": {"follow_redirects": True}}})
    assert get_provider_follow_redirects(
        "anthropic", base_url="https://api.anthropic.com", requested_provider="old-route",
    ) is False


@pytest.mark.parametrize("value", [True, False, "garbage"])
def test_key_survives_normalization_without_warning_or_cache_mutation(configure, caplog, value):
    configure({"providers": {"durable-key": {
        "name": "Mutable Display", "base_url": "https://redirect.example/v1", "follow_redirects": value,
    }}})
    loaded = load_config_readonly()
    before = copy.deepcopy(loaded)
    with caplog.at_level(logging.WARNING, logger="hermes_cli.config"):
        entries = get_compatible_custom_providers(loaded)
        get_provider_follow_redirects("custom:durable-key", base_url="https://redirect.example/v1")
    assert entries[0]["provider_key"] == "durable-key"
    assert entries[0]["follow_redirects"] == value
    assert not [record for record in caplog.records if "unknown config keys" in record.message]
    assert loaded == before


def test_numeric_provider_key_uses_shared_string_identity(configure):
    configure({"providers": {2070: {"follow_redirects": False}}})
    assert get_provider_follow_redirects("2070") is False


@pytest.mark.parametrize("models", [
    {"disabled": {"follow_redirects": False}},
    [{"name": "disabled", "follow_redirects": False}],
])
def test_builtin_model_overrides_accept_supported_model_shapes(configure, models):
    configure({"providers": {"anthropic": {"follow_redirects": True, "models": models}}})
    assert get_provider_follow_redirects("anthropic", "disabled") is False


@pytest.mark.parametrize("value", [True, False, "garbage"])
def test_legacy_provider_translation_preserves_redirect_policy(value):
    from hermes_cli.config import _custom_provider_entry_to_provider_config

    result = _custom_provider_entry_to_provider_config({
        "name": "Legacy Endpoint", "base_url": "https://legacy.example/v1", "follow_redirects": value,
    })
    assert result is not None
    assert result["follow_redirects"] == value


def test_unreadable_config_does_not_reuse_last_known_good_opt_in(configure, tmp_path):
    configure({"providers": {"redirect-provider": {"follow_redirects": True}}})
    assert get_provider_follow_redirects("redirect-provider") is True
    # Simulate a broken user edit, not a config-writer operation.
    (tmp_path / "config.yaml").write_text("providers: [\n", encoding="utf-8")
    assert get_provider_follow_redirects("redirect-provider") is None
