"""A credentialless bare ``custom`` request must ride the named entry serving its model.

Regression for #145614: a bare ``model.provider: custom`` pick with no
``base_url``/``api_key`` of its own fell through the whole ladder and raised
"provider 'custom' resolved without credentials" even though a ``providers:``
entry serving the session's model declared the credential. The resolution seam
was the one place the bare class was never healed to the named identity — the
route-key and runtime-metadata seams already heal it.
"""

import pytest

from hermes_cli.auth_constants import AuthError
from hermes_cli.runtime_provider import resolve_runtime_provider


PROVIDER_CONFIG = {
    "model": {"default": "zai-org/GLM-5.3-Flash", "provider": "custom"},
    "providers": {
        "my-provider": {
            "name": "MyProvider",
            "base_url": "https://api.console.example.test/v1",
            "model": "zai-org/GLM-5.3-Flash",
            "key_env": "HERMES_TEST_MYPROVIDER_KEY",
        }
    },
}


def test_bare_custom_rides_the_serving_named_entry(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: PROVIDER_CONFIG)
    monkeypatch.setenv("HERMES_TEST_MYPROVIDER_KEY", "test-key-123")

    runtime = resolve_runtime_provider(requested="custom", target_model="zai-org/GLM-5.3-Flash")

    assert runtime.get("api_key") == "test-key-123"
    assert runtime.get("base_url") == "https://api.console.example.test/v1"


def test_bare_custom_refusal_stands_when_no_entry_serves_the_model(monkeypatch):
    """A genuinely keyless bare ``custom`` setup keeps the refusal: no named
    entry serves the model, so there is nothing to heal onto."""
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
        "model": {"default": "some-unknown-model", "provider": "custom"},
    })

    with pytest.raises(AuthError, match="resolved without credentials"):
        resolve_runtime_provider(requested="custom", target_model="some-unknown-model")


LAN_URL = "http://192.168.0.155:8080/v1"


def _lan_config(model_cfg=None, **entry):
    return {
        "model": {"default": "lan-model", "provider": "lan", **(model_cfg or {})},
        "providers": {"lan": {"name": "lan", "base_url": LAN_URL, "models": ["lan-model"], **entry}},
    }


def test_bare_custom_never_rides_an_entry_for_its_url_alone(monkeypatch):
    """Sharing ``model.base_url`` is not serving the model: healing here would send a model the
    entry never declared, plus its key, to a server that may not host it."""
    monkeypatch.setattr("hermes_cli.config.load_config",
                        lambda: _lan_config({"base_url": LAN_URL}, key_env="HERMES_TEST_LAN_KEY"))
    monkeypatch.setenv("HERMES_TEST_LAN_KEY", "test-key-123")

    with pytest.raises(AuthError, match="resolved without credentials"):
        resolve_runtime_provider(requested="custom", target_model="staged-local-model")


def test_bare_custom_refusal_stands_when_the_entry_key_is_missing(monkeypatch):
    """An entry that names a credential which resolves to nothing is keyless by accident, and the
    ``no-key-required`` placeholder must not pass for a key."""
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: _lan_config(key_env="HERMES_TEST_LAN_KEY"))
    monkeypatch.delenv("HERMES_TEST_LAN_KEY", raising=False)

    with pytest.raises(AuthError, match="resolved without credentials"):
        resolve_runtime_provider(requested="custom", target_model="lan-model")


def test_bare_custom_rides_a_deliberately_keyless_entry(monkeypatch):
    """An entry naming no credential is a keyless local server: the placeholder is its right key."""
    monkeypatch.setattr("hermes_cli.config.load_config", _lan_config)

    runtime = resolve_runtime_provider(requested="custom", target_model="lan-model")

    assert runtime.get("base_url") == LAN_URL
    assert runtime.get("api_key") == "no-key-required"


def test_bare_custom_prefers_the_serving_entry_on_the_configured_endpoint(monkeypatch):
    config = _lan_config({"base_url": "http://other.test/v1"}, key_env="HERMES_TEST_LAN_KEY")
    config["providers"]["other"] = {"name": "other", "base_url": "http://other.test/v1",
                                    "models": ["lan-model"], "key_env": "HERMES_TEST_OTHER_KEY"}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    monkeypatch.setenv("HERMES_TEST_LAN_KEY", "lan-key-123")
    monkeypatch.setenv("HERMES_TEST_OTHER_KEY", "other-key-123")

    runtime = resolve_runtime_provider(requested="custom", target_model="lan-model")

    assert runtime.get("base_url") == "http://other.test/v1"
    assert runtime.get("api_key") == "other-key-123"
