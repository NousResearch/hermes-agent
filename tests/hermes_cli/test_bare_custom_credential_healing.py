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


CONTINUUM_CONFIG = {
    "model": {"default": "zai-org/GLM-5.3-Flash", "provider": "custom"},
    "providers": {
        "continuum": {
            "name": "Continuum",
            "base_url": "https://api.console.example.test/v1",
            "model": "zai-org/GLM-5.3-Flash",
            "key_env": "HERMES_TEST_CONTINUUM_KEY",
        }
    },
}


def test_bare_custom_rides_the_serving_named_entry(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: CONTINUUM_CONFIG)
    monkeypatch.setenv("HERMES_TEST_CONTINUUM_KEY", "test-key-123")

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
