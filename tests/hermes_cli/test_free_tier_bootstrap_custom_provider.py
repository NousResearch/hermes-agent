"""A config.yaml ``custom``/``custom:<name>`` pin is a configured provider for the boot bootstrap."""

from __future__ import annotations

import pytest

from hermes_cli import auth as auth_mod
from hermes_cli import free_tier_bootstrap as fb

CUSTOM_PIN = {
    "model": {
        "provider": "custom:llama.cpp",
        "base_url": "http://127.0.0.1:8080/v1",
        "api_key": "local",
    }
}


@pytest.mark.parametrize("pin", ["custom:llama.cpp", "llamacpp", "ollama"])
def test_custom_pin_resolves_to_custom_on_the_auto_ladder(monkeypatch, pin):
    """``_config_model_provider`` is rung 2 of ``resolve_provider("auto")``; a ``custom:*`` pin or
    a documented alias (``llamacpp``) is not in ``PROVIDER_REGISTRY`` but must still answer, or the
    ladder falls through to env/host credentials and finally ``no_provider_configured`` (#107918)."""
    cfg = {"model": {**CUSTOM_PIN["model"], "provider": pin}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
    assert auth_mod._config_model_provider() == (cfg["model"], "custom")
    assert auth_mod.resolve_provider("auto", skip_free_tier=True) == "custom"


def test_custom_pin_counts_as_another_provider_for_the_bootstrap(monkeypatch):
    """The bootstrap record must be configured AND ``other_providers`` True, so a later free-tier
    mint does not claim ``active_provider`` over the user's explicit endpoint."""
    monkeypatch.setattr("hermes_cli.anon_auth.current_nous_state", lambda: None)
    monkeypatch.setattr("hermes_cli.anon_auth.guest_enabled", lambda: False)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: CUSTOM_PIN)
    fb.reset_for_tests()
    rec = fb.run_bootstrap(announce=False)
    assert rec.provider_configured is True
    assert rec.other_providers is True
    assert rec.inference_provider == "custom"
