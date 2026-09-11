"""Custom-provider config pin must count as setup without flipping ``other_providers``."""

from __future__ import annotations

from hermes_cli import free_tier_bootstrap as fb

CUSTOM_PIN = {
    "model": {
        "provider": "custom:llama.cpp",
        "base_url": "http://127.0.0.1:8080/v1",
        "api_key": "local",
    }
}


def _stub_unresolved_nous(monkeypatch, config):
    def resolve_provider(requested="auto", skip_free_tier=False, **_kw):
        return "nous"

    monkeypatch.setattr("hermes_cli.auth.resolve_provider", resolve_provider)
    monkeypatch.setattr("hermes_cli.anon_auth.current_nous_state", lambda: None)
    monkeypatch.setattr("hermes_cli.anon_auth.guest_enabled", lambda: False)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)


def test_custom_model_dict_pin_is_provider_configured_without_flipping_other(monkeypatch):
    _stub_unresolved_nous(monkeypatch, CUSTOM_PIN)
    fb.reset_for_tests()
    rec = fb.run_bootstrap(announce=False)
    assert rec.provider_configured is True
    assert rec.other_providers is False


def test_plain_string_model_stays_unconfigured(monkeypatch):
    _stub_unresolved_nous(monkeypatch, {"model": "some-default-string"})
    fb.reset_for_tests()
    rec = fb.run_bootstrap(announce=False)
    assert rec.provider_configured is False
    assert rec.other_providers is False


def test_blank_model_dict_is_not_configured(monkeypatch):
    _stub_unresolved_nous(monkeypatch, {"model": {"provider": "  ", "base_url": "", "api_key": None}})
    fb.reset_for_tests()
    rec = fb.run_bootstrap(announce=False)
    assert rec.provider_configured is False
    assert rec.other_providers is False


def test_load_config_error_does_not_raise_or_flip_other(monkeypatch):
    def _boom():
        raise RuntimeError("config unreadable")

    _stub_unresolved_nous(monkeypatch, CUSTOM_PIN)
    monkeypatch.setattr("hermes_cli.config.load_config", _boom)
    fb.reset_for_tests()
    rec = fb.run_bootstrap(announce=False)
    assert rec.provider_configured is False
    assert rec.other_providers is False
