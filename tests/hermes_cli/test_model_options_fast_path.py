"""Regression tests for fast, degraded desktop model-picker payloads."""

from __future__ import annotations

import time


def _ctx():
    from hermes_cli.inventory import ConfigContext

    return ConfigContext(
        current_provider="hermes-direct-pro",
        current_model="gpt-5.5",
        current_base_url="https://api.weapi.pw/v1",
        user_providers={
            "hermes-direct-pro": {
                "name": "Hermes Direct Pro",
                "api": "https://api.weapi.pw/v1",
                "transport": "codex_responses",
                "default_model": "gpt-5.5",
                "models": {"gpt-5.5": {}, "gpt-5.4-mini": {}},
            },
            "hermes-direct-cc": {
                "name": "Hermes Direct CC",
                "api": "https://api.weapi.pw/v1",
                "transport": "openai",
                "default_model": "claude-fable-5",
                "models": {"claude-fable-5": {}, "claude-opus-4-8": {}},
            },
        },
        custom_providers=[],
    )


def test_build_models_payload_fast_omits_slow_enrichment(monkeypatch):
    """Normal picker opens must not block on pricing/capabilities enrichment."""
    import hermes_cli.inventory as inv

    monkeypatch.setattr(
        "hermes_cli.model_switch.list_authenticated_providers",
        lambda **kwargs: [
            {
                "slug": "hermes-direct-pro",
                "name": "Hermes Direct Pro",
                "is_current": True,
                "is_user_defined": True,
                "models": ["gpt-5.5", "gpt-5.4-mini"],
                "total_models": 2,
                "source": "user-config",
            }
        ],
    )

    def slow_pricing(_rows, **_kwargs):
        raise AssertionError("pricing enrichment should not run on fast picker path")

    def slow_capabilities(_rows):
        raise AssertionError("capability enrichment should not run on fast picker path")

    monkeypatch.setattr(inv, "_apply_pricing", slow_pricing)
    monkeypatch.setattr(inv, "_apply_capabilities", slow_capabilities)

    payload = inv.build_models_payload(
        _ctx(),
        include_unconfigured=True,
        picker_hints=True,
        canonical_order=True,
        pricing=False,
        capabilities=False,
    )

    row = next(p for p in payload["providers"] if p["slug"] == "hermes-direct-pro")
    assert row["models"] == ["gpt-5.5", "gpt-5.4-mini"]
    assert "pricing" not in row
    assert "capabilities" not in row


def test_build_models_payload_degrades_when_enrichment_is_slow(monkeypatch):
    """Explicit enriched payloads should return the base list if enrichment stalls."""
    import hermes_cli.inventory as inv

    monkeypatch.setattr(
        "hermes_cli.model_switch.list_authenticated_providers",
        lambda **kwargs: [
            {
                "slug": "hermes-direct-pro",
                "name": "Hermes Direct Pro",
                "is_current": True,
                "is_user_defined": True,
                "models": ["gpt-5.5"],
                "total_models": 1,
                "source": "user-config",
            }
        ],
    )

    def slow_pricing(_rows, **_kwargs):
        time.sleep(0.05)

    def slow_capabilities(_rows):
        raise RuntimeError("models.dev unavailable")

    monkeypatch.setattr(inv, "_apply_pricing", slow_pricing)
    monkeypatch.setattr(inv, "_apply_capabilities", slow_capabilities)

    payload = inv.build_models_payload(
        _ctx(),
        include_unconfigured=True,
        picker_hints=True,
        canonical_order=True,
        pricing=True,
        capabilities=True,
        enrichment_timeout=0.01,
    )

    row = next(p for p in payload["providers"] if p["slug"] == "hermes-direct-pro")
    assert row["models"] == ["gpt-5.5"]
    assert payload["degraded"] is True
    assert any("model metadata enrichment" in warning for warning in payload["warnings"])


def test_explicit_provider_rejects_model_outside_declared_provider_models(monkeypatch):
    """A session /model switch must fail before hitting the upstream account group."""
    from hermes_cli.model_switch import switch_model

    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model",
        lambda *args, **kwargs: {
            "accepted": True,
            "persist": True,
            "recognized": False,
            "message": "soft accepted",
        },
    )

    result = switch_model(
        raw_input="claude-fable-5",
        current_provider="hermes-direct-pro",
        current_model="gpt-5.5",
        current_base_url="https://api.weapi.pw/v1",
        explicit_provider="hermes-direct-pro",
        user_providers=_ctx().user_providers,
        custom_providers=[],
    )

    assert result.success is False
    assert "does not belong to provider" in result.error_message
    assert "hermes-direct-pro" in result.error_message


def test_explicit_provider_accepts_declared_model(monkeypatch):
    from hermes_cli.model_switch import switch_model

    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model",
        lambda *args, **kwargs: {
            "accepted": True,
            "persist": True,
            "recognized": True,
            "message": None,
        },
    )
    monkeypatch.setattr("hermes_cli.model_switch.get_model_capabilities", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_info", lambda *a, **k: None)

    result = switch_model(
        raw_input="gpt-5.5",
        current_provider="hermes-direct-pro",
        current_model="gpt-5.4-mini",
        current_base_url="https://api.weapi.pw/v1",
        explicit_provider="hermes-direct-pro",
        user_providers=_ctx().user_providers,
        custom_providers=[],
    )

    assert result.success is True
    assert result.new_model == "gpt-5.5"
    assert result.target_provider == "hermes-direct-pro"
