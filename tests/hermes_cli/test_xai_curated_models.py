"""Regression tests for xAI curated model list (OAuth picker)."""

from hermes_cli.inventory import ConfigContext, build_models_payload
from hermes_cli.models import _PROVIDER_MODELS, provider_model_ids


def test_xai_oauth_includes_grok_composer_2_5_fast():
    models = provider_model_ids("xai-oauth")
    assert "grok-composer-2.5-fast" in models


def test_grok_composer_slots_after_grok_build():
    models = _PROVIDER_MODELS["xai-oauth"]
    assert models[0] == "grok-build-0.1"
    assert models[1] == "grok-composer-2.5-fast"


def test_grok_composer_surfaces_in_shared_model_payload(monkeypatch):
    """Dashboard, TUI, and web pickers share build_models_payload.

    xAI OAuth Composer must survive that shared inventory layer, not just the
    low-level provider_model_ids() helper.
    """

    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(
        "hermes_cli.auth._load_auth_store",
        lambda: {"providers": {"xai-oauth": {"tokens": {"access_token": "token"}}}},
    )

    payload = build_models_payload(
        ConfigContext(
            current_provider="xai-oauth",
            current_model="",
            current_base_url="",
            user_providers={},
            custom_providers=[],
        ),
        include_unconfigured=True,
        picker_hints=True,
        canonical_order=True,
    )

    xai = next(p for p in payload["providers"] if p["slug"] == "xai-oauth")
    assert xai["authenticated"] is True
    assert "grok-composer-2.5-fast" in xai["models"]
