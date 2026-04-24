from hermes_cli.model_switch import DirectAlias, resolve_alias, switch_model
import hermes_cli.model_switch as ms
import hermes_cli.models as models
import hermes_cli.runtime_provider as runtime_provider


def test_direct_alias_exact_match_overrides_catalog_alias(monkeypatch):
    monkeypatch.setattr(
        ms,
        "DIRECT_ALIASES",
        {"sonnet": DirectAlias("custom-sonnet", "custom", "https://example.invalid/v1")},
    )

    def _unexpected_catalog_lookup(provider: str):
        raise AssertionError(f"catalog lookup should not run for exact direct alias: {provider}")

    monkeypatch.setattr(ms, "list_provider_models", _unexpected_catalog_lookup)

    assert resolve_alias("sonnet", "openrouter") == ("custom", "custom-sonnet", "sonnet")


def test_direct_alias_reverse_lookup_overrides_catalog_fuzzy_match(monkeypatch):
    monkeypatch.setattr(
        ms,
        "DIRECT_ALIASES",
        {"glm": DirectAlias("glm-4.7", "custom", "https://example.invalid/v1")},
    )

    def _unexpected_catalog_lookup(provider: str):
        raise AssertionError(f"catalog lookup should not run for reverse direct alias: {provider}")

    monkeypatch.setattr(ms, "list_provider_models", _unexpected_catalog_lookup)

    assert resolve_alias("glm-4.7", "openrouter") == ("custom", "glm-4.7", "glm")


def test_switch_model_exposes_model_policy_trace_for_provider_reroute(monkeypatch):
    monkeypatch.setattr(ms, "resolve_alias", lambda raw, provider: None)
    monkeypatch.setattr(ms, "is_aggregator", lambda provider: False)
    monkeypatch.setattr(models, "detect_provider_for_model", lambda model, current: ("anthropic", "claude-sonnet-4-6"))
    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", lambda requested, target_model: {
        "api_key": "secret-key-must-not-leak",
        "base_url": "https://api.anthropic.com",
        "api_mode": "anthropic_messages",
    })
    monkeypatch.setattr(ms, "normalize_model_for_provider", lambda model, provider: model)
    monkeypatch.setattr(models, "validate_requested_model", lambda *args, **kwargs: {"accepted": True})
    monkeypatch.setattr(ms, "get_model_capabilities", lambda provider, model: None)
    monkeypatch.setattr(ms, "get_model_info", lambda provider, model: None)

    result = switch_model(
        "claude-sonnet-4-6",
        current_provider="openrouter",
        current_model="openai/gpt-5.4",
        current_base_url="https://openrouter.ai/api/v1",
        current_api_key="primary-secret",
    )

    assert result.success is True
    assert result.target_provider == "anthropic"
    assert result.provider_changed is True
    assert result.model_policy_trace
    rendered_trace = repr(result.model_policy_trace)
    assert "explicit_override" in rendered_trace
    assert "secret-key-must-not-leak" not in rendered_trace
    assert "primary-secret" not in rendered_trace
