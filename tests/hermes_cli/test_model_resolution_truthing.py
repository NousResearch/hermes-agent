from hermes_cli.model_switch import DirectAlias, resolve_alias
import hermes_cli.model_switch as ms


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
