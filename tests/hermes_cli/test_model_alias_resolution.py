"""Regression tests for short model alias resolution ordering."""

from hermes_cli.model_switch import resolve_alias


def test_resolve_alias_prefers_curated_provider_order(monkeypatch):
    """Curated provider order should win over raw models.dev insertion order."""
    import hermes_cli.model_switch as ms

    monkeypatch.setattr(ms, "DIRECT_ALIASES", {})
    monkeypatch.setattr(ms, "_ensure_direct_aliases", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.models.provider_model_ids",
        lambda provider: [
            "claude-sonnet-4-6",
            "claude-sonnet-4-5-20250929",
        ],
    )
    monkeypatch.setattr(
        ms,
        "list_provider_models",
        lambda provider: [
            "claude-sonnet-4-0",
            "claude-sonnet-4-6",
            "claude-sonnet-4-5-20250929",
        ],
    )

    assert resolve_alias("sonnet", "anthropic") == (
        "anthropic",
        "claude-sonnet-4-6",
        "sonnet",
    )


def test_resolve_alias_falls_back_to_models_dev_when_curated_list_misses(monkeypatch):
    """Models.dev breadth remains available when the curated list lacks a match."""
    import hermes_cli.model_switch as ms

    monkeypatch.setattr(ms, "DIRECT_ALIASES", {})
    monkeypatch.setattr(ms, "_ensure_direct_aliases", lambda: None)
    monkeypatch.setattr("hermes_cli.models.provider_model_ids", lambda provider: ["gpt-5.4"])
    monkeypatch.setattr(
        ms,
        "list_provider_models",
        lambda provider: [
            "claude-sonnet-4-6",
            "gpt-5.4",
        ],
    )

    assert resolve_alias("sonnet", "anthropic") == (
        "anthropic",
        "claude-sonnet-4-6",
        "sonnet",
    )
