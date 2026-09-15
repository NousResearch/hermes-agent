"""Regression coverage for the configured model-picker allowlist (#111540)."""

from hermes_cli.model_allowlist import filter_allowed_model_rows, model_is_allowed


ALLOWED = [
    {"provider": "bedrock", "model": "anthropic.claude-sonnet-4"},
    {"provider": "nous", "model": "solar-pro-4"},
]


def test_filter_allowed_model_rows_keeps_only_configured_provider_model_pairs():
    rows = [
        {"slug": "bedrock", "models": ["anthropic.claude-sonnet-4", "meta.llama-4"]},
        {"slug": "nous", "models": ["solar-pro-4", "other"]},
        {"slug": "openrouter", "models": ["openai/gpt-5.6"]},
    ]

    assert filter_allowed_model_rows(rows, ALLOWED) == [
        {"slug": "bedrock", "models": ["anthropic.claude-sonnet-4"], "total_models": 1},
        {"slug": "nous", "models": ["solar-pro-4"], "total_models": 1},
    ]


def test_filter_allowed_model_rows_keeps_legacy_picker_when_not_configured():
    rows = [{"slug": "nous", "models": ["solar-pro-4"], "total_models": 1}]

    assert filter_allowed_model_rows(rows, []) == rows


def test_model_is_allowed_is_case_insensitive_and_requires_provider_match():
    assert model_is_allowed("SOLAR-PRO-4", "Nous", ALLOWED)
    assert not model_is_allowed("solar-pro-4", "bedrock", ALLOWED)


def test_authenticated_provider_picker_filters_after_current_model_injection():
    from hermes_cli.model_switch_providers import _finalize_picker_rows

    rows = [{
        "slug": "nous", "models": ["other"], "total_models": 1,
        "is_current": True, "native_catalog_empty": False,
    }]

    assert _finalize_picker_rows(rows, {}, "solar-pro-4", ALLOWED) == [{
        "slug": "nous", "models": ["solar-pro-4"], "total_models": 1,
        "is_current": True, "native_catalog_empty": False,
    }]


def test_switch_model_rejects_a_resolved_model_outside_the_allowlist(monkeypatch):
    from hermes_cli import model_switch

    def resolve_route(state):
        state.target_provider = "nous"
        state.new_model = "other"

    monkeypatch.setattr(model_switch, "_route_explicit_provider", resolve_route)
    monkeypatch.setattr(model_switch, "_resolve_switch_credentials", lambda _state: None)
    monkeypatch.setattr(model_switch, "_validate_switch", lambda _state: None)

    result = model_switch.switch_model(
        "other", "nous", "solar-pro-4", explicit_provider="nous", allowed_models=ALLOWED)

    assert not result.success
    assert "allowed_models" in result.error_message
