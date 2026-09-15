"""Regression coverage for the configured model-picker allowlist (#111540)."""

from types import SimpleNamespace

from hermes_cli.model_allowlist import filter_allowed_model_rows, model_is_allowed
from hermes_cli.model_switch import ModelSwitchResult


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


def test_filter_keeps_valid_pairs_from_a_partially_invalid_allowlist():
    rows = [
        {"slug": "nous", "models": ["solar-pro-4", "other"]},
        {"slug": "bedrock", "models": ["anthropic.claude-sonnet-4"]},
    ]
    mixed = [{"provider": "nous"}, {"provider": "nous", "model": "solar-pro-4"}, "skip"]

    assert filter_allowed_model_rows(rows, mixed) == [
        {"slug": "nous", "models": ["solar-pro-4"], "total_models": 1},
    ]


def test_all_invalid_nonempty_allowlist_is_restrictive_not_unrestricted():
    rows = [{"slug": "nous", "models": ["solar-pro-4"], "total_models": 1}]
    malformed = [{"provider": "nous"}]

    assert filter_allowed_model_rows(rows, malformed) == []
    assert not model_is_allowed("solar-pro-4", "nous", malformed)
    assert model_is_allowed("solar-pro-4", "nous", [])
    assert model_is_allowed("solar-pro-4", "nous", None)


def _bound(fn, instance):
    return fn.__get__(instance, type(instance))


def _select_picker_row(monkeypatch, allowed_models):
    import cli as cli_mod

    captured = {}

    def _switch(**kwargs):
        captured["kwargs"] = kwargs
        return ModelSwitchResult(success=True, new_model="solar-pro-4", target_provider="nous")

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", _switch)
    applied = {}
    self_ = SimpleNamespace(
        _app=None,
        _model_picker_state={
            "stage": "model",
            "provider_data": {
                "slug": "nous",
                "capabilities": {"solar-pro-4": {"reasoning": False}},
            },
            "model_list": ["solar-pro-4"],
            "selected": 0,
            "user_provs": {"nous": {}},
            "custom_provs": [],
            "allowed_models": allowed_models,
        },
        provider="nous",
        model="other",
        base_url="",
        api_key="",
        _restore_modal_input_snapshot=lambda: None,
        _invalidate=lambda **_kwargs: None,
        _confirm_and_apply_model_switch_result=lambda *args: applied.setdefault("args", args),
    )
    self_._close_model_picker = _bound(cli_mod.HermesCLI._close_model_picker, self_)
    self_._commit_picker_result = _bound(cli_mod.HermesCLI._commit_picker_result, self_)
    _bound(cli_mod.HermesCLI._handle_model_picker_selection, self_)(persist_global=False)
    return captured["kwargs"], applied["args"]


def test_interactive_picker_selection_passes_configured_allowlist(monkeypatch):
    kwargs, applied = _select_picker_row(monkeypatch, ALLOWED)

    assert kwargs["allowed_models"] == ALLOWED
    assert applied[0].success


def test_interactive_picker_selection_passes_empty_allowlist(monkeypatch):
    kwargs, applied = _select_picker_row(monkeypatch, [])

    assert kwargs["allowed_models"] == []
    assert applied[0].success
