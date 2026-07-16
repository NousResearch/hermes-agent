"""Tests for /model picker search/filter functionality.

Covers:
- Filtered selection resolves correct model from filtered list
- Escape-clear then selection resolves from unfiltered list
- Back → another provider → selection works correctly
"""
from types import SimpleNamespace

import pytest

from hermes_cli.model_switch import ModelSwitchResult


def _bound(fn, instance):
    return fn.__get__(instance, type(instance))


def test_model_picker_filtered_selection(monkeypatch):
    """Selecting from filtered list resolves correct model."""
    import cli as cli_mod

    result = ModelSwitchResult(
        success=True,
        new_model="anthropic/claude-opus-4",
        target_provider="anthropic",
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **_kwargs: result,
    )

    captured = {}

    class _Thread:
        def __init__(self, *, target, args, daemon):
            captured["target"] = target
            captured["args"] = args
            captured["daemon"] = daemon

        def start(self):
            captured["started"] = True

    monkeypatch.setattr(cli_mod.threading, "Thread", _Thread)

    self_ = SimpleNamespace(
        _app=object(),
        _model_picker_state={
            "stage": "model",
            "provider_data": {"slug": "anthropic"},
            "model_list": [
                "anthropic/claude-3.5-sonnet",
                "anthropic/claude-opus-4",
                "openai/gpt-4o",
            ],
            "filter_text": "opus",
            "selected": 0,  # Selects first (and only) filtered result
            "user_provs": None,
            "custom_provs": None,
        },
        provider="anthropic",
        model="anthropic/claude-3.5-sonnet",
        base_url="",
        api_key="",
        _restore_modal_input_snapshot=lambda: None,
        _invalidate=lambda **_kwargs: None,
    )
    self_._close_model_picker = _bound(cli_mod.HermesCLI._close_model_picker, self_)
    self_._confirm_and_apply_model_switch_result = (
        lambda *_args: captured.setdefault("ran_inline", True)
    )
    self_._get_filtered_models = cli_mod.HermesCLI._get_filtered_models

    _bound(cli_mod.HermesCLI._handle_model_picker_selection, self_)(persist_global=True)

    assert self_._model_picker_state is None
    assert captured["started"] is True
    assert captured["daemon"] is True
    # Should pass the filtered model (opus-4), not the first unfiltered model
    assert captured["args"][0].new_model == "anthropic/claude-opus-4"


def test_model_picker_escape_clear_then_selection(monkeypatch):
    """Escape clears filter; subsequent selection uses full list."""
    import cli as cli_mod

    result = ModelSwitchResult(
        success=True,
        new_model="anthropic/claude-3.5-sonnet",
        target_provider="anthropic",
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **_kwargs: result,
    )

    captured = {}

    class _Thread:
        def __init__(self, *, target, args, daemon):
            captured["target"] = target
            captured["args"] = args
            captured["daemon"] = daemon

        def start(self):
            captured["started"] = True

    monkeypatch.setattr(cli_mod.threading, "Thread", _Thread)

    # Simulate: filter="opus" (1 match), then ESC clears filter, then select index 0
    self_ = SimpleNamespace(
        _app=object(),
        _model_picker_state={
            "stage": "model",
            "provider_data": {"slug": "anthropic"},
            "model_list": [
                "anthropic/claude-3.5-sonnet",
                "anthropic/claude-opus-4",
            ],
            "filter_text": "",  # ESC already cleared it
            "selected": 0,  # First item in full list
            "user_provs": None,
            "custom_provs": None,
        },
        provider="anthropic",
        model="anthropic/claude-3.5-sonnet",
        base_url="",
        api_key="",
        _restore_modal_input_snapshot=lambda: None,
        _invalidate=lambda **_kwargs: None,
    )
    self_._close_model_picker = _bound(cli_mod.HermesCLI._close_model_picker, self_)
    self_._confirm_and_apply_model_switch_result = (
        lambda *_args: captured.setdefault("ran_inline", True)
    )
    self_._get_filtered_models = cli_mod.HermesCLI._get_filtered_models

    _bound(cli_mod.HermesCLI._handle_model_picker_selection, self_)(persist_global=True)

    assert self_._model_picker_state is None
    assert captured["started"] is True
    # Should resolve to first model in full list (sonnet), not filtered
    assert captured["args"][0].new_model == "anthropic/claude-3.5-sonnet"


def test_model_picker_back_then_provider_switch_then_selection(monkeypatch):
    """Back to provider list, pick different provider, then select model."""
    import cli as cli_mod

    result = ModelSwitchResult(
        success=True,
        new_model="openai/gpt-4o",
        target_provider="openai",
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **_kwargs: result,
    )

    captured = {}

    class _Thread:
        def __init__(self, *, target, args, daemon):
            captured["target"] = target
            captured["args"] = args
            captured["daemon"] = daemon

        def start(self):
            captured["started"] = True

    monkeypatch.setattr(cli_mod.threading, "Thread", _Thread)

    # State simulates: was in model stage for anthropic, pressed Back,
    # now in provider stage with openai selected, then enter model stage for openai
    self_ = SimpleNamespace(
        _app=object(),
        _model_picker_state={
            "stage": "model",
            "provider_data": {"slug": "openai"},
            "model_list": ["openai/gpt-4o", "openai/gpt-4o-mini"],
            "filter_text": "",
            "selected": 0,
            "user_provs": None,
            "custom_provs": None,
            "providers": [
                {"slug": "anthropic", "name": "Anthropic", "is_current": False},
                {"slug": "openai", "name": "OpenAI", "is_current": True},
            ],
        },
        provider="anthropic",
        model="anthropic/claude-3.5-sonnet",
        base_url="",
        api_key="",
        _restore_modal_input_snapshot=lambda: None,
        _invalidate=lambda **_kwargs: None,
    )
    self_._close_model_picker = _bound(cli_mod.HermesCLI._close_model_picker, self_)
    self_._confirm_and_apply_model_switch_result = (
        lambda *_args: captured.setdefault("ran_inline", True)
    )
    self_._get_filtered_models = cli_mod.HermesCLI._get_filtered_models

    _bound(cli_mod.HermesCLI._handle_model_picker_selection, self_)(persist_global=True)

    assert self_._model_picker_state is None
    assert captured["started"] is True
    # Should resolve openai model, not anthropic
    assert captured["args"][0].new_model == "openai/gpt-4o"
    assert captured["args"][0].target_provider == "openai"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])