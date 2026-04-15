"""Tests for context-length mismatch warnings after /model --global."""

from agent.models_dev import ModelInfo
from hermes_cli.model_switch import (
    ModelSwitchResult,
    get_context_length_mismatch_warning,
)


def _result(*, is_global=True, model_info=None):
    return ModelSwitchResult(
        success=True,
        new_model="glm-5-turbo",
        target_provider="zai",
        provider_changed=True,
        api_key="test-key",
        base_url="https://api.z.ai/v1",
        api_mode="chat_completions",
        model_info=model_info,
        is_global=is_global,
    )


def _model_info(context_window: int) -> ModelInfo:
    return ModelInfo(
        id="glm-5-turbo",
        name="GLM-5 Turbo",
        family="glm-5",
        provider_id="zai",
        context_window=context_window,
    )


def test_warns_when_global_switch_leaves_stale_config_override(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model": {"context_length": 256000}},
    )

    warning = get_context_length_mismatch_warning(
        _result(model_info=_model_info(202752))
    )

    assert "256,000" in warning
    assert "/context 202752" in warning


def test_no_warning_without_global_persistence(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model": {"context_length": 256000}},
    )

    warning = get_context_length_mismatch_warning(
        _result(is_global=False, model_info=_model_info(202752))
    )

    assert warning == ""


def test_no_warning_when_config_has_no_explicit_override(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"model": {}})

    warning = get_context_length_mismatch_warning(
        _result(model_info=_model_info(202752))
    )

    assert warning == ""


def test_falls_back_to_runtime_context_lookup_when_model_info_missing(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model": {"context_length": "256000"}},
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *args, **kwargs: 202752,
    )

    warning = get_context_length_mismatch_warning(_result(model_info=None))

    assert "256,000" in warning
    assert "/context 202752" in warning
