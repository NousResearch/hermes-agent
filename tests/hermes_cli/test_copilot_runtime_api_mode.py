"""Regression tests for Copilot runtime api_mode resolution."""

from __future__ import annotations


def test_copilot_runtime_api_mode_uses_target_model_over_stale_config_default(monkeypatch):
    """MoA/fallback slots must derive Copilot api_mode from the slot model.

    Repro: user's main/default Copilot model is GPT-5.5 (Responses API), but a
    MoA reference slot is Copilot Claude Opus (Chat Completions). Runtime
    resolution previously looked only at model.default and returned
    codex_responses for the Claude slot, causing Copilot to reject it with
    "model ... does not support Responses API".
    """
    from hermes_cli import runtime_provider as rp

    monkeypatch.setattr(
        "hermes_cli.models.copilot_model_api_mode",
        lambda model, api_key=None: "codex_responses" if str(model).startswith("gpt-5") else "chat_completions",
    )

    assert rp._copilot_runtime_api_mode(
        {"provider": "copilot", "default": "gpt-5.5"},
        "token",
        target_model="claude-opus-4.8",
    ) == "chat_completions"


def test_copilot_runtime_api_mode_still_uses_default_without_target(monkeypatch):
    from hermes_cli import runtime_provider as rp

    monkeypatch.setattr(
        "hermes_cli.models.copilot_model_api_mode",
        lambda model, api_key=None: "codex_responses" if str(model).startswith("gpt-5") else "chat_completions",
    )

    assert rp._copilot_runtime_api_mode(
        {"provider": "copilot", "default": "gpt-5.5"},
        "token",
    ) == "codex_responses"
