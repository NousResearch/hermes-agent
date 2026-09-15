"""Tests for Copilot runtime api_mode resolution."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


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


@pytest.mark.parametrize("credential_source", ["pool", "explicit", "env", "hermes-auth-store"])
@pytest.mark.parametrize(
    ("configured_model", "target_model", "expected_mode"),
    [
        ("gpt-5.5", "claude-opus-4.8", "chat_completions"),
        ("gpt-5.5", "gemini-3-pro-preview", "chat_completions"),
        ("claude-opus-4.8", "gpt-5.5", "codex_responses"),
    ],
)
def test_resolver_routes_copilot_by_target_model_for_every_credential_path(
    monkeypatch,
    credential_source,
    configured_model,
    target_model,
    expected_mode,
):
    """The public resolver must propagate the target through every auth path."""
    from hermes_cli import models
    from hermes_cli import runtime_provider as rp

    model_cfg = {"provider": "copilot", "default": configured_model}
    monkeypatch.setattr(rp, "_get_model_config", lambda: model_cfg)
    monkeypatch.setattr(rp, "resolve_provider", lambda *_args, **_kwargs: "copilot")
    # Keep this a real copilot_model_api_mode decision without making a live
    # catalog request. The API-mode contract is determined by the model family.
    monkeypatch.setattr(models, "fetch_github_model_catalog", lambda **_kwargs: [])

    kwargs = {"requested": "copilot", "target_model": target_model}
    if credential_source == "pool":
        entry = SimpleNamespace(
            runtime_api_key="pool-token",
            access_token="pool-token",
            runtime_base_url="https://api.githubcopilot.com",
            base_url="https://api.githubcopilot.com",
            source="pool",
        )
        pool = SimpleNamespace(
            has_credentials=lambda: True,
            select=lambda: entry,
        )
        monkeypatch.setattr(rp, "load_pool", lambda _provider: pool)
    elif credential_source == "explicit":
        kwargs["explicit_api_key"] = "explicit-token"
        monkeypatch.setattr(
            rp,
            "load_pool",
            lambda _provider: pytest.fail("explicit credentials must bypass the pool"),
        )
    else:
        monkeypatch.setattr(rp, "load_pool", lambda _provider: None)
        monkeypatch.setattr(
            rp,
            "resolve_api_key_provider_credentials",
            lambda _provider: {
                "api_key": f"{credential_source}-token",
                "base_url": "https://api.githubcopilot.com",
                "source": credential_source,
            },
        )

    runtime = rp.resolve_runtime_provider(**kwargs)

    assert runtime["provider"] == "copilot"
    assert runtime["api_mode"] == expected_mode
    assert runtime["source"] == credential_source
