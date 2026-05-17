"""Regression tests for hermes_cli.runtime_provider helpers."""
from __future__ import annotations

import pytest

from hermes_cli import runtime_provider as rp


def test_copilot_api_mode_honors_explicit_target_model_over_stale_config_default(monkeypatch):
    """Regression: when a user runs e.g.

        hermes --profile X -m claude-opus-4.7 --provider copilot ...

    from a profile whose ``model.default`` is ``gpt-5.x`` (a Responses-API
    model), ``_copilot_runtime_api_mode`` must compute api_mode from the
    *override* model — not silently fall back to ``model_cfg["default"]``
    and pick ``codex_responses`` for a Claude model that only serves
    ``/chat/completions``.

    Before the fix this surfaced as Copilot returning::

        HTTP 400: model claude-opus-4.7 does not support Responses API

    which was swallowed by oneshot's stderr redirect (exit 0, empty stdout).
    """
    # Profile config that mirrors the real-world repro: provider mismatch
    # disables the persisted api_mode shortcut, so the function falls through
    # to model-name based detection — which is the path under test.
    model_cfg = {
        "provider": "openai-codex",
        "api_mode": "codex_responses",
        "default": "gpt-5.5",
        "base_url": "https://example.invalid/codex",
    }

    # Stub out the catalog/normalization path so the test is fully offline.
    # _should_use_copilot_responses_api is pattern-based (offline) for the
    # exact IDs we pass, but normalize_copilot_model_id may try to hit the
    # catalog — short-circuit it to identity.
    monkeypatch.setattr(
        "hermes_cli.models.normalize_copilot_model_id",
        lambda model_id, catalog=None, api_key=None: model_id,
    )
    monkeypatch.setattr(
        "hermes_cli.models.fetch_github_model_catalog",
        lambda api_key=None: [],
    )

    # Override → Claude. Must NOT pick up gpt-5.5's codex_responses.
    assert (
        rp._copilot_runtime_api_mode(model_cfg, api_key="x", target_model="claude-opus-4.7")
        == "chat_completions"
    )

    # No override → falls back to model_cfg.default (gpt-5.5) → codex_responses.
    # Preserves the previous behaviour for callers that don't pass target_model.
    assert (
        rp._copilot_runtime_api_mode(model_cfg, api_key="x")
        == "codex_responses"
    )

    # Override → GPT-5 sibling: should resolve to codex_responses via target_model.
    assert (
        rp._copilot_runtime_api_mode(model_cfg, api_key="x", target_model="gpt-5.4")
        == "codex_responses"
    )


def test_copilot_api_mode_still_honors_matching_config(monkeypatch):
    """When the configured provider matches (``copilot``) and a persisted
    api_mode is set, the explicit-config path still wins regardless of
    target_model. This guards the existing happy path."""
    model_cfg = {
        "provider": "copilot",
        "api_mode": "codex_responses",
        "default": "gpt-5.5",
    }
    monkeypatch.setattr(
        "hermes_cli.models.normalize_copilot_model_id",
        lambda model_id, catalog=None, api_key=None: model_id,
    )
    monkeypatch.setattr(
        "hermes_cli.models.fetch_github_model_catalog",
        lambda api_key=None: [],
    )
    # Even with a Claude target_model, the explicit matching-provider config wins.
    assert (
        rp._copilot_runtime_api_mode(model_cfg, api_key="x", target_model="claude-opus-4.7")
        == "codex_responses"
    )
