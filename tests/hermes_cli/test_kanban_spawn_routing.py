"""Regression tests for deterministic local-first Kanban worker routing."""

from hermes_cli.kanban_db import _resolve_local_first_route


def test_local_first_route_uses_effective_local_fallback_before_remote() -> None:
    """A remote primary must not cost a doomed egress attempt first."""
    profile = {
        "model": {"provider": "nous", "default": "poolside/laguna-xs-2.1:free"},
        "fallback_model": [
            {"provider": "openai-codex", "model": "gpt-5.6-luna"},
            {"provider": "ollama-launch", "model": "hermes-review-fast:latest"},
        ],
    }

    assert _resolve_local_first_route(profile) == (
        "ollama-launch",
        "hermes-review-fast:latest",
    )


def test_local_first_route_prefers_local_primary() -> None:
    profile = {
        "model": {"provider": "ollama-launch", "default": "hermes-cron-fast:latest"},
        "fallback_model": [
            {"provider": "nous", "model": "poolside/laguna-xs-2.1:free"},
        ],
    }

    assert _resolve_local_first_route(profile) == (
        "ollama-launch",
        "hermes-cron-fast:latest",
    )


def test_local_first_route_fails_closed_when_no_local_model_is_configured() -> None:
    profile = {
        "model": {"provider": "nous", "default": "poolside/laguna-xs-2.1:free"},
        "fallback_model": [
            {"provider": "openai-codex", "model": "gpt-5.6-luna"},
        ],
    }

    assert _resolve_local_first_route(profile) is None
