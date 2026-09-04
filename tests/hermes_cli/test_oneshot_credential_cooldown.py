"""One-shot runs must not spend a credential the pool already benched.

``hermes -z`` exists to make exactly one request. When every live credential
for the configured provider is serving a 429, resolution still hands back a
key, so that one request 429s before the agent's own fallback can engage --
the whole run is wasted.

A one-shot process is single-shot by definition, so there is nothing to
restore: the next invocation resolves from the configured primary again and
returns to it on its own once the window lifts.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.runtime_provider import CREDENTIALS_COOLING_DOWN_KEY

_PRIMARY = {
    "api_key": "sk-test-000-not-a-real-key",
    "base_url": "https://generativelanguage.googleapis.com/v1beta",
    "provider": "gemini",
    "api_mode": "chat_completions",
}
_FALLBACK = {
    "api_key": "sk-test-000-not-a-real-fallback-key",
    "base_url": "https://openrouter.ai/api/v1",
    "provider": "openrouter",
    "api_mode": "chat_completions",
}

_CONFIG = {
    "model": {"default": "gemini-3.7-flash", "provider": "gemini"},
    "fallback_providers": [{"provider": "openrouter", "model": "z-ai/glm-5.2"}],
}


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    """The fallback's own pool probe must not read the developer's store."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


def _run(monkeypatch, primary):
    requested: list[str | None] = []

    def _resolve(**kwargs):
        requested.append(kwargs.get("requested"))
        if (kwargs.get("requested") or "gemini") == "gemini":
            return dict(primary)
        return dict(_FALLBACK)

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider", _resolve
    )
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: dict(_CONFIG))

    import hermes_cli.oneshot as oneshot

    with patch("run_agent.AIAgent") as agent_cls, \
         patch("hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build"), \
         patch("hermes_cli.oneshot._create_session_db_for_oneshot", return_value=None):
        agent = MagicMock()
        agent.run_conversation.return_value = {"final_response": "ok"}
        agent_cls.return_value = agent
        oneshot._run_agent(prompt="hi")

    return requested, agent_cls.call_args.kwargs


def test_a_rate_limited_primary_runs_the_turn_on_a_fallback(monkeypatch):
    """Provider and model move together, or a Gemini model id reaches OpenRouter."""
    primary = {**_PRIMARY, CREDENTIALS_COOLING_DOWN_KEY: time.time() + 1800}

    requested, kwargs = _run(monkeypatch, primary)

    assert "openrouter" in requested
    assert kwargs["provider"] == "openrouter"
    assert kwargs["api_key"] == "sk-test-000-not-a-real-fallback-key"
    assert kwargs["model"] == "z-ai/glm-5.2"


def test_a_healthy_primary_is_left_alone(monkeypatch):
    """The chain is only consulted for a provider that is actually benched."""
    requested, kwargs = _run(monkeypatch, _PRIMARY)

    assert requested == [None]
    assert kwargs["provider"] == "gemini"
    assert kwargs["model"] == "gemini-3.7-flash"
