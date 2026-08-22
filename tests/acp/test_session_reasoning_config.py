"""ACP sessions must honor configured reasoning settings (#85153).

``SessionManager._make_agent`` builds its ``AIAgent`` from config like every
other surface, but never passed ``reasoning_config`` — so
``agent.reasoning_effort: none`` was silently ignored and ACP sessions fell
back to the provider default. For non-reasoning models (e.g. ``gpt-4o-mini``)
that default injects an unsupported reasoning parameter and the request fails.

These tests exercise the real config-file → ``load_config`` →
``resolve_reasoning_config`` chain against the per-test ``HERMES_HOME``; only
the ``AIAgent`` constructor and provider resolution are stubbed.
"""

import os
from pathlib import Path

import pytest
import yaml

from acp_adapter.session import SessionManager


class _CapturingAgent:
    """Stand-in for AIAgent that records its constructor kwargs."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = kwargs.get("model") or "stub-model"


@pytest.fixture
def acp_env(monkeypatch):
    """Stub the heavyweight edges of ``_make_agent``; return a config writer."""
    monkeypatch.setattr("run_agent.AIAgent", _CapturingAgent)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None, **_kwargs: {
            "provider": requested or "openai-api",
            "api_mode": "chat_completions",
            "base_url": "https://example.invalid/v1",
            "api_key": "test-key",
        },
    )
    monkeypatch.setattr(
        "acp_adapter.session._register_task_cwd", lambda task_id, cwd: None
    )
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build",
        lambda **_kwargs: None,
    )

    def _write_config(cfg: dict) -> None:
        home = Path(os.environ["HERMES_HOME"])
        (home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    return _write_config


def test_acp_agent_receives_reasoning_effort_none(acp_env):
    """``reasoning_effort: none`` must reach the ACP agent as disabled."""
    acp_env(
        {
            "model": {"default": "gpt-4o-mini", "provider": "openai-api"},
            "agent": {"reasoning_effort": "none"},
        }
    )

    state = SessionManager(db=None).create_session(cwd=".")

    assert state.agent.kwargs["reasoning_config"] == {"enabled": False}


def test_acp_agent_receives_global_effort_level(acp_env):
    """A global effort level propagates to ACP sessions like the CLI."""
    acp_env(
        {
            "model": {"default": "deepseek/deepseek-v4", "provider": "openrouter"},
            "agent": {"reasoning_effort": "high"},
        }
    )

    state = SessionManager(db=None).create_session(cwd=".")

    assert state.agent.kwargs["reasoning_config"] == {
        "enabled": True,
        "effort": "high",
    }


def test_acp_per_model_override_keys_off_session_model(acp_env):
    """Overrides resolve against the session's model, not ``model.default``."""
    acp_env(
        {
            "model": {"default": "default-model", "provider": "openai-api"},
            "agent": {
                "reasoning_effort": "high",
                "reasoning_overrides": {"session-model": "none"},
            },
        }
    )

    agent = SessionManager(db=None)._make_agent(
        session_id="acp-reasoning-test", cwd=".", model="session-model"
    )

    assert agent.kwargs["model"] == "session-model"
    assert agent.kwargs["reasoning_config"] == {"enabled": False}


def test_acp_unset_reasoning_keeps_provider_default(acp_env):
    """No configured reasoning → None, so the provider default still applies."""
    acp_env({"model": {"default": "gpt-4o-mini", "provider": "openai-api"}})

    state = SessionManager(db=None).create_session(cwd=".")

    assert state.agent.kwargs.get("reasoning_config") is None
