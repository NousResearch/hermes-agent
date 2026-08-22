"""Regression: ACP sessions must resolve ``reasoning_config``.

Bug: ``SessionManager._make_agent`` built ``AIAgent(**kwargs)`` without a
``reasoning_config`` entry, so ``agent_init`` stored ``None``. Nothing
re-resolves it at build time, so the Anthropic adapter skipped emitting the
``thinking`` parameter entirely. Adaptive Claude then fell back to the API
default ``display: "omitted"`` and returned ``thinking`` blocks whose text was
empty with only an opaque ``signature`` populated -- i.e. every ACP host (Buzz,
Zed) lost reasoning text that the CLI and gateway both show, regardless of
``agent.reasoning_effort``.

Every other surface that constructs ``AIAgent`` resolves effort through
``hermes_constants.resolve_reasoning_config`` (CLI, TUI gateway, api_server,
cron). ACP is asserted to do the same, including honouring a per-model
override.
"""

import pytest

from acp_adapter.session import SessionManager


class _FakeAgent:
    model = "fake-model"

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _install_fakes(monkeypatch, config):
    monkeypatch.setattr("run_agent.AIAgent", _FakeAgent)
    monkeypatch.setattr("acp_adapter.session.load_config", lambda: config, raising=False)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None: {
            "provider": requested,
            "api_mode": "anthropic_messages",
            "base_url": "https://example.invalid",
            "api_key": "test-key",
        },
    )
    monkeypatch.setattr("acp_adapter.session._register_task_cwd", lambda task_id, cwd: None)


def _config(**agent_cfg):
    return {
        "model": {"default": "fake-model", "provider": "fake-provider"},
        "mcp_servers": {},
        "agent": agent_cfg,
    }


class TestACPReasoningConfig:
    def test_make_agent_forwards_global_reasoning_effort(self, monkeypatch):
        """The global ``agent.reasoning_effort`` must reach AIAgent."""
        _install_fakes(monkeypatch, _config(reasoning_effort="high"))

        state = SessionManager(db=None).create_session(cwd="/tmp/project")

        assert state.agent.kwargs.get("reasoning_config") == {
            "enabled": True,
            "effort": "high",
        }

    def test_make_agent_honors_per_model_override(self, monkeypatch):
        """A per-model override must beat the global effort, as elsewhere."""
        _install_fakes(
            monkeypatch,
            _config(
                reasoning_effort="low",
                reasoning_overrides={"fake-model": "high"},
            ),
        )

        state = SessionManager(db=None).create_session(cwd="/tmp/project")

        assert state.agent.kwargs.get("reasoning_config") == {
            "enabled": True,
            "effort": "high",
        }

    def test_make_agent_passes_reasoning_config_key(self, monkeypatch):
        """The kwarg must always be present, so AIAgent never silently
        defaults it to None (the original bug)."""
        _install_fakes(monkeypatch, _config(reasoning_effort="medium"))

        state = SessionManager(db=None).create_session(cwd="/tmp/project")

        assert "reasoning_config" in state.agent.kwargs

    def test_make_agent_forwards_disabled_reasoning(self, monkeypatch):
        """``reasoning_effort: none`` must reach ACP as an explicit disable.

        Regression for #85153: a non-reasoning model (gpt-4o-mini) 400s on the
        transport's default reasoning setting when ``None`` lets it fall
        through. The disable has to arrive as ``{"enabled": False}``.
        """
        _install_fakes(monkeypatch, _config(reasoning_effort="none"))

        state = SessionManager(db=None).create_session(cwd="/tmp/project")

        assert state.agent.kwargs.get("reasoning_config") == {"enabled": False}
