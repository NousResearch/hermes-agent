"""Regression tests for CLI reasoning state after ``/model`` switches."""

from __future__ import annotations

from unittest.mock import patch

from hermes_cli.model_switch import ModelSwitchResult


class _FakeAgent:
    def __init__(self):
        self.switches = []
        self.reasoning_config = {"enabled": True, "effort": "medium"}

    def switch_model(self, **kwargs):
        self.switches.append(kwargs)
        self.reasoning_config = {"enabled": False}


class _StubCLI:
    def __init__(self, agent=None):
        self.agent = agent
        self.model = "reasoning-model"
        self.provider = "test-provider"
        self.requested_provider = "test-provider"
        self.api_key = "test-key"
        self._explicit_api_key = "test-key"
        self.base_url = "https://example.invalid/v1"
        self._explicit_base_url = self.base_url
        self.api_mode = "chat_completions"
        self.reasoning_config = {"enabled": True, "effort": "medium"}
        self._pending_model_switch_note = ""


def _switch_result():
    return ModelSwitchResult(
        success=True,
        new_model="non-reasoning-model",
        target_provider="test-provider",
        provider_changed=False,
        api_key="test-key",
        base_url="https://example.invalid/v1",
        api_mode="chat_completions",
        warning_message="",
        provider_label="Test Provider",
        resolved_via_alias="",
        capabilities=None,
        model_info=None,
        is_global=False,
    )


def _apply_switch(cli, monkeypatch):
    import cli as cli_mod

    monkeypatch.setattr(cli_mod, "_cprint", lambda *_a, **_k: None)
    config = {
        "agent": {
            "reasoning_effort": "medium",
            "reasoning_overrides": {"non-reasoning-model": "none"},
        }
    }
    with patch("hermes_cli.config.load_config", return_value=config):
        cli_mod.HermesCLI._apply_model_switch_result(
            cli, _switch_result(), persist_global=False
        )


def test_preinit_model_switch_updates_cli_reasoning_state(monkeypatch):
    cli = _StubCLI(agent=None)

    _apply_switch(cli, monkeypatch)

    assert cli.reasoning_config == {"enabled": False}


def test_initialized_model_switch_updates_cli_reasoning_state(monkeypatch):
    agent = _FakeAgent()
    cli = _StubCLI(agent=agent)

    _apply_switch(cli, monkeypatch)

    assert cli.reasoning_config == {"enabled": False}
    assert len(agent.switches) == 1


def test_initialized_switch_copies_live_agent_state_without_reloading(monkeypatch):
    import cli as cli_mod

    agent = _FakeAgent()
    cli = _StubCLI(agent=agent)
    monkeypatch.setattr(cli_mod, "_cprint", lambda *_a, **_k: None)

    with patch("hermes_cli.config.load_config", side_effect=RuntimeError("unavailable")):
        cli_mod.HermesCLI._apply_model_switch_result(
            cli, _switch_result(), persist_global=False
        )

    assert cli.reasoning_config == {"enabled": False}
