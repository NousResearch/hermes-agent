"""Tests for /focus slash command helpers (CLI mixin logic, no TTY)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.fixture
def focus_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    from agent.brain_networks.runtime import reset_orchestrator_for_tests

    reset_orchestrator_for_tests()
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"brain_networks": {"enabled": True, "ecn_max_task_stack": 8}},
    )
    yield home
    reset_orchestrator_for_tests()


class _FakeCLI:
    """Minimal stand-in that uses the real mixin handler."""

    def __init__(self, session_id: str = "cli-focus"):
        from agent.brain_networks.runtime import get_orchestrator

        orch = get_orchestrator()
        assert orch is not None
        orch.bind_session(session_id)
        self.agent = SimpleNamespace(session_id=session_id, _brain_orchestrator=orch)
        self._prints = []

    # Mixin prints via print() — capture by monkeypatching in tests


def test_focus_pin_show_clear(focus_env, monkeypatch, capsys):
    from hermes_cli.cli_commands_mixin import CLICommandsMixin

    class H(CLICommandsMixin, _FakeCLI):
        pass

    h = H("sess-focus")
    h._handle_focus_command("/focus ship the brain networks upgrade")
    out = capsys.readouterr().out
    assert "pinned" in out.lower() or "Focus pinned" in out

    h._handle_focus_command("/focus show")
    out = capsys.readouterr().out
    assert "ship the brain networks upgrade" in out

    h._handle_focus_command("/focus clear")
    out = capsys.readouterr().out
    assert "cleared" in out.lower()

    h._handle_focus_command("/focus status")
    out = capsys.readouterr().out
    assert "(none)" in out or "Focus:" in out


def test_focus_requires_enabled(focus_env, monkeypatch, capsys):
    from hermes_cli.cli_commands_mixin import CLICommandsMixin

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"brain_networks": {"enabled": False}},
    )

    class H(CLICommandsMixin, _FakeCLI):
        def __init__(self):
            self.agent = SimpleNamespace(session_id="x", _brain_orchestrator=None)

    h = H()
    h._handle_focus_command("/focus something")
    out = capsys.readouterr().out
    assert "brain_networks.enabled" in out


def test_command_registry_has_focus():
    from hermes_cli.commands import COMMAND_REGISTRY

    names = {c.name for c in COMMAND_REGISTRY}
    assert "focus" in names
