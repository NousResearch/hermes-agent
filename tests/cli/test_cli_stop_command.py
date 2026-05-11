"""Tests for /stop slash command session scoping."""

from cli import HermesCLI


def test_stop_command_stops_only_current_session_processes(monkeypatch, capsys):
    """Regression: /stop must not kill background jobs from other sessions."""
    from tools import process_registry as process_registry_module

    calls = {"list": [], "kill": []}

    class _FakeRegistry:
        def list_sessions(self, task_id=None):
            calls["list"].append(task_id)
            # The scoped session has one running process. If the command asks
            # globally, expose another session too so the old bug is visible.
            if task_id == "session-a":
                return [{"status": "running", "session_id": "proc-a"}]
            return [
                {"status": "running", "session_id": "proc-a"},
                {"status": "running", "session_id": "proc-b"},
            ]

        def kill_all(self, task_id=None):
            calls["kill"].append(task_id)
            return 1 if task_id == "session-a" else 2

    monkeypatch.setattr(process_registry_module, "process_registry", _FakeRegistry())

    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "session-a"

    cli._handle_stop_command()

    assert calls == {"list": ["session-a"], "kill": ["session-a"]}
    out = capsys.readouterr().out
    assert "Stopping 1 background process" in out
    assert "Stopped 1 process" in out
