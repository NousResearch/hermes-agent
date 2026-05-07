from __future__ import annotations

import pytest


def test_stop_command_cleans_terminal_environments_without_background_processes(
    monkeypatch, capsys
):
    import cli
    import tools.terminal_tool as terminal_tool
    from tools.process_registry import process_registry

    monkeypatch.setattr(process_registry, "list_sessions", lambda: [])
    monkeypatch.setattr(
        process_registry,
        "kill_all",
        lambda: pytest.fail("kill_all should not run without active processes"),
    )
    monkeypatch.setattr(terminal_tool, "cleanup_all_environments", lambda: 2)

    instance = cli.HermesCLI.__new__(cli.HermesCLI)
    instance._handle_stop_command()

    output = capsys.readouterr().out
    assert "Cleaned up 2 terminal environment(s)." in output
    assert "No running background processes." not in output
