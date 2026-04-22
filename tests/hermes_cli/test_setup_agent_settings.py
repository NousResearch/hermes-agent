"""Tests for agent-settings copy in the interactive setup wizard."""

from hermes_cli.setup import setup_agent_settings


def test_setup_agent_settings_uses_displayed_max_iterations_value(tmp_path, monkeypatch, capsys):
    """The helper text should match the value shown in the prompt."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    config = {
        "agent": {"max_turns": 90},
        "display": {"tool_progress": "all"},
        "compression": {"threshold": 0.50},
        "session_reset": {"mode": "both", "idle_minutes": 1440, "at_hour": 4},
    }

    prompt_answers = iter(["60", "all", "0.5"])

    monkeypatch.setattr("hermes_cli.setup.get_env_value", lambda key: "60" if key == "HERMES_MAX_ITERATIONS" else "")
    monkeypatch.setattr("hermes_cli.setup.prompt", lambda *args, **kwargs: next(prompt_answers))
    monkeypatch.setattr("hermes_cli.setup.prompt_choice", lambda *args, **kwargs: 4)
    monkeypatch.setattr("hermes_cli.setup.save_env_value", lambda *args, **kwargs: None)
    monkeypatch.setattr("hermes_cli.setup.save_config", lambda *args, **kwargs: None)

    setup_agent_settings(config)

    out = capsys.readouterr().out
    assert "Press Enter to keep 60." in out
    assert "Default is 90" not in out
