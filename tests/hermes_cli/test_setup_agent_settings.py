"""Tests for agent-settings copy in the interactive setup wizard."""

from hermes_cli.setup import setup_agent_settings


def test_setup_agent_settings_uses_displayed_max_iterations_value(tmp_path, monkeypatch, capsys):
    """The helper text should match the value shown in the prompt.

    After PR#18413 max_turns is read exclusively from config.yaml — the
    .env `HERMES_MAX_ITERATIONS` fallback was removed because it was
    shadowing the user's current config (see the 60-vs-500 incident).
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    config = {
        "agent": {"max_turns": 60},
        "display": {"tool_progress": "all"},
        "compression": {"threshold": 0.50},
    }

    prompt_answers = iter(["60", "all", "0.5"])

    monkeypatch.setattr("hermes_cli.setup.prompt", lambda *args, **kwargs: next(prompt_answers))
    monkeypatch.setattr("hermes_cli.setup.prompt_choice", lambda *args, **kwargs: 4)
    monkeypatch.setattr("hermes_cli.setup.save_env_value", lambda *args, **kwargs: None)
    monkeypatch.setattr("hermes_cli.setup.remove_env_value", lambda *args, **kwargs: None)
    monkeypatch.setattr("hermes_cli.setup.save_config", lambda *args, **kwargs: None)

    setup_agent_settings(config)

    out = capsys.readouterr().out
    assert "Press Enter to keep 60." in out
    assert "Default is 90" not in out


def test_setup_agent_settings_prefers_config_over_stale_env(tmp_path, monkeypatch, capsys):
    """Config.yaml wins even when a stale .env value disagrees.

    Regression guard for the bug where `.env HERMES_MAX_ITERATIONS=60`
    from an old `hermes setup` run shadowed `agent.max_turns: 500` in
    config.yaml. The wizard must now display the config value.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    config = {
        "agent": {"max_turns": 500},  # user bumped this in config.yaml
        "display": {"tool_progress": "all"},
        "compression": {"threshold": 0.50},
    }

    prompt_answers = iter(["500", "all", "0.5"])

    # Simulate stale .env value — the wizard must ignore this.
    monkeypatch.setattr(
        "hermes_cli.setup.get_env_value",
        lambda key: "60" if key == "HERMES_MAX_ITERATIONS" else "",
    )
    monkeypatch.setattr("hermes_cli.setup.prompt", lambda *args, **kwargs: next(prompt_answers))
    monkeypatch.setattr("hermes_cli.setup.prompt_choice", lambda *args, **kwargs: 4)
    monkeypatch.setattr("hermes_cli.setup.save_env_value", lambda *args, **kwargs: None)

    removed_keys: list[str] = []
    monkeypatch.setattr(
        "hermes_cli.setup.remove_env_value",
        lambda key: (removed_keys.append(key), True)[1],
    )
    monkeypatch.setattr("hermes_cli.setup.save_config", lambda *args, **kwargs: None)

    setup_agent_settings(config)

    out = capsys.readouterr().out
    # Config value wins
    assert "Press Enter to keep 500." in out
    assert "Press Enter to keep 60." not in out
    # And the stale .env entry gets cleaned up
    assert "HERMES_MAX_ITERATIONS" in removed_keys


def test_first_time_defaults_keep_every_platform_tool_progress_default(tmp_path, monkeypatch):
    """Quick and full first-time setup run this. A global display.tool_progress it
    writes beats every platform tier, so Telegram and Slack went from off to all."""
    from gateway.display_config import _PLATFORM_DEFAULTS, resolve_tool_progress
    from gateway.run import _load_gateway_config
    from hermes_cli.config import load_config
    from hermes_cli.setup import _apply_default_agent_settings

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    _apply_default_agent_settings(load_config())

    on_disk = _load_gateway_config(tmp_path / "config.yaml")
    assert on_disk.get("agent", {}).get("max_turns") == 150  # the save really landed
    for platform in _PLATFORM_DEFAULTS:
        assert resolve_tool_progress(on_disk, platform) == resolve_tool_progress({}, platform), platform
