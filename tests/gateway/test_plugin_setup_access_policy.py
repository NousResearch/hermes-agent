"""Regression tests for plugin setup access-policy prompts."""

import hermes_cli.cli_output as cli_output_mod
import plugins.platforms.discord.adapter as discord_mod
import plugins.platforms.mattermost.adapter as mattermost_mod


_VALID_DISCORD_TOKEN = "discord-bot-token"


def _read_env(tmp_path):
    return (tmp_path / ".env").read_text(encoding="utf-8")


def test_discord_setup_blank_allowlist_can_enable_open_access(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    answers = iter([_VALID_DISCORD_TOKEN, "", ""])
    monkeypatch.setattr(cli_output_mod, "prompt", lambda *_a, **_kw: next(answers))
    monkeypatch.setattr(cli_output_mod, "prompt_yes_no", lambda *_a, **_kw: True)
    monkeypatch.setattr(cli_output_mod, "print_header", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli_output_mod, "print_info", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli_output_mod, "print_success", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli_output_mod, "print_warning", lambda *_a, **_kw: None)

    discord_mod.interactive_setup()

    env_text = _read_env(tmp_path)
    assert f"DISCORD_BOT_TOKEN={_VALID_DISCORD_TOKEN}" in env_text
    assert "DISCORD_ALLOWED_USERS=" in env_text
    assert "DISCORD_ALLOW_ALL_USERS=true" in env_text


def test_discord_setup_allowlist_still_cleans_and_disables_open_access(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    answers = iter([_VALID_DISCORD_TOKEN, "<@123>, user:456, alice", ""])
    monkeypatch.setattr(cli_output_mod, "prompt", lambda *_a, **_kw: next(answers))
    monkeypatch.setattr(cli_output_mod, "prompt_yes_no", lambda *_a, **_kw: False)
    monkeypatch.setattr(cli_output_mod, "print_header", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli_output_mod, "print_info", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli_output_mod, "print_success", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli_output_mod, "print_warning", lambda *_a, **_kw: None)

    discord_mod.interactive_setup()

    env_text = _read_env(tmp_path)
    assert "DISCORD_ALLOWED_USERS=123,456,alice" in env_text
    assert "DISCORD_ALLOW_ALL_USERS=false" in env_text


def test_mattermost_setup_blank_allowlist_can_enable_open_access(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    answers = iter(["https://mm.example.com", "mattermost-token", "", ""])
    monkeypatch.setattr(cli_output_mod, "prompt", lambda *_a, **_kw: next(answers))
    monkeypatch.setattr(cli_output_mod, "prompt_yes_no", lambda *_a, **_kw: True)
    monkeypatch.setattr(cli_output_mod, "print_header", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli_output_mod, "print_info", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli_output_mod, "print_success", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli_output_mod, "print_warning", lambda *_a, **_kw: None)

    mattermost_mod.interactive_setup()

    env_text = _read_env(tmp_path)
    assert "MATTERMOST_URL=https://mm.example.com" in env_text
    assert "MATTERMOST_TOKEN=mattermost-token" in env_text
    assert "MATTERMOST_ALLOWED_USERS=" in env_text
    assert "MATTERMOST_ALLOW_ALL_USERS=true" in env_text
