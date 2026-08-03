import logging

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter, interactive_setup


def _make_adapter() -> DiscordAdapter:
    return DiscordAdapter(PlatformConfig(enabled=True, token="***"))


def test_discord_fail_closed_default_logs_once(monkeypatch, caplog):
    adapter = _make_adapter()
    adapter._allowed_user_ids = set()
    adapter._allowed_role_ids = set()

    for var in (
        "DISCORD_ALLOWED_CHANNELS",
        "DISCORD_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(var, raising=False)

    with caplog.at_level(logging.WARNING):
        adapter._warn_if_fail_closed_default()
        adapter._warn_if_fail_closed_default()

    messages = [record.message for record in caplog.records]
    matches = [
        msg for msg in messages
        if "Discord messages are being denied because no allowlist is configured" in msg
    ]
    assert len(matches) == 1
    assert "DISCORD_ALLOWED_USERS" in matches[0]
    assert "DISCORD_ALLOW_ALL_USERS=true" in matches[0]


def test_discord_fail_closed_default_warning_skips_explicit_channel_gate(monkeypatch, caplog):
    adapter = _make_adapter()
    adapter._allowed_user_ids = set()
    adapter._allowed_role_ids = set()
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "12345")
    monkeypatch.delenv("DISCORD_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)

    with caplog.at_level(logging.WARNING):
        adapter._warn_if_fail_closed_default()

    assert "no allowlist is configured" not in caplog.text


def test_discord_setup_existing_token_warns_fail_closed_not_fail_open(monkeypatch):
    info_lines: list[str] = []
    yes_no_answers = iter([False, False])

    def fake_get_env_value(key: str):
        return "token" if key == "DISCORD_BOT_TOKEN" else ""

    monkeypatch.setattr("hermes_cli.config.get_env_value", fake_get_env_value)
    monkeypatch.setattr("hermes_cli.config.save_env_value", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("hermes_cli.cli_output.print_header", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("hermes_cli.cli_output.print_success", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("hermes_cli.cli_output.print_info", lambda msg="", **_kwargs: info_lines.append(str(msg)))
    monkeypatch.setattr("hermes_cli.cli_output.prompt", lambda *_args, **_kwargs: "")
    monkeypatch.setattr("hermes_cli.cli_output.prompt_yes_no", lambda *_args, **_kwargs: next(yes_no_answers))

    interactive_setup()

    joined = "\n".join(info_lines)
    assert "anyone can use your bot" not in joined
    assert "fail-closed default" in joined
    assert "DISCORD_ALLOW_ALL_USERS=true" in joined


def test_discord_setup_new_token_empty_allowlist_warns_denied_until_configured(monkeypatch):
    info_lines: list[str] = []
    prompts = iter(["token", "", ""])

    monkeypatch.setattr("hermes_cli.config.get_env_value", lambda _key: "")
    monkeypatch.setattr("hermes_cli.config.save_env_value", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("hermes_cli.cli_output.print_header", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("hermes_cli.cli_output.print_success", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("hermes_cli.cli_output.print_info", lambda msg="", **_kwargs: info_lines.append(str(msg)))
    monkeypatch.setattr("hermes_cli.cli_output.prompt", lambda *_args, **_kwargs: next(prompts))
    monkeypatch.setattr("hermes_cli.cli_output.prompt_yes_no", lambda *_args, **_kwargs: False)

    interactive_setup()

    joined = "\n".join(info_lines)
    assert "anyone in servers with your bot can use it" not in joined
    assert "Discord will deny messages" in joined
    assert "DISCORD_ALLOWED_ROLES" in joined
    assert "DISCORD_ALLOW_ALL_USERS=true" in joined
