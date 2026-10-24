"""Tests for the Discord plugin's interactive_setup wizard.

The interactive_setup wizard lazy-imports its CLI helpers from
``hermes_cli.config`` (get_env_value / save_env_value / remove_env_value) and
``hermes_cli.cli_output`` (prompt / prompt_yes_no / print_*); we patch those
source modules. Covers the home-channel clear-on-blank behavior added in
PR #58421 and extended in the follow-up.
"""
import hermes_cli.config as config_mod
import hermes_cli.cli_output as cli_output_mod
from tools import discord_tool
from plugins.platforms.discord import onboarding
from plugins.platforms.discord.onboarding import interactive_setup
from tests.fakes.platforms.discord_standin import APP_ID, TOKEN, DiscordStandin

_real_check = onboarding.check_bot_token


def _offline(_token):
    raise OSError("offline")


def _patch_setup_io(monkeypatch, prompts, saved, removed, existing, infos=None, yes=False, check=_offline,
                    writes=None):
    prompt_iter = iter(prompts)
    monkeypatch.setattr(onboarding, "check_bot_token", check)
    monkeypatch.setattr(config_mod, "get_env_value", lambda key: existing.get(key, ""))
    def _save(key, value):
        if writes is not None:
            writes.append((key, value))
        saved[key] = value

    monkeypatch.setattr(config_mod, "save_env_value", _save)

    def _remove(key):
        removed.append(key)
        return existing.pop(key, None) is not None

    monkeypatch.setattr(config_mod, "remove_env_value", _remove)
    monkeypatch.setattr(cli_output_mod, "prompt", lambda *_a, **_kw: next(prompt_iter))
    monkeypatch.setattr(cli_output_mod, "prompt_yes_no", lambda *_a, **_kw: yes)
    for name in ("print_header", "print_success", "print_warning"):
        monkeypatch.setattr(cli_output_mod, name, lambda *_a, **_kw: None)

    def _info(*args, **_kw):
        if infos is not None:
            infos.append(" ".join(str(a) for a in args))

    monkeypatch.setattr(cli_output_mod, "print_info", _info)


# Discord prompts: bot_token (password), allowed_users, home_channel.
_PROMPTS_NONEMPTY = ["fake-bot-token.part2.part3", "", "123456789012345678"]
_PROMPTS_BLANK = ["fake-bot-token.part2.part3", "", ""]
_PROMPTS_WHITESPACE = ["fake-bot-token.part2.part3", "", "   "]


class TestDiscordHomeChannelClear:
    """Blank home-channel answer must clear DISCORD_HOME_CHANNEL (#12423)."""

    def test_blank_removes_existing_home_channel(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        saved, removed = {}, []
        _patch_setup_io(
            monkeypatch,
            _PROMPTS_BLANK,
            saved,
            removed,
            existing={"DISCORD_HOME_CHANNEL": "987654321098765432"},
        )
        interactive_setup()
        assert "DISCORD_HOME_CHANNEL" in removed
        assert "DISCORD_HOME_CHANNEL" not in saved


class TestDiscordTokenCheckedWithDiscord:
    """The wizard asks Discord about the pasted token: a rejected token is never written, and an
    accepted one yields the invite link and adds the bot's owner to whoever is already allowed."""

    def test_rejected_token_never_written_and_owner_added_to_allowlist(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        standin = DiscordStandin().start()
        try:
            monkeypatch.setattr(discord_tool, "DISCORD_API_BASE", standin.api_base)
            saved, removed, infos, errors, writes = {}, [], [], [], []
            # A rejected token, then the real one pasted with rich-text curly quotes around it.
            prompts = ["not.the.token", f"\u201c{TOKEN}\u201d", "", ""]
            _patch_setup_io(monkeypatch, prompts, saved, removed, existing={"DISCORD_ALLOWED_USERS": "999"},
                            infos=infos, yes=True, check=_real_check, writes=writes)
            monkeypatch.setattr(cli_output_mod, "print_error", lambda *a, **_kw: errors.append(" ".join(map(str, a))))
            interactive_setup()
        finally:
            standin.stop()
        assert any("rejected" in e for e in errors)
        assert [v for k, v in writes if k == "DISCORD_BOT_TOKEN"] == [TOKEN]
        assert saved["DISCORD_ALLOWED_USERS"] == "999,1"  # existing entry kept, stand-in owner added
        invite = next(line.strip() for line in infos if "oauth2/authorize" in line)
        assert f"client_id={APP_ID}" in invite and f"permissions={onboarding.INVITE_PERMISSIONS}" in invite


class TestDiscordTokenShapeGuard:
    """A numeric application ID pasted as the bot token is rejected with guidance
    (port of openclaw/openclaw#140531)."""

    def test_numeric_app_id_reprompts_then_accepts_real_token(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        saved, removed, errors = {}, [], []
        real_token = "fake-bot-token." + "part2.part3"
        _patch_setup_io(
            monkeypatch,
            ["1234567890123456789", real_token, "", ""],
            saved,
            removed,
            existing={},
        )
        monkeypatch.setattr(cli_output_mod, "print_error", lambda *a, **_kw: errors.append(" ".join(map(str, a))))
        interactive_setup()
        assert saved.get("DISCORD_BOT_TOKEN") == real_token
        assert any("application ID" in e for e in errors)

    def test_non_numeric_token_saves_without_error(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        saved, removed, errors = {}, [], []
        _patch_setup_io(monkeypatch, _PROMPTS_BLANK, saved, removed, existing={})
        monkeypatch.setattr(cli_output_mod, "print_error", lambda *a, **_kw: errors.append(" ".join(map(str, a))))
        interactive_setup()
        assert saved.get("DISCORD_BOT_TOKEN") == _PROMPTS_BLANK[0]
        assert errors == []
