"""Tests for the Discord plugin's interactive_setup wizard home-channel flow.

The interactive_setup wizard lazy-imports its CLI helpers from
``hermes_cli.config`` (get_env_value / save_env_value / remove_env_value) and
``hermes_cli.cli_output`` (prompt / prompt_yes_no / print_*); we patch those
source modules. Covers the home-channel clear-on-blank behavior added in
PR #58421 and extended in the follow-up.
"""
import hermes_cli.config as config_mod
import hermes_cli.cli_output as cli_output_mod
from plugins.platforms.discord.adapter import interactive_setup


def _patch_setup_io(monkeypatch, prompts, saved, removed, existing, infos=None):
    prompt_iter = iter(prompts)
    monkeypatch.setattr(config_mod, "get_env_value", lambda key: existing.get(key, ""))
    monkeypatch.setattr(config_mod, "save_env_value", lambda k, v: saved.update({k: v}))

    def _remove(key):
        removed.append(key)
        return existing.pop(key, None) is not None

    monkeypatch.setattr(config_mod, "remove_env_value", _remove)
    monkeypatch.setattr(cli_output_mod, "prompt", lambda *_a, **_kw: next(prompt_iter))
    monkeypatch.setattr(cli_output_mod, "prompt_yes_no", lambda *_a, **_kw: False)
    for name in ("print_header", "print_success", "print_warning"):
        monkeypatch.setattr(cli_output_mod, name, lambda *_a, **_kw: None)

    def _info(*args, **_kw):
        if infos is not None:
            infos.append(" ".join(str(a) for a in args))

    monkeypatch.setattr(cli_output_mod, "print_info", _info)


# Discord prompts: bot_token (password), allowed_users, home_channel.
_PROMPTS_NONEMPTY = ["«redacted:discord-bot-token»", "", "123456789012345678"]
_PROMPTS_BLANK = ["«redacted:discord-bot-token»", "", ""]
_PROMPTS_WHITESPACE = ["«redacted:discord-bot-token»", "", "   "]


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


class TestDiscordSetupPrivilegedIntentsGuidance:
    """Setup must name Privileged Gateway Intents before asking for the token (#79430)."""

    def test_setup_mentions_message_content_intent(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        saved, removed, infos = {}, [], []
        _patch_setup_io(
            monkeypatch,
            _PROMPTS_BLANK,
            saved,
            removed,
            existing={},
            infos=infos,
        )
        interactive_setup()
        joined = "\n".join(infos)
        assert "Message Content Intent" in joined
        assert "Privileged Gateway Intents" in joined
        assert "discord.com/developers/applications" in joined


