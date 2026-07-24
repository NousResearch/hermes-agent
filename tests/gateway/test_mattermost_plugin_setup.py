"""Tests for the Mattermost plugin's interactive_setup wizard home-channel flow.

The interactive_setup wizard lazy-imports its CLI helpers from
``hermes_cli.config`` (get_env_value / save_env_value / remove_env_value) and
``hermes_cli.cli_output`` (prompt / prompt_yes_no / print_*); we patch those
source modules. Covers the home-channel clear-on-blank behavior added in
PR #58421 and extended in the follow-up.
"""
import hermes_cli.config as config_mod
import hermes_cli.cli_output as cli_output_mod
from plugins.platforms.mattermost.adapter import interactive_setup


def _patch_setup_io(monkeypatch, prompts, saved, removed, existing):
    prompt_iter = iter(prompts)
    monkeypatch.setattr(config_mod, "get_env_value", lambda key: existing.get(key, ""))
    monkeypatch.setattr(config_mod, "save_env_value", lambda k, v: saved.update({k: v}))

    def _remove(key):
        removed.append(key)
        return existing.pop(key, None) is not None

    monkeypatch.setattr(config_mod, "remove_env_value", _remove)
    monkeypatch.setattr(cli_output_mod, "prompt", lambda *_a, **_kw: next(prompt_iter))
    monkeypatch.setattr(cli_output_mod, "prompt_yes_no", lambda *_a, **_kw: False)
    for name in ("print_header", "print_info", "print_success", "print_warning"):
        monkeypatch.setattr(cli_output_mod, name, lambda *_a, **_kw: None)


# Mattermost prompts: server_url, bot_token (password), allowed_users, home_channel.
_PROMPTS_NONEMPTY = ["https://mm.example.com", "«redacted:mm-token»", "", "town-square-id"]
_PROMPTS_BLANK = ["https://mm.example.com", "«redacted:mm-token»", "", ""]
_PROMPTS_WHITESPACE = ["https://mm.example.com", "«redacted:mm-token»", "", "   "]


class TestMattermostHomeChannelClear:
    """Blank home-channel answer must clear MATTERMOST_HOME_CHANNEL (#12423)."""

    def test_blank_removes_existing_home_channel(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        saved, removed = {}, []
        _patch_setup_io(
            monkeypatch,
            _PROMPTS_BLANK,
            saved,
            removed,
            existing={"MATTERMOST_HOME_CHANNEL": "old-channel-id"},
        )
        interactive_setup()
        assert "MATTERMOST_HOME_CHANNEL" in removed
        assert "MATTERMOST_HOME_CHANNEL" not in saved

    def test_blank_without_prior_home_still_attempts_remove(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        saved, removed = {}, []
        _patch_setup_io(
            monkeypatch, _PROMPTS_BLANK, saved, removed, existing={}
        )
        interactive_setup()
        assert removed.count("MATTERMOST_HOME_CHANNEL") == 1

    def test_nonempty_saves_home_channel(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        saved, removed = {}, []
        _patch_setup_io(
            monkeypatch, _PROMPTS_NONEMPTY, saved, removed, existing={}
        )
        interactive_setup()
        assert saved["MATTERMOST_HOME_CHANNEL"] == "town-square-id"
        assert "MATTERMOST_HOME_CHANNEL" not in removed

    def test_whitespace_only_clears_home_channel(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        saved, removed = {}, []
        _patch_setup_io(
            monkeypatch,
            _PROMPTS_WHITESPACE,
            saved,
            removed,
            existing={"MATTERMOST_HOME_CHANNEL": "old-channel-id"},
        )
        interactive_setup()
        assert "MATTERMOST_HOME_CHANNEL" in removed
        assert "MATTERMOST_HOME_CHANNEL" not in saved