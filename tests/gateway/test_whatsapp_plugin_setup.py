"""Tests for the WhatsApp plugin's interactive_setup wizard home-channel flow.

The interactive_setup wizard lazy-imports its CLI helpers from
``hermes_cli.config`` (get_env_value / save_env_value / remove_env_value) and
``hermes_cli.cli_output`` (prompt / prompt_yes_no / print_*); we patch those
source modules. Covers the home-channel clear-on-blank behavior added in
PR #58421 and extended in the follow-up.
"""
import hermes_cli.config as config_mod
import hermes_cli.cli_output as cli_output_mod
from plugins.platforms.whatsapp.adapter import interactive_setup


def _patch_setup_io(monkeypatch, prompts, yes_no_responses, saved, removed, existing):
    prompt_iter = iter(prompts)
    yes_no_iter = iter(yes_no_responses)
    monkeypatch.setattr(config_mod, "get_env_value", lambda key: existing.get(key, ""))
    monkeypatch.setattr(config_mod, "save_env_value", lambda k, v: saved.update({k: v}))

    def _remove(key):
        removed.append(key)
        return existing.pop(key, None) is not None

    monkeypatch.setattr(config_mod, "remove_env_value", _remove)
    monkeypatch.setattr(cli_output_mod, "prompt", lambda *_a, **_kw: next(prompt_iter))
    monkeypatch.setattr(
        cli_output_mod, "prompt_yes_no", lambda *_a, **_kw: next(yes_no_iter)
    )
    for name in ("print_header", "print_info", "print_success", "print_warning"):
        monkeypatch.setattr(cli_output_mod, name, lambda *_a, **_kw: None)


# WhatsApp prompts (after the Enable? prompt_yes_no): allowed_users, home_channel.
# Enable? = True so we don't return early.
_PROMPTS_NONEMPTY = ["", "12025550100@c.us"]
_PROMPTS_BLANK = ["", ""]
_PROMPTS_WHITESPACE = ["", "   "]
_YES_NO = [True]  # Enable WhatsApp? -> True


class TestWhatsAppHomeChannelClear:
    """Blank home-channel answer must clear WHATSAPP_HOME_CHANNEL (#12423)."""

    def test_blank_removes_existing_home_channel(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        saved, removed = {}, []
        _patch_setup_io(
            monkeypatch,
            _PROMPTS_BLANK,
            _YES_NO,
            saved,
            removed,
            existing={"WHATSAPP_HOME_CHANNEL": "12025550100@c.us"},
        )
        interactive_setup()
        assert "WHATSAPP_HOME_CHANNEL" in removed
        assert "WHATSAPP_HOME_CHANNEL" not in saved

    def test_blank_without_prior_home_still_attempts_remove(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        saved, removed = {}, []
        _patch_setup_io(
            monkeypatch, _PROMPTS_BLANK, _YES_NO, saved, removed, existing={}
        )
        interactive_setup()
        assert removed.count("WHATSAPP_HOME_CHANNEL") == 1

    def test_nonempty_saves_home_channel(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        saved, removed = {}, []
        _patch_setup_io(
            monkeypatch, _PROMPTS_NONEMPTY, _YES_NO, saved, removed, existing={}
        )
        interactive_setup()
        assert saved["WHATSAPP_HOME_CHANNEL"] == "12025550100@c.us"
        assert "WHATSAPP_HOME_CHANNEL" not in removed

    def test_whitespace_only_clears_home_channel(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        saved, removed = {}, []
        _patch_setup_io(
            monkeypatch,
            _PROMPTS_WHITESPACE,
            _YES_NO,
            saved,
            removed,
            existing={"WHATSAPP_HOME_CHANNEL": "12025550100@c.us"},
        )
        interactive_setup()
        assert "WHATSAPP_HOME_CHANNEL" in removed
        assert "WHATSAPP_HOME_CHANNEL" not in saved