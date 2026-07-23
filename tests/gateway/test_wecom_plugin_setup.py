"""Tests for the WeCom plugin's interactive_setup wizard home-channel flow.

The interactive_setup wizard lazy-imports its CLI helpers from
``hermes_cli.config`` (get_env_value / save_env_value / remove_env_value),
``hermes_cli.cli_output`` (prompt / prompt_yes_no / print_*), and
``hermes_cli.setup`` (prompt_choice); we patch each at its source module so
the QR scan / pip paths never fire. Covers the home-channel clear-on-blank
behavior added in the follow-up to PR #58421.
"""
import hermes_cli.config as config_mod
import hermes_cli.cli_output as cli_output_mod
import hermes_cli.setup as setup_mod
from plugins.platforms.wecom.adapter import interactive_setup


def _patch_setup_io(monkeypatch, prompts, choice_responses, saved, removed, existing):
    prompt_iter = iter(prompts)
    choice_iter = iter(choice_responses)
    monkeypatch.setattr(config_mod, "get_env_value", lambda key: existing.get(key, ""))
    monkeypatch.setattr(config_mod, "save_env_value", lambda k, v: saved.update({k: v}))

    def _remove(key):
        removed.append(key)
        return existing.pop(key, None) is not None

    monkeypatch.setattr(config_mod, "remove_env_value", _remove)
    monkeypatch.setattr(cli_output_mod, "prompt", lambda *_a, **_kw: next(prompt_iter))
    monkeypatch.setattr(cli_output_mod, "prompt_yes_no", lambda *_a, **_kw: False)
    monkeypatch.setattr(setup_mod, "prompt_choice", lambda *_a, **_kw: next(choice_iter))
    for name in ("print_header", "print_info", "print_success", "print_warning", "print_error"):
        monkeypatch.setattr(cli_output_mod, name, lambda *_a, **_kw: None)
    # Stub QR scan: returns None so the wizard falls through to manual entry.
    import plugins.platforms.wecom.adapter as wecom_adapter_mod
    monkeypatch.setattr(wecom_adapter_mod, "qr_scan_for_bot_info", lambda: None)


# WeCom prompts (after method_choice): bot_id, secret (password), allowed_users, home.
# choice_responses: 1 = "Enter Bot ID/Secret manually" so the QR path is skipped.
_PROMPTS_NONEMPTY = [
    "wecom-bot-id-123",   # bot_id
    "wecom-secret-abc",   # secret
    "",                   # allowed_users (empty -> triggers DM policy choice below)
    "wecom-home-chat-id", # home chat ID
]
_PROMPTS_BLANK = [
    "wecom-bot-id-123",
    "wecom-secret-abc",
    "",
    "",
]
_PROMPTS_WHITESPACE = [
    "wecom-bot-id-123",
    "wecom-secret-abc",
    "",
    "   ",
]
# 1 = method "Enter manually", 1 = DM policy "pairing" (skip the 4-way choice path).
_CHOICES = [1, 1]


class TestWeComHomeChannelClear:
    """Blank home-channel answer must clear WECOM_HOME_CHANNEL (#12423)."""

    def test_blank_removes_existing_home_channel(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        saved, removed = {}, []
        _patch_setup_io(
            monkeypatch,
            _PROMPTS_BLANK,
            _CHOICES,
            saved,
            removed,
            existing={"WECOM_HOME_CHANNEL": "old-wecom-chat-id"},
        )
        interactive_setup()
        assert "WECOM_HOME_CHANNEL" in removed
        assert "WECOM_HOME_CHANNEL" not in saved

    def test_blank_without_prior_home_still_attempts_remove(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        saved, removed = {}, []
        _patch_setup_io(
            monkeypatch, _PROMPTS_BLANK, _CHOICES, saved, removed, existing={}
        )
        interactive_setup()
        assert removed.count("WECOM_HOME_CHANNEL") == 1

    def test_nonempty_saves_home_channel(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        saved, removed = {}, []
        _patch_setup_io(
            monkeypatch, _PROMPTS_NONEMPTY, _CHOICES, saved, removed, existing={}
        )
        interactive_setup()
        assert saved["WECOM_HOME_CHANNEL"] == "wecom-home-chat-id"
        assert "WECOM_HOME_CHANNEL" not in removed

    def test_whitespace_only_clears_home_channel(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        saved, removed = {}, []
        _patch_setup_io(
            monkeypatch,
            _PROMPTS_WHITESPACE,
            _CHOICES,
            saved,
            removed,
            existing={"WECOM_HOME_CHANNEL": "old-wecom-chat-id"},
        )
        interactive_setup()
        assert "WECOM_HOME_CHANNEL" in removed
        assert "WECOM_HOME_CHANNEL" not in saved