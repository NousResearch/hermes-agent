"""Tests for the configurable Telegram bot-command menu cap.

``telegram.menu_max_commands`` lets an operator raise the per-scope command
menu cap above the conservative default (``MAX_COMMANDS_PER_SCOPE``) so plugin
and profile slash commands (e.g. ``/finance``) surface in Telegram's "/" menu
instead of being trimmed once core + skill commands fill the default slots.
Telegram hard-limits 100 commands per scope, so the value is clamped to 1..100.
"""
import sys
from unittest.mock import MagicMock

from gateway.config import PlatformConfig


def _ensure_telegram_mock():
    """Mock the python-telegram-bot package if it's not installed."""
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    mod = MagicMock()
    mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)


def _adapter(extra=None):
    _ensure_telegram_mock()
    from gateway.platforms.telegram import TelegramAdapter

    cfg = PlatformConfig(enabled=True, token="fake-token", extra=extra or {})
    return TelegramAdapter(cfg)


def test_menu_cap_defaults_to_constant():
    from gateway.platforms.telegram import MAX_COMMANDS_PER_SCOPE

    assert _adapter()._menu_max_commands == MAX_COMMANDS_PER_SCOPE


def test_menu_cap_override_is_honored():
    assert _adapter({"menu_max_commands": 50})._menu_max_commands == 50


def test_menu_cap_accepts_string_encoded_int():
    # config/env round-trips can yield strings
    assert _adapter({"menu_max_commands": "45"})._menu_max_commands == 45


def test_menu_cap_clamped_to_telegram_maximum():
    assert _adapter({"menu_max_commands": 500})._menu_max_commands == 100


def test_menu_cap_clamped_to_minimum():
    assert _adapter({"menu_max_commands": 0})._menu_max_commands == 1


def test_menu_cap_invalid_value_falls_back_to_default():
    from gateway.platforms.telegram import MAX_COMMANDS_PER_SCOPE

    assert (
        _adapter({"menu_max_commands": "not-an-int"})._menu_max_commands
        == MAX_COMMANDS_PER_SCOPE
    )


def test_config_loader_passes_menu_max_commands_into_extra(tmp_path, monkeypatch):
    """Top-level ``telegram.menu_max_commands`` must reach PlatformConfig.extra
    (mirrors disable_link_previews) so the adapter can read it."""
    from gateway.config import load_gateway_config, Platform

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "telegram:\n  enabled: true\n  token: fake-token\n  menu_max_commands: 50\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    cfg = load_gateway_config()
    tg = cfg.platforms.get(Platform.TELEGRAM)

    assert tg is not None
    assert (tg.extra or {}).get("menu_max_commands") == 50
