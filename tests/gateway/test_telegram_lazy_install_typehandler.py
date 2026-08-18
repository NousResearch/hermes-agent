"""Regression test for the Telegram lazy-install rebind path (#85272).

When ``plugins.platforms.telegram.adapter`` is imported before
python-telegram-bot is available, the top-level ``except ImportError`` block
binds every SDK symbol to a placeholder. ``check_telegram_requirements()`` then
lazy-installs the package and re-imports, rebinding those module globals.

Any symbol the rebind misses stays a placeholder while ``TELEGRAM_AVAILABLE``
flips to ``True``, so the adapter believes the SDK is present and fails at the
first use of that symbol. ``TypeHandler`` was missed, and handler registration
raised ``TypeError: Any cannot be instantiated`` — see #85272.

The test drives the real transition rather than asserting on one name, so it
also catches the next symbol that gets dropped from the rebind list.
"""

import sys
import types
from typing import Any

import pytest


# Symbols the fallback block binds to a placeholder, paired with the fake the
# rebind is expected to install. Keep in sync with the `except ImportError`
# block in plugins/platforms/telegram/adapter.py.
PLACEHOLDER = Any


@pytest.fixture
def fake_telegram_sdk(monkeypatch):
    """Install a fake python-telegram-bot into sys.modules.

    ``check_telegram_requirements()`` runs its imports at call time, so seeding
    sys.modules is enough to make them resolve without touching the filesystem
    or the network.
    """
    fakes = {
        name: type(f"Fake{name}", (), {})
        for name in (
            "Update",
            "Bot",
            "Message",
            "InlineKeyboardButton",
            "InlineKeyboardMarkup",
            "LinkPreviewOptions",
            "Application",
            "CommandHandler",
            "CallbackQueryHandler",
            "MessageHandler",
            "TypeHandler",
            "HTTPXRequest",
        )
    }

    telegram_pkg = types.ModuleType("telegram")
    for name in (
        "Update",
        "Bot",
        "Message",
        "InlineKeyboardButton",
        "InlineKeyboardMarkup",
        "LinkPreviewOptions",
    ):
        setattr(telegram_pkg, name, fakes[name])

    ext_mod = types.ModuleType("telegram.ext")
    for name in (
        "Application",
        "CommandHandler",
        "CallbackQueryHandler",
        "MessageHandler",
        "TypeHandler",
    ):
        setattr(ext_mod, name, fakes[name])
    ext_mod.ContextTypes = types.SimpleNamespace(DEFAULT_TYPE=object)
    ext_mod.filters = types.SimpleNamespace()

    constants_mod = types.ModuleType("telegram.constants")
    constants_mod.ParseMode = types.SimpleNamespace(MARKDOWN_V2="MarkdownV2")
    constants_mod.ChatType = types.SimpleNamespace(
        GROUP="group",
        SUPERGROUP="supergroup",
        CHANNEL="channel",
        PRIVATE="private",
    )

    request_mod = types.ModuleType("telegram.request")
    request_mod.HTTPXRequest = fakes["HTTPXRequest"]

    for name, module in {
        "telegram": telegram_pkg,
        "telegram.ext": ext_mod,
        "telegram.constants": constants_mod,
        "telegram.request": request_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    return fakes


def test_lazy_install_rebinds_every_placeholder(monkeypatch, fake_telegram_sdk):
    """Every symbol the fallback stubbed must be rebound by the lazy install."""
    import plugins.platforms.telegram.adapter as adapter

    from tools import lazy_deps

    # The package is already in sys.modules; installing it would be a no-op.
    monkeypatch.setattr(lazy_deps, "ensure", lambda key, prompt=False: None)

    # Reconstruct the imported-before-installed state the fallback leaves behind.
    monkeypatch.setattr(adapter, "TELEGRAM_AVAILABLE", False)
    stubbed = (
        "Update",
        "Bot",
        "Message",
        "InlineKeyboardButton",
        "InlineKeyboardMarkup",
        "Application",
        "CommandHandler",
        "CallbackQueryHandler",
        "TelegramMessageHandler",
        "TypeHandler",
        "ContextTypes",
        "HTTPXRequest",
    )
    for name in stubbed:
        monkeypatch.setattr(adapter, name, PLACEHOLDER)
    for name in ("LinkPreviewOptions", "filters", "ParseMode", "ChatType"):
        monkeypatch.setattr(adapter, name, None)

    assert adapter.check_telegram_requirements() is True
    assert adapter.TELEGRAM_AVAILABLE is True

    survivors = [name for name in stubbed if getattr(adapter, name) is PLACEHOLDER]
    assert not survivors, (
        "lazy install left these bound to the import-time placeholder: "
        f"{survivors}"
    )

    # Spot-check that the rebind wired up the real objects, not just anything
    # that is no longer the placeholder.
    assert adapter.TypeHandler is fake_telegram_sdk["TypeHandler"]
    assert adapter.Update is fake_telegram_sdk["Update"]
    assert adapter.TelegramMessageHandler is fake_telegram_sdk["MessageHandler"]
