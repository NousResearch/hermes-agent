"""Regression coverage for opt-in Telegram edited-update suppression."""

import sys
from unittest.mock import MagicMock

import pytest

# If telegram was mocked (e.g. by gateway conftest collection), restore real telegram if installed
if "telegram" in sys.modules and not hasattr(sys.modules["telegram"], "__file__"):
    for _m in list(sys.modules.keys()):
        if _m == "telegram" or _m.startswith("telegram."):
            del sys.modules[_m]
    if "plugins.platforms.telegram.adapter" in sys.modules:
        del sys.modules["plugins.platforms.telegram.adapter"]

pytest.importorskip("telegram", reason="python-telegram-bot not installed")
from telegram import Update

from gateway.config import Platform, PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter, _apply_yaml_config


_HANDLER_CASES = (
    (0, {"text": "Corrected question"}),
    (1, {"text": "/help", "entities": [{"type": "bot_command", "offset": 0, "length": 5}]}),
    (2, {"location": {"latitude": 1.0, "longitude": 2.0}}),
    (3, {"photo": [{"file_id": "photo-id", "file_unique_id": "photo-unique", "width": 1, "height": 1}]}),
)


def _handlers(*, ignore_edited_messages: bool):
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(
        enabled=True, token="test-token", extra={"ignore_edited_messages": ignore_edited_messages},
    )
    for name in (
        "_handle_text_message", "_handle_command", "_handle_location_message",
        "_handle_media_message", "_handle_callback_query", "_on_platform_update",
    ):
        setattr(adapter, name, object())
    app = MagicMock()
    adapter._register_handlers(app)
    return [call.args[0] for call in app.add_handler.call_args_list[:4]]


def _update(*, update_key: str, message: dict) -> Update:
    payload = {
        "message_id": 1,
        "date": 0,
        "chat": {"id": 1, "type": "private"},
        "from": {"id": 2, "is_bot": False, "first_name": "Owner"},
        **message,
    }
    return Update.de_json({"update_id": 2, update_key: payload}, None)


@pytest.mark.parametrize("handler_index,message", _HANDLER_CASES)
@pytest.mark.parametrize("edited_key", ("edited_message", "edited_channel_post"))
def test_edited_updates_are_ignored_by_every_inbound_handler_when_enabled(handler_index, message, edited_key):
    edited = _update(update_key=edited_key, message=message)
    handler = _handlers(ignore_edited_messages=True)[handler_index]

    assert not handler.check_update(edited)


@pytest.mark.parametrize("handler_index,message", _HANDLER_CASES)
def test_normal_updates_and_default_behavior_are_unchanged(handler_index, message):
    normal = _update(update_key="message", message=message)
    edited = _update(update_key="edited_message", message=message)
    handler = _handlers(ignore_edited_messages=False)[handler_index]

    assert handler.check_update(normal)
    assert handler.check_update(edited)


def test_yaml_telegram_option_reaches_adapter_extra(monkeypatch):
    """The profile setting is not merely an environment-variable escape hatch."""
    monkeypatch.delenv("TELEGRAM_IGNORE_EDITED_MESSAGES", raising=False)

    extras = _apply_yaml_config({}, {"ignore_edited_messages": True})
    assert extras == {"ignore_edited_messages": True}

    extras_false = _apply_yaml_config({}, {"ignore_edited_messages": False})
    assert extras_false == {"ignore_edited_messages": False}


def test_adapter_effective_value_resolution(monkeypatch):
    """Verify that _telegram_ignore_edited_messages resolves correctly from config extra and env."""
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM

    # 1. Config extra True wins over env False
    monkeypatch.setenv("TELEGRAM_IGNORE_EDITED_MESSAGES", "false")
    adapter.config = PlatformConfig(enabled=True, token="tok", extra={"ignore_edited_messages": True})
    assert adapter._telegram_ignore_edited_messages() is True

    # 2. Config extra False wins over env True
    monkeypatch.setenv("TELEGRAM_IGNORE_EDITED_MESSAGES", "true")
    adapter.config = PlatformConfig(enabled=True, token="tok", extra={"ignore_edited_messages": False})
    assert adapter._telegram_ignore_edited_messages() is False

    # 3. Falls back to env var when not in extra
    adapter.config = PlatformConfig(enabled=True, token="tok", extra={})
    monkeypatch.setenv("TELEGRAM_IGNORE_EDITED_MESSAGES", "true")
    assert adapter._telegram_ignore_edited_messages() is True
    monkeypatch.setenv("TELEGRAM_IGNORE_EDITED_MESSAGES", "false")
    assert adapter._telegram_ignore_edited_messages() is False

    # 4. Default is False when neither extra nor env is set
    monkeypatch.delenv("TELEGRAM_IGNORE_EDITED_MESSAGES", raising=False)
    assert adapter._telegram_ignore_edited_messages() is False
