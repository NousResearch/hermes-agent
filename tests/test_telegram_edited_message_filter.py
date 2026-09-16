import asyncio
import os
import sys
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Import ordering matters: a sibling suite may leave a MOCKED ``telegram`` in sys.modules, which would
# make ``importorskip`` succeed against the stub. So: check the SDK really exists first, and only then
# drop a stub — a PTB-less environment skips this file without mutating other suites' module state.
if sys.modules.get("telegram") is not None and not hasattr(sys.modules["telegram"], "__file__"):
    for _m in [m for m in list(sys.modules) if m == "telegram" or m.startswith("telegram.")]:
        del sys.modules[_m]
    # The adapter may have been imported against the stub's placeholders — re-import it for real.
    sys.modules.pop("plugins.platforms.telegram.adapter", None)

try:
    from telegram import Update
except ImportError:  # pragma: no cover - a stub was present but the real SDK is not installed
    pytest.skip("python-telegram-bot not installed", allow_module_level=True)

from gateway.config import Platform, PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter, _apply_yaml_config


_HANDLER_CASES = (
    (0, {"text": "Corrected question"}),
    (1, {"text": "/help", "entities": [{"type": "bot_command", "offset": 0, "length": 5}]}),
    (2, {"location": {"latitude": 1.0, "longitude": 2.0}}),
    (3, {"photo": [{"file_id": "photo-id", "file_unique_id": "photo-unique", "width": 1, "height": 1}]}),
)

_SENT_AT = 1_700_000_000
_FRESH_EDIT_AT = _SENT_AT + 42  # resident fixes the message they just sent (adding the bot mention)
_STALE_EDIT_AT = _SENT_AT + 3_600  # edit to a long-past message


def _handlers(*, ignore_edited_messages: bool, extra: dict | None = None):
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(
        enabled=True, token="test-token",
        extra={"ignore_edited_messages": ignore_edited_messages, **(extra or {})},
    )
    for name in (
        "_handle_text_message", "_handle_command", "_handle_location_message",
        "_handle_media_message", "_handle_callback_query", "_on_platform_update",
    ):
        setattr(adapter, name, object())
    app = MagicMock()
    adapter._register_handlers(app)
    return [call.args[0] for call in app.add_handler.call_args_list[:4]]


def _update(*, update_key: str, message: dict, edit_date: int | None = None) -> Update:
    payload = {
        "message_id": 1,
        "date": _SENT_AT,
        "chat": {"id": 1, "type": "private"},
        "from": {"id": 2, "is_bot": False, "first_name": "Owner"},
        **message,
    }
    if "business" in update_key:
        payload.setdefault("business_connection_id", "business-connection")
    if edit_date is not None:
        payload["edit_date"] = edit_date
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


@pytest.mark.parametrize("handler_index,message", _HANDLER_CASES)
@pytest.mark.parametrize("edited_key", ("edited_message", "edited_channel_post"))
def test_freshness_ceiling_admits_recent_edits_and_drops_stale_ones(handler_index, message, edited_key):
    """A profile with ``edited_message_max_age_sec`` accepts a just-made correction, never a late edit."""
    handlers = _handlers(ignore_edited_messages=False, extra={"edited_message_max_age_sec": 600})
    handler = handlers[handler_index]

    assert handler.check_update(_update(update_key="message", message=message))
    assert handler.check_update(
        _update(update_key=edited_key, message=message, edit_date=_FRESH_EDIT_AT))
    assert not handler.check_update(
        _update(update_key=edited_key, message=message, edit_date=_STALE_EDIT_AT))


@pytest.mark.parametrize("handler_index,message", _HANDLER_CASES)
@pytest.mark.parametrize("edited_key", ("edited_message", "edited_channel_post"))
def test_freshness_ceiling_is_ignored_when_all_edits_are_dropped(handler_index, message, edited_key):
    handler = _handlers(
        ignore_edited_messages=True, extra={"edited_message_max_age_sec": 600})[handler_index]

    assert handler.check_update(_update(update_key="message", message=message))
    assert not handler.check_update(
        _update(update_key=edited_key, message=message, edit_date=_FRESH_EDIT_AT))


@pytest.mark.parametrize("edited_key", ("edited_message", "edited_channel_post", "edited_business_message"))
def test_media_handler_reads_the_edited_payload(edited_key):
    """A media edit the ceiling admits must not be dropped by the media handler's own None guard.

    The filter admits the update, so the handler has to look at the edited payload the way its three
    sibling handlers already do — otherwise the documented promise silently does not hold for media.
    """
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(
        enabled=True, token="tok", extra={"edited_message_max_age_sec": 600})
    seen = {}

    def _deny(msg):
        seen["msg"] = msg
        return False

    adapter._is_user_authorized_from_message = _deny
    update = _update(update_key=edited_key, message=_HANDLER_CASES[3][1], edit_date=_FRESH_EDIT_AT)

    asyncio.run(TelegramAdapter._handle_media_message(adapter, update, None))

    assert seen.get("msg") is getattr(update, edited_key)


def test_ceiling_boundary_is_inclusive_and_measured_from_the_original_send():
    """Exactly-at-the-ceiling is admitted; one second later is not (the window is ``<=``)."""
    handler = _handlers(ignore_edited_messages=False, extra={"edited_message_max_age_sec": 600})[0]

    assert handler.check_update(
        _update(update_key="edited_message", message={"text": "at the boundary"}, edit_date=_SENT_AT + 600))
    assert not handler.check_update(
        _update(update_key="edited_message", message={"text": "a second late"}, edit_date=_SENT_AT + 601))


def test_filter_fails_closed_on_payloads_it_cannot_age():
    """Unusable date payloads are rejected, never raised — and never silently admitted."""
    from plugins.platforms.telegram.adapter import TelegramFreshEditFilter

    flt = TelegramFreshEditFilter(600)
    assert flt.filter(SimpleNamespace(date="yesterday", edit_date="today")) is False
    assert flt.filter(SimpleNamespace(date=None, edit_date=_SENT_AT)) is False
    assert flt.filter(SimpleNamespace(date=Decimal(1), edit_date=Decimal(2))) is False
    assert flt.filter(SimpleNamespace(date=_SENT_AT)) is True  # not an edit at all


def test_ceiling_does_not_borrow_another_profiles_environment_value(monkeypatch):
    """Under multiplex, a scoped miss must not fall through to the process env (allowlist leak, #72348)."""
    from agent.secret_scope import reset_secret_scope, set_multiplex_active, set_secret_scope

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="tok", extra={})
    monkeypatch.setenv("TELEGRAM_EDITED_MESSAGE_MAX_AGE_SEC", "600")

    # Single profile: the documented environment escape hatch applies.
    assert adapter._telegram_edited_message_max_age_seconds() == 600.0

    set_multiplex_active(True)
    token = set_secret_scope({})  # this profile defines nothing
    try:
        assert os.environ["TELEGRAM_EDITED_MESSAGE_MAX_AGE_SEC"] == "600"
        assert adapter._telegram_edited_message_max_age_seconds() is None
    finally:
        reset_secret_scope(token)
        set_multiplex_active(False)

    # A profile that does define its own value still gets it.
    set_multiplex_active(True)
    token = set_secret_scope({"TELEGRAM_EDITED_MESSAGE_MAX_AGE_SEC": "30"})
    try:
        assert adapter._telegram_edited_message_max_age_seconds() == 30.0
    finally:
        reset_secret_scope(token)
        set_multiplex_active(False)


def test_yaml_telegram_option_reaches_adapter_extra(monkeypatch):
    """The profile setting is not merely an environment-variable escape hatch."""
    monkeypatch.delenv("TELEGRAM_IGNORE_EDITED_MESSAGES", raising=False)

    extras = _apply_yaml_config({}, {"ignore_edited_messages": True})
    assert extras == {"ignore_edited_messages": True}

    extras_false = _apply_yaml_config({}, {"ignore_edited_messages": False})
    assert extras_false == {"ignore_edited_messages": False}

    extras_ceiling = _apply_yaml_config({}, {"edited_message_max_age_sec": 600})
    assert extras_ceiling == {"edited_message_max_age_sec": 600}


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


def test_edited_message_ceiling_resolution(monkeypatch):
    """The ceiling resolves from config extra first, then env; unusable values mean 'no ceiling'."""
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    monkeypatch.delenv("TELEGRAM_EDITED_MESSAGE_MAX_AGE_SEC", raising=False)

    # 1. Unset in both places = no ceiling (upstream behavior)
    adapter.config = PlatformConfig(enabled=True, token="tok", extra={})
    assert adapter._telegram_edited_message_max_age_seconds() is None

    # 2. Config extra wins over env
    monkeypatch.setenv("TELEGRAM_EDITED_MESSAGE_MAX_AGE_SEC", "60")
    adapter.config = PlatformConfig(enabled=True, token="tok", extra={"edited_message_max_age_sec": 600})
    assert adapter._telegram_edited_message_max_age_seconds() == 600.0

    # 3. Env fallback when the extra is absent
    adapter.config = PlatformConfig(enabled=True, token="tok", extra={})
    assert adapter._telegram_edited_message_max_age_seconds() == 60.0

    # 4. Non-numeric, zero, boolean and negative values do not impose a ceiling
    for bad in ("", "later", 0, -1, True):
        adapter.config = PlatformConfig(enabled=True, token="tok", extra={"edited_message_max_age_sec": bad})
        assert adapter._telegram_edited_message_max_age_seconds() is None


def test_inbound_filter_is_the_profiles_policy(monkeypatch):
    """Drop-all, freshness ceiling, and upstream default are three distinguishable behaviors."""
    from telegram.ext import filters

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    monkeypatch.delenv("TELEGRAM_IGNORE_EDITED_MESSAGES", raising=False)
    monkeypatch.delenv("TELEGRAM_EDITED_MESSAGE_MAX_AGE_SEC", raising=False)
    text = {"text": "Corrected question"}

    # Drop-all: even a just-made edit is rejected, ordinary messages still pass.
    adapter.config = PlatformConfig(enabled=True, token="tok", extra={"ignore_edited_messages": True})
    drop_all = adapter._telegram_inbound_update_filter()
    assert drop_all.check_update(_update(update_key="message", message=text))
    assert not drop_all.check_update(_update(update_key="edited_message", message=text, edit_date=_FRESH_EDIT_AT))

    # Ceiling: a recent edit passes, a late one does not.
    adapter.config = PlatformConfig(enabled=True, token="tok", extra={"edited_message_max_age_sec": 600})
    ceiling = adapter._telegram_inbound_update_filter()
    assert ceiling.check_update(_update(update_key="edited_message", message=text, edit_date=_FRESH_EDIT_AT))
    assert not ceiling.check_update(_update(update_key="edited_message", message=text, edit_date=_STALE_EDIT_AT))

    # Neither option: the framework default (no ceiling at all).
    adapter.config = PlatformConfig(enabled=True, token="tok", extra={})
    assert adapter._telegram_inbound_update_filter() is filters.ALL
