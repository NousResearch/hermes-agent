"""Regression for #127430 — core Telegram handlers must not treat Business updates as owner DMs.

Without a Business plugin, a ``business_message`` the owner writes to a contact matched the core text
handler (``effective_message``), passed the allowlist (sender = owner) and was answered to the contact.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

telegram = pytest.importorskip("telegram")

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


class _PtbApp:
    def __init__(self):
        self.handlers: dict = {}

    def add_handler(self, handler, group=0):
        self.handlers.setdefault(group, []).append(handler)


def _update(text: str, *, business: bool, edited: bool = False) -> "telegram.Update":
    now = datetime.now(timezone.utc)
    msg = telegram.Message(
        message_id=1, date=now, chat=telegram.Chat(id=222, type="private"),
        from_user=telegram.User(id=111, first_name="Owner", is_bot=False), text=text,
        business_connection_id="conn" if business else None,
    )
    if not business:
        return telegram.Update(update_id=1, message=msg)
    field = "edited_business_message" if edited else "business_message"
    return telegram.Update(update_id=1, **{field: msg})


def _core_matches(update) -> bool:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="t", extra={}))
    app = _PtbApp()
    adapter._register_handlers(app)
    return any(h.check_update(update) for h in app.handlers[0])


@pytest.mark.parametrize("text", ["hello", "/status"])
def test_core_handlers_skip_business_messages(text):
    assert _core_matches(_update(text, business=False))  # control: the same message as a plain DM is handled
    assert not _core_matches(_update(text, business=True))
    assert not _core_matches(_update(text, business=True, edited=True))
