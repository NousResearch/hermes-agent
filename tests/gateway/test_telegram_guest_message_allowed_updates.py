"""Regression test: guest_message must be requested via allowed_updates.

PTB's ``Update.ALL_TYPES`` is a hardcoded enumeration of update kinds PTB has
a typed field for. Bot API 10.0's ``guest_message`` isn't one of them, so
passing ``Update.ALL_TYPES`` verbatim as ``allowed_updates`` tells Telegram to
send every OTHER update type and silently omit guest_message -- Telegram
drops it server-side, before it ever reaches this process. The
``_handle_guest_message_update`` TypeHandler registers fine and no exception
is ever raised; the bot just never receives a guest @mention, from any chat,
indistinguishable from an outage. There was no test coverage for
``allowed_updates`` at all before this, which is how the bug shipped.
"""
from __future__ import annotations

import inspect
import re

import plugins.platforms.telegram.adapter as tg_adapter


def test_allowed_updates_constant_includes_guest_message():
    assert "guest_message" in tg_adapter._ALLOWED_UPDATES_WITH_GUEST
    # And still covers everything PTB itself knows about, so this isn't a
    # narrower request that drops some other update type instead.
    for update_type in tg_adapter.Update.ALL_TYPES:
        assert update_type in tg_adapter._ALLOWED_UPDATES_WITH_GUEST


def test_no_call_site_uses_bare_update_all_types_for_allowed_updates():
    """Bug-class contract: every ``allowed_updates=`` argument in the adapter
    must route through ``_ALLOWED_UPDATES_WITH_GUEST``, not
    ``Update.ALL_TYPES`` directly -- a new call site written the naive way
    silently drops guest_message again, with no error anywhere."""
    src = inspect.getsource(tg_adapter)
    bare = [
        (i + 1, line.strip())
        for i, line in enumerate(src.splitlines())
        if re.search(r"allowed_updates\s*=\s*Update\.ALL_TYPES\b", line)
    ]
    assert not bare, f"allowed_updates call sites bypassing the guest_message fix: {bare}"
