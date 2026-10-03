"""Observed group chatter must carry an epoch-float timestamp the write path accepts.

The Telegram and Yuanbao observe paths stamped ``datetime.now(tz=timezone.utc).isoformat()``
onto the transcript row. ``hermes_state_messages._coerce_timestamp`` only trusts numbers,
numeric strings and ``datetime``, so every observed message was persisted with a silently
substituted fallback and logged one ``Ignoring corrupt message timestamp`` WARNING per message
(hundreds a day in an observed group). The contract asserted here is the relationship: the
timestamp an observe path writes is the timestamp the row ends up with.
"""

import dataclasses
import logging
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from hermes_state import SessionDB


class _CapturingStore:
    """Minimal SessionStore stand-in that records the entry an observe path writes."""

    def __init__(self):
        self.entries = []

    def get_or_create_session(self, source):
        return SimpleNamespace(session_id="observed-session")

    def append_to_transcript(self, session_id, message, skip_db=False):
        self.entries.append(message)


def _group_source():
    return SessionSource(
        platform=Platform.TELEGRAM, chat_id="-100", chat_type="group",
        user_id="4242", user_name="Igor", thread_id="7")


def _telegram_observed_entry():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter.__new__(TelegramAdapter)
    store = _CapturingStore()
    adapter._session_store = store
    source = _group_source()
    event = MessageEvent(
        text="just group chatter", message_type=MessageType.TEXT, source=source,
        user_id=source.user_id, user_name=source.user_name, message_id="991")
    adapter._observe_unmentioned_group_message(
        SimpleNamespace(chat=SimpleNamespace(id=-100)), MessageType.TEXT, event=event)
    assert len(store.entries) == 1
    return store.entries[0]


def _yuanbao_observed_entry():
    from gateway.platforms.yuanbao import GroupAtGuardMiddleware

    store = _CapturingStore()
    adapter = SimpleNamespace(name="yuanbao", session_store=store, _session_store=store)
    source = dataclasses.replace(_group_source(), platform=Platform.YUANBAO)
    GroupAtGuardMiddleware._observe_group_message(
        adapter, source, "Igor", "just group chatter",
        ctx=SimpleNamespace(adapter=adapter, event=None), msg_id="991")
    assert len(store.entries) == 1
    return store.entries[0]


@pytest.mark.parametrize(
    "entry_factory", [_telegram_observed_entry, _yuanbao_observed_entry],
    ids=["telegram", "yuanbao"])
def test_observed_group_message_timestamp_survives_the_write_path(entry_factory, tmp_path, caplog):
    entry = entry_factory()
    written = entry["timestamp"]

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("observed-session", "telegram")
        with caplog.at_level(logging.WARNING, logger="hermes_cli.timefmt"):
            db.append_message(
                session_id="observed-session", role=entry["role"], content=entry["content"],
                observed=True, timestamp=written)
        stored = db.get_messages("observed-session")[-1]["timestamp"]
    finally:
        db.close()

    # The row keeps the adapter's own stamp instead of a substituted fallback ...
    assert stored == pytest.approx(float(written), abs=1e-6)
    # ... and the write path does not have to reject it.
    assert "Ignoring corrupt message timestamp" not in caplog.text
