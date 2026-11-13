"""Observed group chatter keeps the timestamp the observe path stamps on it.

The Telegram observe path stamped ``datetime.now(tz=UTC).isoformat()`` onto the transcript row, but
``hermes_cli.timefmt.coerce_epoch`` trusts only numbers, numeric strings and ``datetime``: every
observed row was persisted with a substituted time and logged ``Ignoring corrupt message
timestamp`` (one WARNING per observed message).
"""

import logging
import time
from types import SimpleNamespace

from hermes_agent.gateway.config import GatewayConfig, Platform
from hermes_agent.gateway.platforms.event import MessageEvent, MessageType
from hermes_agent.gateway.session import SessionSource, SessionStore
from hermes_agent.hermes_state import SessionDB


def test_observed_group_message_is_persisted_with_its_own_timestamp(tmp_path, caplog):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    db = SessionDB(tmp_path / "state.db")
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    store._db = db  # production: the routing store writes the profile's state.db
    adapter = TelegramAdapter.__new__(TelegramAdapter)
    adapter._session_store = store
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-100", chat_type="group",
                           user_id="4242", user_name="Igor", thread_id="7")
    event = MessageEvent(text="just group chatter", message_type=MessageType.TEXT, source=source,
                         user_id=source.user_id, user_name=source.user_name, message_id="991")
    try:
        caplog.set_level(logging.WARNING, logger="hermes_cli.timefmt")
        before = time.time()
        adapter._observe_unmentioned_group_message(
            SimpleNamespace(chat=SimpleNamespace(id=-100)), MessageType.TEXT, event=event)
        after = time.time()
        session_id = store.get_or_create_session(adapter._telegram_group_observe_shared_source(source)).session_id
        rows = [m for m in db.get_messages(session_id) if m.get("observed")]
    finally:
        db.close()

    assert [r["content"] for r in rows] == ["[Igor|4242]\njust group chatter"]
    assert before <= float(rows[0]["timestamp"]) <= after
    # The stamp was trusted as written, not rejected and replaced by a fallback.
    assert "Ignoring corrupt message timestamp" not in caplog.text
