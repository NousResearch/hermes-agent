"""Observed group chatter must carry an epoch-float timestamp the write path accepts.

The Yuanbao observe path stamped ``datetime.now(tz=timezone.utc).isoformat()``
onto the transcript row. ``hermes_state_messages._coerce_timestamp`` only trusts numbers,
numeric strings and ``datetime``, so every observed message was persisted with a silently
substituted fallback and logged one ``Ignoring corrupt message timestamp`` WARNING per message
(hundreds a day in an observed group). The contract asserted here is the relationship: the
timestamp an observe path writes is the timestamp the row ends up with.
"""

import logging
from types import SimpleNamespace

import pytest

from gateway.config import Platform
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
        platform=Platform.YUANBAO, chat_id="group:100", chat_type="group",
        user_id="4242", user_name="Igor", thread_id="7")


def _yuanbao_observed_entry():
    from plugins.platforms.yuanbao.adapter import GroupAtGuardMiddleware

    store = _CapturingStore()
    adapter = SimpleNamespace(name="yuanbao", session_store=store, _session_store=store)
    source = _group_source()
    GroupAtGuardMiddleware._observe_group_message(
        adapter, source, "Igor", "just group chatter",
        ctx=SimpleNamespace(adapter=adapter, event=None), msg_id="991")
    assert len(store.entries) == 1
    return store.entries[0]


def test_observed_group_message_timestamp_survives_the_write_path(tmp_path, caplog):
    entry = _yuanbao_observed_entry()
    written = entry["timestamp"]

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("observed-session", "yuanbao")
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
