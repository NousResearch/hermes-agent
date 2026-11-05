"""Reads never wait for the writer lock.

The gateway shares ONE ``StoreHolder`` per state.db across every agent. A write holds ``holder._lock`` for its whole transaction, so a read that took the same lock would convoy every concurrent turn's persistence behind it. Reads go through ``_read_ctx()`` (a pooled WAL reader, no lock) instead.

Each case starts a write that holds the lock open, runs one read verb while it is held, and requires the read to finish before the write is let go. ``state_meta.get`` is the one read that stays on the writer lock by design (rebuild progress is read on the writer's connection before its write transaction, where a WAL reader would not see it); it is the harness's own check that a locked read is caught.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

import pytest

from persistence import SessionId
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession

SID = SessionId("s-read")
# A read that waited for the writer lock is still blocked when this runs out.
_DEADLINE_S = 5.0

READS: dict[str, Callable[[Any, Any], Any]] = {
    "sessions.get": lambda store, admin: store.sessions.get(SID),
    "sessions.list_row": lambda store, admin: store.sessions.list_row(SID),
    "sessions.is_kanban_owned": lambda store, admin: store.sessions.is_kanban_owned(SID),
    "sessions.cron_run_identity": lambda store, admin: store.sessions.cron_run_identity(SID),
    "sessions.list_cron_job_runs": lambda store, admin: store.sessions.list_cron_job_runs("job"),
    "sessions.list_skill_scaffolded_sessions": lambda store, admin: store.sessions.list_skill_scaffolded_sessions(),
    "sessions.distinct_session_cwds": lambda store, admin: store.sessions.distinct_session_cwds(),
    "messages.read_transcript": lambda store, admin: store.messages.read_transcript(SID),
    "messages.read_conversation": lambda store, admin: store.messages.read_conversation(SID),
    "messages.count_all": lambda store, admin: store.messages.count_all(),
    "messages.display_count": lambda store, admin: store.messages.display_count(SID),
    "messages.active_watermark": lambda store, admin: store.messages.active_watermark(SID),
    "search.search_messages": lambda store, admin: store.search.search_messages("hello"),
    "search.search_index_rebuild_status": lambda store, admin: store.search.search_index_rebuild_status(),
    "maintenance.list_prune_candidates": lambda store, admin: store.maintenance.list_prune_candidates(),
    "maintenance.list_idle_archive_candidates": lambda store, admin: store.maintenance.list_idle_archive_candidates(0),
    "state_meta.list_prefix": lambda store, admin: store.state_meta.list_prefix("k"),
    "admin.freelist_ratio": lambda store, admin: admin.freelist_ratio(),
    "admin.fts_optimize_available": lambda store, admin: admin.fts_optimize_available(),
}


@pytest.fixture
def stores(backend, scope_a):
    store = backend.open(scope_a)
    store.sessions.create(NewSession(session_id=SID, source="cli"))
    store.messages.append(SID, NewMessage(role="user", content="hello there"))
    store.state_meta.set("k", "v")
    return store, backend.open_store_admin(scope_a, read_only=False)


def _finishes_under_a_held_write(store, read: Callable[[], Any]) -> bool:
    """Whether ``read`` returns while a write on ``store``'s holder holds the writer lock; the write ends either way."""
    holding, release = threading.Event(), threading.Event()

    def write(conn):
        holding.set()
        release.wait(_DEADLINE_S * 2)

    writer = threading.Thread(target=store.holder()._execute_write, args=(write,), daemon=True)
    writer.start()
    assert holding.wait(_DEADLINE_S), "the write never began"
    outcome: list = []
    reader = threading.Thread(target=lambda: outcome.append(read()), daemon=True)
    try:
        reader.start()
        reader.join(_DEADLINE_S)
        finished = not reader.is_alive()
    finally:
        release.set()
        writer.join(_DEADLINE_S)
        reader.join(_DEADLINE_S)
    assert outcome, "the read raised"
    return finished


@pytest.mark.parametrize("name", sorted(READS))
def test_a_read_finishes_while_a_write_holds_the_writer_lock(stores, name):
    store, admin = stores
    assert _finishes_under_a_held_write(store, lambda: READS[name](store, admin)), f"{name} waited for the writer lock"


def test_a_read_on_the_writer_lock_is_caught(stores):
    store, _admin = stores
    assert _finishes_under_a_held_write(store, lambda: store.state_meta.get("k")) is False
