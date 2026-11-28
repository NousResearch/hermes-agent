"""#154994: a turn moves its route only onto a compression continuation of its own session.

An async subagent's session came back as the agent's session id after a gateway turn. Both split
syncs treated the change as compression and saved the route onto the subagent, and the peer record
then refused ``admission_conflict`` with the route already saved: every later lookup, resume and
delivery for the conversation was refused. These tests run the shared checks against a real store
(through the storage interface) and the agent-result sync in ``_hmwa_shape_agent_response``.
"""
import threading
import time
from types import SimpleNamespace

import pytest

from gateway.run_turn import GatewayTurnMixin
from gateway.session_split import is_compression_continuation, restore_route
from persistence.conversation.sessions import NewSession, SessionLineageKind
from persistence.core.errors import AuthorityRefusal
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_seed import seed_messages
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle

KEY = "agent:main:telegram:dm:12345"


def _create(db, session_id, source, parent=None, *, delegate=False):
    db.sessions.create(NewSession(
        session_id=session_id, source=source, parent_session_id=parent,
        lineage_kind=SessionLineageKind.DELEGATE if delegate else SessionLineageKind.NONE))


def _say(db, session_id, role, content):
    seed_messages(db, session_id, {"role": role, "content": content})


@pytest.fixture
def db(tmp_path):
    db = storage_handle(tmp_path)
    base = int(time.time()) - 10_000
    _create(db, "root", "telegram")
    _say(db, "root", "user", "before compression")
    db.sessions.end("root", "compression")
    _create(db, "cont", "telegram", "root")
    _say(db, "cont", "assistant", "after compression")
    # Async batch children as #154994 found them: source=subagent, no _delegate_from marker.
    _create(db, "subagent", "subagent", "root")
    _say(db, "subagent", "assistant", "subagent report")
    _create(db, "delegate", "subagent", "root", delegate=True)

    def backdate(conn):
        # Subagents hanging off a compressed parent were spawned before the compression ended it.
        conn.execute("UPDATE sessions SET started_at = ?, ended_at = ? WHERE id = 'root'", (base, base + 50))
        for started, sid in ((base + 10, "subagent"), (base + 20, "delegate"), (base + 100, "cont")):
            conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (started, sid))
    raw_holder(db)._execute_write(backdate)
    return db


class _Store:
    def __init__(self, db=None, session_id="root"):
        self.db = db
        self.entry = SimpleNamespace(session_key=KEY, session_id=session_id)
        self._lock = threading.Lock()
        self.saved = []

    def _persistence_for_key(self, session_key):
        assert session_key == KEY
        return self.db

    def _save(self):
        assert self._lock.locked(), "_save runs under the routing lock"
        self.saved.append(self.entry.session_id)


def test_only_the_compression_child_continues_the_session(db):
    store = _Store(db)
    assert is_compression_continuation(store, KEY, "root", "cont") is True
    assert is_compression_continuation(store, KEY, "root", "subagent") is False
    assert is_compression_continuation(store, KEY, "root", "delegate") is False


def test_an_unverifiable_split_keeps_the_previous_behaviour(db):
    assert is_compression_continuation(_Store(db), KEY, "root", "no-row-yet") is True
    assert is_compression_continuation(_Store(None), KEY, "root", "subagent") is True
    assert is_compression_continuation(SimpleNamespace(), KEY, "root", "subagent") is True


def test_restore_route_rolls_back_under_the_lock():
    store = _Store(session_id="cont")
    restore_route(store, store.entry, published_id="cont", restored_id="root")
    assert store.entry.session_id == "root"
    assert store.saved == ["root"]


def test_restore_route_never_overwrites_a_concurrent_route_change():
    store = _Store(session_id="after-new")  # a /new moved the route before the rollback ran
    restore_route(store, store.entry, published_id="cont", restored_id="root")
    assert store.entry.session_id == "after-new"
    assert store.saved == []


def test_a_live_markerless_subagent_never_becomes_the_compression_tip(db):
    # Even live and most recently active, a subagent child is not the compressed parent's continuation.
    _say(db, "subagent", "assistant", "still running")
    assert db.lineage.tip("root") == "cont"
    assert is_compression_continuation(_Store(db), KEY, "root", "subagent") is False


def test_a_subagents_own_compression_chain_still_walks(db):
    _create(db, "sub-a", "subagent")
    db.sessions.end("sub-a", "compression")
    _create(db, "sub-b", "subagent", "sub-a")
    assert db.lineage.tip("sub-a") == "sub-b"


def test_a_sourceless_parents_compression_child_still_walks(db):
    # _compression_child_source falls back to the agent's platform when the parent row has no source.
    _create(db, "legacy", "telegram")
    raw_holder(db)._execute_write(lambda conn: conn.execute("UPDATE sessions SET source = '' WHERE id = 'legacy'"))
    db.sessions.end("legacy", "compression")
    _create(db, "legacy-next", "subagent", "legacy")
    assert db.lineage.tip("legacy") == "legacy-next"


def test_the_resume_walk_skips_a_markerless_subagent_child(db):
    # The lease-wait reload adopts resolve_resume_session_id(); before the fix it walked into the
    # newest child with messages, the async subagent, and the turn ran on that session.
    assert db.lineage.resolve_resume_target("cont") == "cont"
    _create(db, "chat", "telegram")
    _say(db, "chat", "user", "hi")
    _create(db, "chat-subagent", "subagent", "chat")
    _say(db, "chat-subagent", "assistant", "subagent report")
    assert db.lineage.resolve_resume_target("chat") == "chat"


def test_a_subagents_own_continuation_stays_walkable(db):
    _create(db, "sub-root", "subagent")
    _create(db, "sub-next", "subagent", "sub-root")
    _say(db, "sub-next", "assistant", "later turn")
    assert db.lineage.resolve_resume_target("sub-root") == "sub-next"


class _Runner(GatewayTurnMixin):
    def __init__(self, store, *, refuse_peer=False):
        self.session_store = store
        self.refuse_peer = refuse_peer
        self.lease_rebinds = []
        self.async_session_store = SimpleNamespace(
            clear_resume_pending=self._noop, _save=self._save, _record_gateway_session_peer=self._record)
        self._sync_telegram_topic_binding = lambda *_a, **_k: None

    async def _noop(self, *_a, **_k):
        return None

    async def _save(self):
        with self.session_store._lock:
            self.session_store._save()

    async def _record(self, session_id, session_key, source):
        if self.refuse_peer:
            raise AuthorityRefusal("admission_conflict")

    async def _clear_restart_failure_count(self, *_a, **_k):
        return None

    def _rebind_turn_lease(self, _key, _generation, session_id):
        self.lease_rebinds.append(session_id)


async def _shape(runner, new_session_id):
    agent_result = {"final_response": "done", "messages": [], "api_calls": 1, "session_id": new_session_id}
    source = SimpleNamespace(chat_id="12345", platform=SimpleNamespace(value="telegram"))
    return await runner._hmwa_shape_agent_response(
        agent_result, source, history=[], session_entry=runner.session_store.entry, session_key=KEY,
        _quick_key=KEY, run_generation=0, _run_start_session_id="root", _platform_name="telegram",
        _msg_start_time=0.0,
    )


@pytest.mark.asyncio
async def test_agent_result_sync_keeps_the_route_off_another_conversation(db):
    runner = _Runner(_Store(db))
    await _shape(runner, "subagent")
    assert runner.session_store.entry.session_id == "root"
    assert runner.session_store.saved == []
    assert runner.lease_rebinds == []


@pytest.mark.asyncio
async def test_agent_result_sync_still_follows_a_compression_child(db):
    runner = _Runner(_Store(db))
    await _shape(runner, "cont")
    assert runner.session_store.entry.session_id == "cont"
    assert runner.session_store.saved == ["cont"]
    assert runner.lease_rebinds == ["cont"]


@pytest.mark.asyncio
async def test_agent_result_sync_rolls_back_a_refused_publication(db):
    runner = _Runner(_Store(db), refuse_peer=True)
    with pytest.raises(AuthorityRefusal, match="admission_conflict"):
        await _shape(runner, "cont")
    assert runner.session_store.entry.session_id == "root"
    assert runner.session_store.saved == ["cont", "root"]
    assert runner.lease_rebinds == ["cont", "root"]
