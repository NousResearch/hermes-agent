"""Real engine identity writes and DB-backed rotation share one local guard.

No model execution or gateway acceptance is claimed by these engine-boundary tests.
"""
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB
from run_agent import AIAgent


def engine(db=None):
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = "parent"
    agent._session_db = db
    agent.context_compressor = SimpleNamespace()
    agent._memory_manager = None
    return agent


def test_direct_assignments_are_serialized_and_revisions_never_rewind():
    agent = engine()
    started = Event()
    def rotate():
        started.set()
        agent.session_id = "child"
        agent.session_id = "parent"
    with ThreadPoolExecutor(max_workers=1) as pool:
        with agent.session_identity_guard():
            revision = agent.session_identity_revision
            agent.session_id = "parent"  # reentrant no-op preserves revision
            pending = pool.submit(rotate)
            assert started.wait(5)
            assert agent.session_id == "parent" and agent.session_identity_revision == revision
        pending.result(timeout=5)
    with agent.session_identity_guard():
        assert agent.session_id == "parent" and agent.session_identity_revision == revision + 2
        assert vars(agent)["session_id"] == "parent"  # preserve existing attribute storage


@pytest.mark.parametrize("writer", ["compression_adoption", "persistence_adoption", "publication", "failed_publication"])
def test_db_rotation_and_adoption_are_guarded_through_commit(monkeypatch, tmp_path, writer):
    from agent import conversation_compression as compression
    from agent import session_persistence as persistence

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("parent", source="tui")
    history = [{"role": "user", "content": "question"}, {"role": "assistant", "content": "summary"}]
    agent = engine(db)
    entered, release = Event(), Event()
    # Park in the real DB operation, inside the production identity transaction.
    if "adoption" in writer:
        db.publish_compression_child(parent_session_id="parent", child_session_id="child", source="tui",
                                     model="test", messages=history, system_prompt="", require_compression_lease=False)
        original = type(db).get_compression_tip
        def tip(self, sid):
            entered.set()
            assert release.wait(5)
            return original(self, sid)
        monkeypatch.setattr(type(db), "get_compression_tip", tip)
        if writer == "compression_adoption":
            mutate = lambda: compression._adopt_live_compression_child(agent, db, "parent")
        else:
            mutate = lambda: persistence._db_flush_adopt_compression_tip(agent)
    else:
        agent.model = "test"
        agent._session_init_model_config = {}
        agent._flush_messages_to_session_db = lambda *a, **k: None
        monkeypatch.setattr(compression, "mint_session_id", lambda: "child")
        monkeypatch.setattr(compression, "_carry_session_state_to_child", lambda *a: None)
        original = db.publish_compression_child
        def publish(**kwargs):
            entered.set()
            assert release.wait(5)
            if writer == "failed_publication":
                raise RuntimeError("publication refused")
            return original(**kwargs)
        monkeypatch.setattr(db, "publish_compression_child", publish)
        lease = SimpleNamespace(holder=None, ttl=300, watermark=None)
        mutate = lambda: compression._publish_rotated_compaction(
            agent, history, history, new_system_prompt="sys", lease=lease,
            old_session_id="parent", compressed_user_turn_outcome="already_present")
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(mutate)
            try:
                assert entered.wait(5)
                guard = agent.session_identity_guard()
                acquired = guard.acquire(blocking=False)
                if acquired:
                    guard.release()
                assert not acquired, "identity must be unavailable while its DB transaction is settling"
                assert agent.session_id == "parent"
            finally:
                release.set()
            if writer == "failed_publication":
                with pytest.raises(RuntimeError, match="publication refused"):
                    pending.result(timeout=5)
            else:
                pending.result(timeout=5)
        guard = agent.session_identity_guard()
        assert guard.acquire(blocking=False), "failed or successful transitions must release the guard"
        try:
            if writer == "failed_publication":
                assert agent.session_id == "parent" and db.get_session("child") is None
            else:
                assert agent.session_id == "child" and db.get_session("child") is not None
        finally:
            guard.release()
    finally:
        db.close()
