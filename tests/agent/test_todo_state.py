import threading
from types import SimpleNamespace

import agent.todo_state as todo_state_module
from agent.todo_state import build_todo_store, persist_todo_store
from hermes_state import SessionDB


class FakeSessionDB:
    def __init__(self, state=None):
        self.state = state
        self.writes = []

    def get_session_todo_state(self, session_id):
        return self.state if session_id == "s1" else None

    def update_session_todo_state(self, session_id, state):
        self.writes.append((session_id, state))
        self.state = state
        return True


class InvertingSessionDB:
    """Delay the first write so an unlocked second writer can overtake it."""

    def __init__(self):
        self.first_started = threading.Event()
        self.release_first = threading.Event()
        self._calls = 0
        self._lock = threading.Lock()
        self.state = None
        self.writes = []

    def update_session_todo_state(self, session_id, state):
        with self._lock:
            self._calls += 1
            call = self._calls
        if call == 1:
            self.first_started.set()
            assert self.release_first.wait(timeout=5)
        self.writes.append((session_id, state))
        self.state = state
        return True


def test_build_todo_store_restores_state_and_persists_later_changes():
    db = FakeSessionDB(
        {
            "revision": 4,
            "todos": [{"id": "build", "content": "Build tray", "status": "completed"}],
            "user_status_overrides": {"build": "completed"},
        }
    )
    agent = SimpleNamespace(_persist_disabled=False, _session_db=db, session_id="s1")

    store = build_todo_store(agent)
    agent._todo_store = store
    assert persist_todo_store(agent) is True
    assert db.writes == []  # restored state is already durable; no UI-thread rewrite
    assert store.snapshot_state()["generation"] == 4  # pre-generation sidecar upgrade
    store.update_status("build", "pending", actor="user")

    assert store.read()[0]["status"] == "pending"
    assert db.writes[-1][0] == "s1"
    assert db.writes[-1][1]["todos"][0]["status"] == "pending"


def test_build_todo_store_keeps_an_intentionally_empty_durable_snapshot():
    db = FakeSessionDB({"revision": 5, "todos": [], "user_status_overrides": {}})
    agent = SimpleNamespace(_persist_disabled=False, _session_db=db, session_id="s1")

    store = build_todo_store(agent)

    assert store.read() == []
    assert store.needs_history_reconciliation is False


def test_build_todo_store_seeds_a_new_branch_from_parent_snapshot():
    db = FakeSessionDB()
    agent = SimpleNamespace(_persist_disabled=False, _session_db=db, session_id="branch")
    parent_state = {
        "revision": 7,
        "todos": [{"id": "build", "content": "Build tray", "status": "completed"}],
        "user_status_overrides": {"build": "completed"},
        "pending_user_notices": [],
    }

    store = build_todo_store(agent, fallback_state=parent_state)

    assert store.read()[0]["status"] == "completed"
    assert db.writes[-1][0] == "branch"
    assert db.writes[-1][1]["generation"] == 7


def test_persist_todo_store_follows_a_rotated_session_id():
    db = FakeSessionDB()
    agent = SimpleNamespace(_persist_disabled=False, _session_db=db, session_id="s1")
    store = build_todo_store(agent)
    agent._todo_store = store
    store.write([{"id": "1", "content": "Task", "status": "pending"}])

    agent.session_id = "s2"
    assert persist_todo_store(agent) is True

    assert db.writes[-1][0] == "s2"
    assert db.writes[-1][1]["todos"][0]["id"] == "1"


def test_replaced_store_cannot_write_old_state_into_a_rotated_session():
    db = FakeSessionDB()
    agent = SimpleNamespace(_persist_disabled=False, _session_db=db, session_id="s1")
    old_store = build_todo_store(agent)
    agent._todo_store = old_store
    old_store.write([{"id": "old", "content": "Old session task", "status": "pending"}])
    db.writes.clear()

    agent.session_id = "s2"
    current_store = build_todo_store(agent)
    agent._todo_store = current_store

    old_store.write([{"id": "late", "content": "Late old write", "status": "pending"}])

    assert db.writes == []
    assert current_store.read() == []


def test_concurrent_persistence_cannot_let_an_older_snapshot_win(monkeypatch):
    db = InvertingSessionDB()
    agent = SimpleNamespace(_persist_disabled=False, _session_db=db, session_id="s1")
    older = {"generation": 1, "revision": 1, "todos": []}
    newer = {"generation": 2, "revision": 2, "todos": []}
    second_ready = threading.Event()
    real_key = todo_state_module._persistence_key

    def signal_before_lock(session_id, state):
        if state is newer:
            second_ready.set()
        return real_key(session_id, state)

    monkeypatch.setattr(todo_state_module, "_persistence_key", signal_before_lock)

    first = threading.Thread(target=persist_todo_store, args=(agent, older))
    second = threading.Thread(target=persist_todo_store, args=(agent, newer))
    first.start()
    assert db.first_started.wait(timeout=5)
    second.start()
    # Both implementations have prepared the newer payload. The broken one can
    # now overtake the blocked DB write; the fixed one waits on the agent lock.
    assert second_ready.wait(timeout=5)
    db.release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive() and not second.is_alive()
    assert db.state == newer
    assert agent._todo_state_persist_generation == ("s1", 2)


def test_persist_rejects_a_non_mapping_snapshot():
    db = FakeSessionDB()
    agent = SimpleNamespace(_persist_disabled=False, _session_db=db, session_id="s1")

    assert persist_todo_store(agent, ["not", "a", "snapshot"]) is False
    assert db.writes == []


def test_real_session_db_resume_preserves_user_completion(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="cli")
    try:
        agent = SimpleNamespace(_persist_disabled=False, _session_db=db, session_id="s1")
        store = build_todo_store(agent)
        agent._todo_store = store
        store.write([{"id": "build", "content": "Build tray", "status": "in_progress"}])
        store.update_status("build", "completed", actor="user")

        resumed_agent = SimpleNamespace(_persist_disabled=False, _session_db=db, session_id="s1")
        resumed = build_todo_store(resumed_agent)

        assert resumed.read()[0]["status"] == "completed"
    finally:
        db.close()


def test_build_todo_store_degrades_to_memory_when_persistence_is_disabled():
    db = FakeSessionDB()
    agent = SimpleNamespace(_persist_disabled=True, _session_db=db, session_id="s1")

    store = build_todo_store(agent)
    store.write([{"id": "1", "content": "Task", "status": "pending"}])

    assert store.read()[0]["id"] == "1"
    assert db.writes == []
