"""Todo-store hydration must not rewind the revision clock across compaction.

The desktop task-list panel (``apps/desktop/src/store/todos.ts``
``acceptRevision``) and the gateway's ``tool_progress._session_todo_state``
both treat a todo update whose revision is LOWER than the last one they saw
as a stale replay and drop it. That assumes the revision clock is monotonic
per session.

Context compression breaks the assumption: it archives the ``todo_list``
tool results out of the model-visible history (``_summarize_tool_result``
collapses them to a one-line summary; the synthetic snapshot row carries
items but no revision). A fresh agent built after compaction therefore
hydrates an empty store at revision 0, and every post-compression
``todo_list`` write lands below the client's pre-compaction watermark: the
writes are silently rejected as stale and the archived list resurrects on
the next session activation.

The session DB keeps every todo result — ``SessionDB.get_latest_todo_result``
deliberately includes ``compacted = 1`` rows — so hydration must use it as a
revision floor.
"""

import json
from types import SimpleNamespace

import run_agent


def _todo_result(items, revision):
    return json.dumps({"todos": items, "revision": revision, "summary": {}})


def _assistant_todo_call(call_id):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "todo_list", "arguments": "{}"},
            }
        ],
    }


def _tool_result(call_id, content):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _todo_pair(call_id, items, revision):
    return [_assistant_todo_call(call_id), _tool_result(call_id, _todo_result(items, revision))]


class FakeStore:
    def __init__(self):
        self.calls = []
        self._revision = 0

    def restore(self, items, merge=False, revision=0):
        self.calls.append({"items": items, "merge": merge, "revision": revision})
        self._revision = revision

    def snapshot(self):
        return {"revision": self._revision}


class FakeDB:
    def __init__(self, content):
        self._content = content
        self.calls = 0

    def get_latest_todo_result(self, session_id):
        self.calls += 1
        return self._content


def _agent(store=None, db=None):
    agent = SimpleNamespace(
        session_id="sess-1",
        _session_db=db,
        _todo_store=store or FakeStore(),
        quiet_mode=True,
        log_prefix="",
    )
    # The hydration path delegates to these real methods; bind them to the
    # fake so the test exercises the real pairing/parsing, not a stub of it.
    agent._latest_todo_response = lambda history: run_agent.AIAgent._latest_todo_response(agent, history)
    agent._tool_response_matches_todo_call = (
        lambda history, idx: run_agent.AIAgent._tool_response_matches_todo_call(history, idx)
    )
    agent._todo_db_revision_floor = lambda: run_agent.AIAgent._todo_db_revision_floor(agent)
    return agent


def _hydrate(agent, history):
    run_agent.AIAgent._hydrate_todo_store(agent, history)


# ── preserved behavior ─────────────────────────────────────────────────


def test_hydrate_restores_latest_pair_from_history():
    store = FakeStore()
    agent = _agent(store)
    history = (
        _todo_pair("c1", [{"id": "a", "content": "x", "status": "pending"}], 3)
        + _todo_pair("c2", [{"id": "a", "content": "x", "status": "completed"}], 4)
    )

    _hydrate(agent, history)

    assert store._revision == 4
    assert store.calls[-1]["items"][0]["status"] == "completed"


def test_hydration_survives_db_errors():
    class BoomDB:
        def get_latest_todo_result(self, session_id):
            raise RuntimeError("db locked")

    store = FakeStore()
    agent = _agent(store, BoomDB())
    history = _todo_pair("c1", [], 1)

    _hydrate(agent, history)  # must not raise

    assert store._revision == 1


def test_hydration_swallows_store_failures():
    class BoomStore:
        def snapshot(self):
            raise RuntimeError("boom")

        def restore(self, *args, **kwargs):
            raise RuntimeError("boom")

    agent = _agent(BoomStore(), FakeDB(_todo_result([], 3)))

    _hydrate(agent, [{"role": "user", "content": "hi"}])  # must not raise


# ── the bug: revision floor across compaction ──────────────────────────


def test_revision_floor_survives_compression_wiped_history():
    """Post-compaction: history has no todo pairs left; the DB still has the row.

    Without the floor the store hydrates at revision 0, so the next real
    ``todo_list`` write (revision 1, 2, …) sits below the client's
    pre-compaction watermark and is rejected as stale forever.
    """
    store = FakeStore()
    db = FakeDB(_todo_result([{"id": "a", "content": "x", "status": "in_progress"}], 5))
    agent = _agent(store, db)

    _hydrate(agent, [{"role": "user", "content": "[Context summary…]"}])

    assert store._revision == 5
    assert store.calls[-1]["items"][0]["status"] == "in_progress"


def test_revision_floor_does_not_regress_newer_history():
    """When history still shows a NEWER snapshot than the DB floor, history wins."""
    store = FakeStore()
    db = FakeDB(_todo_result([{"id": "a", "content": "x", "status": "pending"}], 2))
    agent = _agent(store, db)
    history = _todo_pair("c1", [{"id": "a", "content": "x", "status": "completed"}], 7)

    _hydrate(agent, history)

    assert store._revision == 7
    assert store.calls[-1]["items"][0]["status"] == "completed"


def test_equal_revision_does_not_double_restore():
    """Intact history already matches the DB — no redundant second restore."""
    store = FakeStore()
    db = FakeDB(_todo_result([{"id": "a", "content": "x", "status": "completed"}], 4))
    agent = _agent(store, db)
    history = _todo_pair("c1", [{"id": "a", "content": "x", "status": "completed"}], 4)

    _hydrate(agent, history)

    assert len(store.calls) == 1
    assert store._revision == 4


def test_floor_skipped_when_db_has_no_todo_row():
    store = FakeStore()
    agent = _agent(store, FakeDB(None))

    _hydrate(agent, [{"role": "user", "content": "hi"}])

    assert store.calls == []
    assert store._revision == 0
