"""Store contracts: real SessionDB on a temp path, no mocks."""

from pathlib import Path

from hermes_state import SessionDB

from harness.state import Checkpoint, ExecutionState, FeatureState, KnowledgeItem, Task
from harness.store import HarnessStore


def _store(tmp_path: Path) -> HarnessStore:
    db = SessionDB(db_path=tmp_path / "harness-test.db")
    return HarnessStore(db)


def test_task_feature_execution_round_trip(tmp_path):
    store = _store(tmp_path)
    try:
        store.save_task(Task(id="t1", goal="ship it"))
        store.save_feature(FeatureState(id="f1", task_id="t1", name="work"))
        store.save_execution(ExecutionState(task_id="t1", feature_id="f1", iteration=3))
        loaded = store.load_task("t1")
        assert loaded is not None and loaded.goal == "ship it"
        assert [f.id for f in store.features_for_task("t1")] == ["f1"]
        executed = store.load_execution("t1")
        assert executed is not None and executed.iteration == 3
        assert store.load_task("missing") is None
    finally:
        store.close()


def test_events_append_in_order_with_cursor(tmp_path):
    store = _store(tmp_path)
    try:
        first = store.append_event("RUN", "one")
        second = store.append_event("RUN", "two")
        assert second == first + 1
        assert [e["payload"] for e in store.list_events()] == ["one", "two"]
        assert [e["payload"] for e in store.list_events(after=first)] == ["two"]
    finally:
        store.close()


def test_terminal_outcome_replays_from_log(tmp_path):
    store = _store(tmp_path)
    try:
        assert store.task_terminal_outcome("t1") is None
        store.append_event("TASK", "TASK_CREATED:t1")
        store.append_event("TASK", "TASK_STOPPED:t1")
        assert store.task_terminal_outcome("t1") is None
        store.append_event("TASK", "TASK_BUDGET_EXHAUSTED:t1")
        assert store.task_terminal_outcome("t1") == "BUDGET_EXHAUSTED"
        assert store.task_terminal_outcome("t2") is None
    finally:
        store.close()


def test_checkpoints_keep_latest_per_task(tmp_path):
    store = _store(tmp_path)
    try:
        for i in ("cp-1", "cp-2"):
            store.save_checkpoint(
                Checkpoint(id=i, task_id="t1", feature_id="f1", reason="test")
            )
        latest = store.latest_checkpoint("t1")
        assert latest is not None and latest.id == "cp-2"
        assert store.latest_checkpoint("t9") is None
    finally:
        store.close()


def test_knowledge_and_context_docs(tmp_path):
    store = _store(tmp_path)
    try:
        store.save_knowledge(
            KnowledgeItem(id="k1", type="SOLUTION", content="restart it")
        )
        store.save_context("ctx-1", {"items": ["a"]})
        assert [k.id for k in store.list_knowledge()] == ["k1"]
        assert store.load_context("ctx-1") == {"items": ["a"]}
    finally:
        store.close()
