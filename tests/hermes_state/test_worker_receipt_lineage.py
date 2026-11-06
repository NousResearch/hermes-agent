"""Real worker receipts fence compression children without changing schema errors."""

import time

import pytest

from agent.runtime_session_store import RuntimeSessionStore
from hermes_state import SessionDB
from hermes_state_lineage import SessionIdOwnedElsewhereError
from hermes_state_runtime import RuntimeStoreError, begin_runtime_epoch, mutate_worker_execution, register_worker_execution


@pytest.fixture
def managed_worker(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    db = SessionDB(home / "state.db")
    (home / "profiles" / "science").mkdir(parents=True)
    sibling = SessionDB(home / "profiles" / "science" / "state.db")
    db.create_session("owned", "cli", system_prompt="PREFIX")
    epoch = begin_runtime_epoch(db, instance_id="receipt-regression")
    scope = dict(epoch=epoch, execution_id="worker", session_id="owned", generation=0)
    register_worker_execution(db, **scope, kind="compute", adoption_secret="fixture-only", require_idle=True)
    store = RuntimeSessionStore(lambda method, **params: mutate_worker_execution(db, **params), scope, tmp_path / "outbox")
    assert store.try_acquire_compression_lock("owned", "holder")
    try:
        yield db, sibling, store
    finally:
        store.failure = None
        store.journal["pending"] = []
        store.close()
        sibling.close()
        db.close()


def _snapshot(db):
    return {
        "parent": dict(db.get_session("owned")),
        "worker": tuple(db._read_one("SELECT * FROM worker_executions WHERE execution_id='worker'")),
        "receipts": [tuple(row) for row in db._read_all("SELECT * FROM worker_receipts ORDER BY sequence")],
        "lease": tuple(db._read_one("SELECT * FROM compression_locks WHERE session_id='owned'")),
    }


def test_real_compression_receipt_rejects_foreign_child_without_partial_effects(managed_worker):
    db, sibling, store = managed_worker
    sibling.create_session("foreign-child", "cli")
    sibling.append_message("foreign-child", "user", "sibling history")
    before, foreign_before = _snapshot(db), dict(sibling.get_session("foreign-child"))
    with pytest.raises(SessionIdOwnedElsewhereError):
        store.publish_compression_child(
            parent_session_id="owned", child_session_id="foreign-child", source="cli",
            messages=[{"role": "assistant", "content": "SUMMARY"}], system_prompt="PREFIX",
            compression_lock_holder="holder",
        )
    assert db.get_session("foreign-child") is None
    assert _snapshot(db) == before
    assert dict(sibling.get_session("foreign-child")) == foreign_before
    assert [row["content"] for row in sibling.get_messages("foreign-child")] == ["sibling history"]
    assert store.scope["session_id"] == "owned"


@pytest.mark.parametrize("child", [None, "", "x" * 1025], ids=["missing", "empty", "too-long"])
def test_malformed_child_keeps_the_canonical_invalid_params_error(managed_worker, child):
    db, sibling, store = managed_worker
    if isinstance(child, str) and child:
        sibling._conn.execute("INSERT INTO sessions(id,source,started_at,message_count) VALUES (?,'cli',?,0)",
                              (child, time.time()))
        sibling._conn.commit()
    before = _snapshot(db)
    with pytest.raises(RuntimeStoreError, match="invalid_params"):
        store.publish_compression_child(
            parent_session_id="owned", child_session_id=child, source="cli",
            messages=[{"role": "assistant", "content": "SUMMARY"}], system_prompt="PREFIX",
            compression_lock_holder="holder",
        )
    assert _snapshot(db) == before


def test_child_field_on_other_operation_does_not_change_its_schema_error(managed_worker):
    db, sibling, store = managed_worker
    sibling.create_session("foreign-child", "cli")
    before = _snapshot(db)
    with pytest.raises(RuntimeStoreError, match="invalid_params"):
        mutate_worker_execution(
            db, **store.scope, sequence=store.journal["next_sequence"], operation="session.prompt",
            payload={"prompt": "PREFIX", "child_session_id": "foreign-child"},
        )
    assert _snapshot(db) == before
