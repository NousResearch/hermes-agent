"""Regression for #131433: scheduler outcomes belong to compression continuations.

The projection receipt is consumed by Desktop's opt-in open/send regression,
so those gates see actual endpoint payloads rather than hand-written fixtures.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    store = SessionDB(db_path=tmp_path / "state.db")
    yield store
    store.close()


def publish(db, parent, child, messages=None, config=None):
    assert db.try_acquire_compression_lock(parent, "worker", ttl_seconds=60)
    db.publish_compression_child(
        parent_session_id=parent, child_session_id=child, source="cron",
        model_config=config, messages=messages or [{"role": "assistant", "content": "answer"}],
        compression_lock_holder="worker",
    )


def project(monkeypatch, path, sid, job):
    from hermes_cli.web_routers import cron, sessions

    def open_db(*args, **kwargs):
        return SessionDB(db_path=path, read_only=kwargs.get("read_only", False))

    monkeypatch.setattr(cron, "_job_owner_profile", lambda *args: None)
    monkeypatch.setattr(cron, "_open_session_db_for_profile", open_db)
    monkeypatch.setattr(cron, "_list_cron_output_runs", lambda *args: [])
    monkeypatch.setattr(sessions, "_open_session_db_for_profile", open_db)
    monkeypatch.setattr(sessions, "_serving_profile", lambda *args: "default")
    history = next(row for row in cron._list_cron_job_runs_sync(job)["runs"] if row["id"] == sid)
    detail = asyncio.run(sessions.get_session_detail(sid))
    return {"history": history, "detail": detail}


def test_scheduler_compression_projects_real_endpoint_receipts(tmp_path, monkeypatch, record_property):
    from cron import executions
    from cron.scheduler import _finalize_cron_session
    from hermes_cli.web_routers.cron import _run_owned_by

    receipts = []
    for reason, role in [("cron_complete", "assistant"), ("cron_incomplete_no_output", "user")]:
        path = tmp_path / (reason + ".db")
        root = f"cron_{reason}_20261002_030000"
        tip = "20261002_030001_child"
        execution = executions.create_execution(reason, source="scheduled")
        assert executions.mark_execution_running(execution["id"])["status"] == "running"
        store = SessionDB(db_path=path)
        store.create_session(root, "cron")
        store.append_message(root, role="user", content="scheduled task")
        assert _run_owned_by(store.get_session(root), executions.live_inflight_execution(reason))
        # Scheduler owns the attempt while compression rotates it.
        publish(store, root, tip, [{"role": role, "content": "answer or pending task"}])
        assert store.get_session(root)["cron_finalized"] is False
        unfinalized = project(monkeypatch, path, root, reason)
        receipts.append({"case": reason + "_unfinalized", "resumable": False, **unfinalized})
        agent = SimpleNamespace(session_id=tip, _end_session_on_close=True)
        _finalize_cron_session(store, agent, reason, "test job", root)
        assert agent._end_session_on_close is False
        terminal = executions.finish_execution(execution["id"], success=reason == "cron_complete")
        assert terminal["status"] in {"completed", "failed"}
        assert executions.live_inflight_execution(reason) is None
        store = SessionDB(db_path=path)
        try:
            assert store.get_session(tip)["end_reason"] == reason
            assert store.get_session(root)["end_reason"] == "compression"
            assert store.list_cron_job_runs(reason)[0]["cron_finalized"] is True
            receipt = project(monkeypatch, path, root, reason)
            for row in receipt.values():
                assert row["id"] == root
                assert row["cron_finalized"] is True
                assert row["scheduler_owned"] is False
            receipts.append({"case": reason, "resumable": True, **receipt})
            # Real resume and prompt heal must not remove authorization on the tip.
            store.reopen_session(tip)
            assert store.reopen_if_explicitly_closed(tip, provenance="test host") is None
            assert store.get_session(tip)["ended_at"] is None
            for generation in range(2):
                child = f"20261002_03000{generation + 2}_child"
                publish(store, tip, child, config={"max_iterations": 3})
                assert store.get_session(child)["cron_finalized"] is True
                assert store.get_session(root)["cron_finalized"] is True
                tip = child
            assert executions.get_execution(execution["id"]) == terminal
        finally:
            store.close()
    # JUnit carries the actual endpoint rows across the isolated runner boundary.
    record_property("desktop_receipts", json.dumps(receipts))


@pytest.mark.parametrize("reason", ["cron_complete", "cron_incomplete_no_output"])
@pytest.mark.parametrize("legacy", [False, True])
def test_interactive_rotation_preserves_outcome_for_two_generations(db, reason, legacy):
    from agent.conversation_compression import _publish_rotated_compaction

    root = "cron_job_20261002_030000"
    db.create_session(root, "cron")
    if legacy:
        db._write_sql("UPDATE sessions SET ended_at = 1, end_reason = ? WHERE id = ?", (reason, root))
    else:
        db.end_session(root, reason)
    db.reopen_session(root)
    agent = SimpleNamespace(
        session_id=root, model="test", _session_db=db,
        _session_init_model_config={"max_iterations": 3},
        _flush_messages_to_session_db=lambda *args, **kwargs: None,
    )
    for _ in range(2):
        parent = agent.session_id
        assert db.try_acquire_compression_lock(parent, "worker", ttl_seconds=60)
        _publish_rotated_compaction(
            agent, [], [{"role": "assistant", "content": "continued answer"}],
            new_system_prompt="test", lease=SimpleNamespace(holder="worker", ttl=60, watermark=None),
            old_session_id=parent, compressed_user_turn_outcome="absent",
        )
        child = db.get_session(agent.session_id)
        assert agent.session_id != parent
        assert child["cron_finalized"] is True
        assert json.loads(child["model_config"])["_cron_finalized"] == reason
        assert db.get_session(root)["cron_finalized"] is True
    assert agent._session_init_model_config == {"max_iterations": 3}
    # A held lease cannot turn historical finalization into permission to erase a later close.
    tip = agent.session_id
    assert db.try_acquire_compression_lock(tip, "worker", ttl_seconds=60)
    db.end_session(tip, "tui_close")
    with pytest.raises(RuntimeError, match="already ended"):
        db.publish_compression_child(
            parent_session_id=tip, child_session_id="forbidden", source="cron",
            messages=[{"role": "assistant", "content": "must not publish"}],
            compression_lock_holder="worker",
        )
    assert db.get_session(tip)["end_reason"] == "tui_close"
    assert db.get_session("forbidden") is None


@pytest.mark.parametrize("reason", [None, "ws_orphan_reap", "agent_close"])
def test_unfinalized_and_cleanup_continuations_remain_negative(db, reason):
    root = "cron_job_20261002_030000"
    db.create_session(root, "cron")
    publish(db, root, "continuation")
    if reason:
        db.end_session("continuation", reason)
    assert db.get_session(root)["cron_finalized"] is False
    assert db.list_cron_job_runs("job")[0]["cron_finalized"] is False


@pytest.mark.parametrize("kind", ["_branched_from", "_delegate_from", "_reset_from", "tool", "attempt", "no_compression", "ambiguous"])
def test_unrelated_descendants_cannot_finalize_root(db, kind):
    root = "cron_job_20261002_030000"
    db.create_session(root, "cron")
    child = "cron_other_20261002_040000" if kind == "attempt" else "unrelated"
    source = "tool" if kind == "tool" else "cron"
    config = {kind: root} if kind.startswith("_") else None
    db.create_session(child, source, parent_session_id=root, model_config=config)
    db.end_session(child, "cron_complete")
    if kind != "no_compression":
        db.end_session(root, "compression")
    if kind == "ambiguous":
        db.create_session("other_continuation", "cron", parent_session_id=root)
    assert db.get_session(root)["cron_finalized"] is False
    assert db.list_cron_job_runs("job")[0]["cron_finalized"] is False
