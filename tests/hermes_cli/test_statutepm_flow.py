from __future__ import annotations

import json
from pathlib import Path

from hermes_cli import control_db as cp
from hermes_cli.control import _sample_payload
from hermes_cli.control_worker import run_deterministic_dispatch
from hermes_cli.statutepm_flow import StatutePMFlow


def test_statutepm_idle_registers_fresh_instance_then_offlines_it(tmp_path):
    root = tmp_path / ".hermes"
    conn = cp.connect(root=root)
    try:
        cp.bootstrap_statutepm_policies(conn, seed_instances=False)
    finally:
        conn.close()

    flow = StatutePMFlow(root=root, pm_instance_id="statutepm:wave-test", poll_interval_s=0, child_timeout_s=1)
    assert flow.run_once() is None

    conn = cp.connect(root=root)
    try:
        row = conn.execute("SELECT profile_id, status FROM cp_profile_instances WHERE instance_id=?", ("statutepm:wave-test",)).fetchone()
        assert row is not None
        assert row["profile_id"] == "statutepm"
        assert row["status"] == "offline"
    finally:
        conn.close()


def test_statutepm_active_run_uses_idempotent_child_and_offlines_pm(tmp_path):
    root = tmp_path / ".hermes"
    repo = tmp_path / "repo"
    repo.mkdir()
    conn = cp.connect(root=root)
    try:
        boot = cp.bootstrap_statutepm_policies(conn, seed_instances=True)
        parent = cp.create_dispatch_from_instance(
            conn,
            sender_instance_id=boot["instances"]["default"],
            receiver_profile="statutepm",
            payload=_sample_payload(repo),
            idempotency_key="parent-wave-1",
        )
    finally:
        conn.close()

    spawned = []

    def fake_spawn(child_id: str, payload: dict, child_root: Path | None, parent_id: str) -> int:
        spawned.append(child_id)
        run_deterministic_dispatch(root=child_root, profile_id="statute-worker", instance_id="statute-worker:test", dispatch_id=child_id)
        return 42

    flow = StatutePMFlow(root=root, pm_instance_id="statutepm:wave-1", spawn_child=fake_spawn, poll_interval_s=0, child_timeout_s=2)
    outcome = flow.run_once()
    assert outcome and outcome["status"] == "completed"
    assert len(spawned) == 1

    conn = cp.connect(root=root)
    try:
        pm_row = conn.execute("SELECT status FROM cp_profile_instances WHERE instance_id=?", ("statutepm:wave-1",)).fetchone()
        assert pm_row["status"] == "offline"
        child_rows = conn.execute("SELECT idempotency_key, parent_dispatch_id FROM cp_dispatches WHERE receiver_profile='statute-worker'").fetchall()
        assert len(child_rows) == 1
        assert child_rows[0]["parent_dispatch_id"] == parent
        assert child_rows[0]["idempotency_key"] == f"pm-child:{parent}:worker:0:0"
    finally:
        conn.close()


def test_statutepm_rejects_child_success_without_valid_result_contract(tmp_path):
    root = tmp_path / ".hermes"
    repo = tmp_path / "repo"
    repo.mkdir()
    conn = cp.connect(root=root)
    try:
        boot = cp.bootstrap_statutepm_policies(conn, seed_instances=True)
        parent = cp.create_dispatch_from_instance(conn, sender_instance_id=boot["instances"]["default"], receiver_profile="statutepm", payload=_sample_payload(repo))
    finally:
        conn.close()

    def bad_spawn(child_id: str, payload: dict, child_root: Path | None, parent_id: str) -> int:
        worker = cp.connect(root=child_root)
        try:
            cp.register_instance(worker, "statute-worker", instance_id="statute-worker:bad")
            ok, epoch = cp.claim_dispatch_by_id(worker, dispatch_id=child_id, instance_id="statute-worker:bad")
            assert ok and epoch is not None
            cp.advance_dispatch(worker, child_id, instance_id="statute-worker:bad", lease_epoch=epoch, status="running")
            cp.record_result(worker, dispatch_id=child_id, instance_id="statute-worker:bad", lease_epoch=epoch, result={"status": "completed", "summary": "missing schema"})
            cp.advance_dispatch(worker, child_id, instance_id="statute-worker:bad", lease_epoch=epoch, status="completed")
        finally:
            worker.close()
        return 99

    outcome = StatutePMFlow(root=root, pm_instance_id="statutepm:bad-child", spawn_child=bad_spawn, poll_interval_s=0, child_timeout_s=2).run_once()
    assert outcome and outcome["status"] == "failed"

    conn = cp.connect(root=root)
    try:
        latest = cp.get_latest_dispatch_result(conn, parent)["result"]
        assert latest["status"] == "action_required"
        assert latest["blockers"]
    finally:
        conn.close()


def _create_parent(root: Path, repo: Path) -> str:
    conn = cp.connect(root=root)
    try:
        boot = cp.bootstrap_statutepm_policies(conn, seed_instances=True)
        return cp.create_dispatch_from_instance(
            conn,
            sender_instance_id=boot["instances"]["default"],
            receiver_profile="statutepm",
            payload=_sample_payload(repo),
        )
    finally:
        conn.close()


def _complete_child_with_result(child_id: str, child_root: Path | None, result: dict) -> None:
    _finish_child_with_result(child_id, child_root, result, status="completed")


def _finish_child_with_result(child_id: str, child_root: Path | None, result: dict, *, status: str) -> None:
    conn = cp.connect(root=child_root)
    try:
        cp.register_instance(conn, "statute-worker", instance_id="statute-worker:custom")
        ok, epoch = cp.claim_dispatch_by_id(conn, dispatch_id=child_id, instance_id="statute-worker:custom")
        assert ok and epoch is not None
        cp.advance_dispatch(conn, child_id, instance_id="statute-worker:custom", lease_epoch=epoch, status="running")
        cp.record_result(conn, dispatch_id=child_id, instance_id="statute-worker:custom", lease_epoch=epoch, result=result)
        cp.advance_dispatch(conn, child_id, instance_id="statute-worker:custom", lease_epoch=epoch, status=status)
    finally:
        conn.close()


def _create_wave_parent(root: Path, repo: Path) -> str:
    payload = _sample_payload(repo)
    payload["task_permissions"] = ["read", "test", "write"]
    payload["constraints"] = {
        "no_live_db_mutation": True,
        "push_at_successful_wave_closeout": True,
        "wave": "K.1-K.7",
        "sprint_ids": ["K.1", "K.2", "K.3", "K.4", "K.5", "K.6", "K.7"],
    }
    conn = cp.connect(root=root)
    try:
        boot = cp.bootstrap_statutepm_policies(conn, seed_instances=True)
        return cp.create_dispatch_from_instance(
            conn,
            sender_instance_id=boot["instances"]["default"],
            receiver_profile="statutepm",
            payload=payload,
        )
    finally:
        conn.close()


def test_statutepm_wave_closeout_emits_next_wave_audit_notice_without_dispatching_it(tmp_path):
    root = tmp_path / ".hermes"
    repo = tmp_path / "repo"
    dispatch_dir = repo / "docs" / "dispatches"
    dispatch_dir.mkdir(parents=True)
    next_dispatch = dispatch_dir / "L.1-L.3-next-wave.md"
    next_dispatch.write_text("# Dispatch: L.1-L.3\n", encoding="utf-8")
    parent = _create_wave_parent(root, repo)
    child_result = {
        "schema": "control_result_v1",
        "status": "completed",
        "summary": "K wave complete",
        "artifacts": [{"path": "docs/dispatches/L.1-L.3-next-wave.md", "summary": "next wave dispatch"}],
        "tests": [{"command": "pytest", "exit_code": 0}],
        "blockers": [],
    }

    def spawn(child_id: str, payload: dict, child_root: Path | None, parent_id: str) -> int:
        _complete_child_with_result(child_id, child_root, child_result)
        return 12

    outcome = StatutePMFlow(root=root, pm_instance_id="statutepm:next-audit", spawn_child=spawn, poll_interval_s=0, child_timeout_s=2).run_dispatch(parent)

    assert outcome["status"] == "completed"
    conn = cp.connect(root=root)
    try:
        dispatches = [dict(r) for r in conn.execute("SELECT dispatch_id, receiver_profile, parent_dispatch_id FROM cp_dispatches ORDER BY created_at_ms").fetchall()]
        assert [d["receiver_profile"] for d in dispatches] == ["statutepm", "statute-worker"]
        notices = [dict(r) for r in conn.execute("SELECT kind, body, metadata_json FROM cp_messages WHERE sender_profile='statutepm' AND receiver_profile='default' ORDER BY created_at_ms").fetchall()]
        audit_notices = [m for m in notices if json.loads(m["metadata_json"]).get("notice_type") == "next_wave_ready_audit"]
        assert len(audit_notices) == 1
        audit = json.loads(audit_notices[0]["body"])
        metadata = json.loads(audit_notices[0]["metadata_json"])
        assert audit["schema"] == "statutepm_next_wave_audit_notice_v1"
        assert audit["dispatch_enabled"] is False
        assert audit["next_wave_dispatch_artifacts"] == ["docs/dispatches/L.1-L.3-next-wave.md"]
        assert audit["parent_dispatch_id"] == parent
        assert metadata["parent_dispatch_id"] == parent
        assert metadata["dispatch_enabled"] is False
    finally:
        conn.close()


def test_statutepm_child_completed_with_warnings_preserves_warning_status_on_parent(tmp_path):
    root = tmp_path / ".hermes"
    repo = tmp_path / "repo"
    repo.mkdir()
    parent = _create_parent(root, repo)
    result = {
        "schema": "control_result_v1",
        "status": "completed_with_warnings",
        "summary": "child done with warnings",
        "artifacts": [],
        "tests": [],
        "blockers": [],
    }

    def spawn(child_id: str, payload: dict, child_root: Path | None, parent_id: str) -> int:
        _complete_child_with_result(child_id, child_root, result)
        return 7

    outcome = StatutePMFlow(root=root, pm_instance_id="statutepm:warn-child", spawn_child=spawn, poll_interval_s=0, child_timeout_s=2).run_dispatch(parent)
    assert outcome["status"] == "completed"

    conn = cp.connect(root=root)
    try:
        row = conn.execute("SELECT status FROM cp_dispatches WHERE dispatch_id=?", (parent,)).fetchone()
        assert row["status"] == "completed"
        latest = cp.get_latest_dispatch_result(conn, parent)["result"]
        assert latest["status"] == "completed_with_warnings"
        assert latest["summary"] == "child done with warnings"
    finally:
        conn.close()


def test_statutepm_preserves_child_action_required_result_when_child_dispatch_failed(tmp_path):
    root = tmp_path / ".hermes"
    repo = tmp_path / "repo"
    repo.mkdir()
    parent = _create_parent(root, repo)
    result = {
        "schema": "control_result_v1",
        "status": "action_required",
        "summary": "child blocked on CodeRabbit",
        "artifacts": [],
        "tests": [],
        "blockers": [{"kind": "review_gate", "message": "CodeRabbit retry required"}],
    }

    def spawn(child_id: str, payload: dict, child_root: Path | None, parent_id: str) -> int:
        _finish_child_with_result(child_id, child_root, result, status="failed")
        return 9

    outcome = StatutePMFlow(root=root, pm_instance_id="statutepm:failed-child-action", spawn_child=spawn, poll_interval_s=0, child_timeout_s=2).run_dispatch(parent)
    assert outcome["status"] == "action_required"

    conn = cp.connect(root=root)
    try:
        row = conn.execute("SELECT status, last_error FROM cp_dispatches WHERE dispatch_id=?", (parent,)).fetchone()
        assert row["status"] == "blocked"
        assert "requires action" in row["last_error"]
        latest_row = cp.get_latest_dispatch_result(conn, parent)
        assert latest_row is not None
        latest = latest_row["result"]
        assert latest["status"] == "action_required"
        assert latest["summary"] == "child blocked on CodeRabbit"
        assert latest["blockers"][0]["message"] == "CodeRabbit retry required"
        assert latest["blockers"][-1]["child_dispatch_status"] == "failed"
    finally:
        conn.close()


def test_statutepm_child_hard_timeout_action_required_result_remains_failed(tmp_path):
    root = tmp_path / ".hermes"
    repo = tmp_path / "repo"
    repo.mkdir()
    parent = _create_parent(root, repo)
    result = {
        "schema": "control_result_v1",
        "status": "action_required",
        "summary": "agent subprocess hit hard timeout after 5 seconds; partial work may exist",
        "artifacts": [],
        "tests": [],
        "blockers": [{"kind": "hard_timeout", "message": "agent subprocess hit hard timeout"}],
    }

    def spawn(child_id: str, payload: dict, child_root: Path | None, parent_id: str) -> int:
        _finish_child_with_result(child_id, child_root, result, status="failed")
        return 11

    outcome = StatutePMFlow(root=root, pm_instance_id="statutepm:failed-child-hard-timeout", spawn_child=spawn, poll_interval_s=0, child_timeout_s=2).run_dispatch(parent)
    assert outcome["status"] == "failed"

    conn = cp.connect(root=root)
    try:
        row = conn.execute("SELECT status, last_error FROM cp_dispatches WHERE dispatch_id=?", (parent,)).fetchone()
        assert row["status"] == "failed"
        assert "child dispatch" in row["last_error"]
        latest_row = cp.get_latest_dispatch_result(conn, parent)
        assert latest_row is not None
        latest = latest_row["result"]
        assert latest["status"] == "action_required"
        assert latest["summary"] == result["summary"]
        assert latest["blockers"][0]["kind"] == "hard_timeout"
        assert latest["blockers"][-1]["child_dispatch_status"] == "failed"
    finally:
        conn.close()


def test_statutepm_preserves_child_failed_result_when_child_dispatch_failed(tmp_path):
    root = tmp_path / ".hermes"
    repo = tmp_path / "repo"
    repo.mkdir()
    parent = _create_parent(root, repo)
    result = {
        "schema": "control_result_v1",
        "status": "failed",
        "summary": "child tests failed",
        "artifacts": [],
        "tests": [{"command": "pytest", "exit_code": 1}],
        "blockers": [{"kind": "runtime_error", "message": "unit tests failed"}],
    }

    def spawn(child_id: str, payload: dict, child_root: Path | None, parent_id: str) -> int:
        _finish_child_with_result(child_id, child_root, result, status="failed")
        return 10

    outcome = StatutePMFlow(root=root, pm_instance_id="statutepm:failed-child-failed", spawn_child=spawn, poll_interval_s=0, child_timeout_s=2).run_dispatch(parent)
    assert outcome["status"] == "failed"

    conn = cp.connect(root=root)
    try:
        latest_row = cp.get_latest_dispatch_result(conn, parent)
        assert latest_row is not None
        latest = latest_row["result"]
        assert latest["status"] == "action_required"
        assert latest["summary"] == "child tests failed"
        assert latest["tests"][0]["exit_code"] == 1
        assert latest["blockers"][0]["kind"] == "runtime_error"
        assert latest["blockers"][-1]["child_dispatch_status"] == "failed"
    finally:
        conn.close()


def test_statutepm_child_completed_with_blockers_does_not_silently_complete_parent(tmp_path):
    root = tmp_path / ".hermes"
    repo = tmp_path / "repo"
    repo.mkdir()
    parent = _create_parent(root, repo)
    result = {
        "schema": "control_result_v1",
        "status": "completed",
        "summary": "contradictory child result",
        "artifacts": [],
        "tests": [],
        "blockers": [{"kind": "runtime_error", "message": "still blocked"}],
    }

    def spawn(child_id: str, payload: dict, child_root: Path | None, parent_id: str) -> int:
        _complete_child_with_result(child_id, child_root, result)
        return 8

    outcome = StatutePMFlow(root=root, pm_instance_id="statutepm:block-child", spawn_child=spawn, poll_interval_s=0, child_timeout_s=2).run_dispatch(parent)
    assert outcome["status"] == "failed"

    conn = cp.connect(root=root)
    try:
        row = conn.execute("SELECT status, last_error FROM cp_dispatches WHERE dispatch_id=?", (parent,)).fetchone()
        assert row["status"] == "failed"
        assert "invalid child result contract" in row["last_error"]
        latest = cp.get_latest_dispatch_result(conn, parent)["result"]
        assert latest["status"] == "action_required"
    finally:
        conn.close()
