"""C7 adversarial tests — pilot real-gate positive live provenance."""

from __future__ import annotations

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import pilot_evidence as pe


CONTRACT = "evidence-spine-p34"


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    conn = kb.connect()
    yield conn, home
    conn.close()


def _done_task(conn):
    tid = kb.create_task(
        conn, title="pilot", assignee="octacon",
        body="## Problem\nx\n## Success Criteria\ny", triage=True, tier="full",
    )
    conn.execute(
        "UPDATE tasks SET status = 'done', completed_at = created_at + 3600 WHERE id = ?",
        (tid,),
    )
    conn.commit()
    return tid


class TestC7RelabelAttack:
    def test_relabelled_simulated_record_fails_gate(self, env):
        """RED: stripping SIMULATED from a fixture record must NOT pass."""
        conn, home = env
        tid = _done_task(conn)
        record = pe.build_simulated_pilot_record(
            conn, tid, pipeline_contract_version=CONTRACT
        )
        # Attack: relabel
        record["label"] = None
        assert pe.real_pilot_gate_satisfied(record) is False
        record["label"] = "REAL"
        assert pe.real_pilot_gate_satisfied(record) is False

    def test_hand_built_dict_fails_gate(self, env):
        """RED: a hand-built dict cannot satisfy the real-pilot gate."""
        forged = {
            "pilot_id": "pilot-fake", "task_id": "t_fake",
            "pipeline_contract_version": CONTRACT,
            "started_at": 1, "ended_at": 2, "elapsed_seconds": 1,
            "handoff_count": 3, "revision_loops": 0, "failures": 0,
            "rework": 0, "reviewer_verdict": "pass", "final_status": "done",
            "evidence_refs": {}, "label": None,
            "live_provenance": {"live_pilot_authorised": True},
        }
        assert pe.real_pilot_gate_satisfied(forged) is False

    def test_simulated_builder_record_never_gate_eligible(self, env):
        conn, home = env
        tid = _done_task(conn)
        record = pe.build_simulated_pilot_record(
            conn, tid, pipeline_contract_version=CONTRACT
        )
        assert pe.real_pilot_gate_satisfied(record) is False
        # Even mutating every mutable field:
        record["label"] = "LIVE-PILOT"
        record["final_status"] = "done"
        record["reviewer_verdict"] = "pass"
        assert pe.real_pilot_gate_satisfied(record) is False