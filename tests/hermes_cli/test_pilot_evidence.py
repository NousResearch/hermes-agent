"""P4.4 tests — pilot evidence aggregation (instrumentation only)."""

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


def _pipeline_task(conn):
    tid = kb.create_task(
        conn, title="pilot fixture", assignee="octacon",
        body="## Problem\nx\n## Success Criteria\ny", triage=True,
        tier="full", pipeline_mode="full",
    )
    conn.execute(
        "UPDATE tasks SET status = 'done', completed_at = created_at + 3600 WHERE id = ?",
        (tid,),
    )
    conn.commit()
    return tid


class TestAggregation:
    def test_record_fields_complete(self, env):
        conn, home = env
        tid = _pipeline_task(conn)
        record = pe.build_pilot_record(conn, tid, pipeline_contract_version=CONTRACT)
        for field in ("pilot_id", "task_id", "pipeline_contract_version",
                      "started_at", "ended_at", "handoff_count",
                      "revision_loops", "failures", "rework",
                      "artifact_completeness", "reviewer_verdict",
                      "final_status", "evidence_refs"):
            assert field in record
        assert record["pipeline_contract_version"] == CONTRACT

    def test_deterministic_and_idempotent_report(self, env):
        conn, home = env
        tid = _pipeline_task(conn)
        r1 = pe.build_pilot_record(conn, tid, pipeline_contract_version=CONTRACT)
        r2 = pe.build_pilot_record(conn, tid, pipeline_contract_version=CONTRACT)
        assert r1 == r2
        assert pe.render_report([r1]) == pe.render_report([r2])

    def test_no_fabricated_cost_tokens(self, env):
        conn, home = env
        tid = _pipeline_task(conn)
        record = pe.build_pilot_record(conn, tid, pipeline_contract_version=CONTRACT)
        assert record["model_provider"]["token_cost"] == "unavailable"
        assert record["model_provider"]["model"] == "unavailable"

    def test_missing_task_none(self, env):
        conn, home = env
        assert pe.build_pilot_record(conn, "t_none", pipeline_contract_version=CONTRACT) is None


class TestSimulatedLabel:
    def test_simulated_record_labelled_and_gate_never_satisfied(self, env):
        conn, home = env
        tid = _pipeline_task(conn)
        record = pe.build_simulated_pilot_record(conn, tid, pipeline_contract_version=CONTRACT)
        assert record["label"] == pe.SIMULATED_LABEL
        assert pe.real_pilot_gate_satisfied(record) is False

    def test_no_pilot_tasks_created_by_module(self, env):
        conn, home = env
        tid = _pipeline_task(conn)
        count_before = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        pe.build_simulated_pilot_record(conn, tid, pipeline_contract_version=CONTRACT)
        count_after = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        assert count_before == count_after  # no pilot task creation


class TestArtifactCompleteness:
    def test_completeness_from_fixture(self, env):
        conn, home = env
        tid = _pipeline_task(conn)
        artifact = home / "feature-artifacts" / tid
        artifact.mkdir(parents=True, exist_ok=True)
        (artifact / "decompose-output.md").write_text("# d")
        record = pe.build_pilot_record(conn, tid, pipeline_contract_version=CONTRACT,
                                       artifact_dir=artifact)
        assert record["artifact_completeness"]["present"] == ["decompose-output.md"]
        assert record["artifact_completeness"]["complete"] is False


class TestReport:
    def test_simulated_report_labelled(self, env):
        conn, home = env
        tid = _pipeline_task(conn)
        record = pe.build_simulated_pilot_record(conn, tid, pipeline_contract_version=CONTRACT)
        report = pe.render_report([record], simulated=True)
        assert pe.SIMULATED_LABEL in report
        assert "does not satisfy the real-pilot gate" in report

    def test_stage_cron_not_added(self):
        """No scheduler/cron wiring exists in the module: no cron job
        creation, registration or scheduling invocation is present."""
        import ast
        import inspect
        import re
        src = inspect.getsource(pe)
        # No cron/scheduler API calls anywhere in the module (docstring
        # mentions are fine; executable wiring is not).
        for pattern in (r"cronjob", r"register_cron", r"add_cron", r"schedule\(",
                        r"crontab", r"CronJob"):
            assert re.search(pattern, src) is None, pattern