import json
from pathlib import Path
from types import SimpleNamespace

import gateway.meta_router_executor as executor


class _CaptureWriter:
    def __init__(self):
        self.kwargs = None

    def log_routing_outcome(self, **kwargs):
        self.kwargs = kwargs



def test_do_phase2_logs_executor_metadata_from_execution_result(monkeypatch, tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "output.md").write_text("## Status\ncompleted\n", encoding="utf-8")
    (state_dir / "scores.json").write_text(
        json.dumps({"total_weighted_score": 92, "threshold": 90, "verdict": "GOOD"}),
        encoding="utf-8",
    )
    (state_dir / "delivery.json").write_text(
        json.dumps({"oracle": "PASS", "delivery_gate": {"all_passed": True}, "ref_entry": {"id": "ref-1"}}),
        encoding="utf-8",
    )
    (state_dir / "execution_result.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "engine": "omx",
                "workflow": "plain",
                "launch_mode": "exec",
                "omx_version": "0.13.1",
                "codex_version": "0.121.0",
                "team_size": 1,
            }
        ),
        encoding="utf-8",
    )

    som_pipeline = tmp_path / "som_pipeline.py"
    som_pipeline.write_text("# stub\n", encoding="utf-8")
    writer = _CaptureWriter()

    monkeypatch.setattr(executor, "_SOM_PIPELINE", som_pipeline)
    monkeypatch.setattr(executor, "populate_evidence_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(executor, "_validate_evidence", lambda *args, **kwargs: (True, "ok"))
    monkeypatch.setattr(executor, "_run_adv_pass", lambda *args, **kwargs: (True, {"findings": []}, ""))
    monkeypatch.setattr(executor, "_load_log_writer", lambda: writer)
    monkeypatch.setattr(executor, "_maybe_trigger_optimizer", lambda: None)
    monkeypatch.setattr(
        executor.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"status": "complete", "passed": True}),
            stderr="",
        ),
    )

    phase2 = executor._do_phase2(
        request_id="rid-exec-meta",
        task_type="code",
        task_text="Implement ready.py and tests",
        som_state_dir=state_dir,
        final_response="## Status\ncompleted\n",
        t0=0.0,
        routing_artifact_version="candidate-0010",
        session_id="session-1",
    )

    assert phase2.passed is True
    assert writer.kwargs is not None
    assert writer.kwargs["executor_engine"] == "omx"
    assert writer.kwargs["omx_workflow"] == "plain"
    assert writer.kwargs["launch_mode"] == "exec"
    assert writer.kwargs["omx_version"] == "0.13.1"
    assert writer.kwargs["codex_version"] == "0.121.0"
    assert writer.kwargs["team_size"] == 1
    assert any("executor_engine=omx" in note for note in phase2.notes)
    assert any("omx_workflow=plain" in note for note in phase2.notes)


def test_do_phase2_logs_structured_join_fields(monkeypatch, tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "output.md").write_text("## Status\ncompleted\n", encoding="utf-8")
    (state_dir / "scores.json").write_text(
        json.dumps({"total_weighted_score": 92, "threshold": 90, "verdict": "GOOD"}),
        encoding="utf-8",
    )
    (state_dir / "delivery.json").write_text(
        json.dumps({"oracle": "PASS", "delivery_gate": {"all_passed": True}, "ref_entry": {"id": "ref-1"}}),
        encoding="utf-8",
    )

    som_pipeline = tmp_path / "som_pipeline.py"
    som_pipeline.write_text("# stub\n", encoding="utf-8")
    writer = _CaptureWriter()

    monkeypatch.setattr(executor, "_SOM_PIPELINE", som_pipeline)
    monkeypatch.setattr(executor, "populate_evidence_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(executor, "_validate_evidence", lambda *args, **kwargs: (True, "ok"))
    monkeypatch.setattr(executor, "_run_adv_pass", lambda *args, **kwargs: (True, {"findings": []}, ""))
    monkeypatch.setattr(executor, "_load_log_writer", lambda: writer)
    monkeypatch.setattr(executor, "_maybe_trigger_optimizer", lambda: None)
    monkeypatch.setattr(
        executor.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"status": "complete", "passed": True}),
            stderr="",
        ),
    )

    phase2 = executor._do_phase2(
        request_id="rid-join-fields",
        task_type="code",
        task_text="Implement ready.py and tests",
        som_state_dir=state_dir,
        final_response="## Status\ncompleted\n",
        t0=0.0,
        routing_artifact_version="candidate-0010",
        session_id="session-1",
        source="gateway",
        surface="telegram",
    )

    assert phase2.passed is True
    assert writer.kwargs is not None
    assert writer.kwargs["session_id"] == "session-1"
    assert writer.kwargs["source"] == "gateway"
    assert writer.kwargs["surface"] == "telegram"


def test_run_outcome_only_logs_structured_join_fields(monkeypatch):
    writer = _CaptureWriter()

    monkeypatch.setattr(executor, "_load_log_writer", lambda: writer)
    monkeypatch.setattr(executor, "_maybe_trigger_optimizer", lambda: None)

    executor.run_outcome_only(
        request_id="rid-outcome-only",
        task_type="research",
        t0=0.0,
        routing_artifact_version="candidate-0010",
        session_id="session-9",
        source="gateway",
        surface="telegram",
    )

    assert writer.kwargs is not None
    assert writer.kwargs["session_id"] == "session-9"
    assert writer.kwargs["source"] == "gateway"
    assert writer.kwargs["surface"] == "telegram"


def test_maybe_trigger_optimizer_uses_eligible_counts_and_deduplicates_threshold(monkeypatch):
    triggers = []

    monkeypatch.setattr(executor, "_count_eligible_outcomes", lambda: 10)
    monkeypatch.setattr(executor, "_trigger_optimizer_bg", lambda: triggers.append("go"))
    monkeypatch.setattr(executor, "_last_optimizer_trigger_eligible_count", None, raising=False)

    executor._maybe_trigger_optimizer()
    executor._maybe_trigger_optimizer()

    assert triggers == ["go"]
