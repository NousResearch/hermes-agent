from pathlib import Path

from conductor.engine import Conductor, TickResult
from conductor.models import CampaignPlan, Step, StepKind
from conductor.receipts import build_receipt, write_receipt
from conductor.store import ConductorStore


class Launcher:
    def __init__(self):
        self.specs = []
        self.running = set()

    def launch(self, spec):
        self.specs.append(spec)
        self.running.add(spec.tmux_session)
        return {"pid": 1, "start_marker": "x"}

    def is_running(self, name):
        return name in self.running

    def cleanup(self, name):
        self.running.discard(name)


def campaign(tmp_path, *, steps, budgets=None):
    return CampaignPlan(
        "limits",
        str(tmp_path),
        ["known.txt"],
        steps,
        {"command": ["writer"], "provider": "provider", "model": "writer-model"},
        {"command": ["reviewer"], "provider": "provider", "model": "reviewer-model"},
        budgets or {},
    )


def test_reviewer_is_distinct_read_only_and_fallback_cannot_pass(tmp_path):
    launcher = Launcher()
    plan = campaign(
        tmp_path, steps=[Step("review", StepKind.JUDGMENT_REVIEW, "review")]
    )
    store = ConductorStore(tmp_path / "state.sqlite")
    store.create_campaign(plan)
    engine = Conductor(store, launcher)
    assert engine.tick("limits") is TickResult.LAUNCHED_REVIEWER
    worker = store.active_worker("limits")
    assert worker.role == "reviewer"
    assert worker.read_only is True
    assert worker.model == "reviewer-model"
    payload = build_receipt(worker, status="COMPLETE", usage={}, model_fallback=True)
    write_receipt(Path(worker.receipt_path), payload)
    launcher.running.clear()
    assert engine.tick("limits") is TickResult.BLOCKED_INVALID_RECEIPT


def test_turn_run_day_and_backoff_limits_fail_closed(tmp_path):
    launcher = Launcher()
    plan = campaign(
        tmp_path,
        steps=[Step("write", StepKind.IMPLEMENTATION, "write")],
        budgets={
            "max_conductor_turns": 1,
            "max_runs_per_day": 0,
            "max_retries": 0,
            "backoff_base_seconds": 4,
        },
    )
    store = ConductorStore(tmp_path / "state.sqlite")
    store.create_campaign(plan)
    engine = Conductor(store, launcher)
    assert engine.tick("limits") is TickResult.BLOCKED_BUDGET
    assert launcher.specs == []


def test_exact_usage_unavailable_is_labeled_and_bounded(tmp_path):
    plan = campaign(tmp_path, steps=[Step("write", StepKind.IMPLEMENTATION, "write")])
    plan.writer["usage_reporting"] = "unavailable"
    launcher = Launcher()
    store = ConductorStore(tmp_path / "state.sqlite")
    store.create_campaign(plan)
    engine = Conductor(store, launcher)
    engine.tick("limits")
    worker = store.active_worker("limits")
    payload = build_receipt(worker, status="COMPLETE", usage={}, usage_exact=False)
    write_receipt(Path(worker.receipt_path), payload)
    launcher.running.clear()
    assert engine.tick("limits") is TickResult.COMPLETE


def test_empty_observer_output_is_silent_and_uses_no_agent(tmp_path):
    marker = tmp_path / "calls"
    step = Step(
        "observe", StepKind.OBSERVATION, "", command=["/bin/sh", "-c", "exit 0"]
    )
    store = ConductorStore(tmp_path / "state.sqlite")
    store.create_campaign(campaign(tmp_path, steps=[step]))
    result = Conductor(store, Launcher()).tick("limits")
    assert result is TickResult.OBSERVED_SILENT
    assert not marker.exists()
