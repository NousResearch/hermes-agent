from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from conductor.engine import Conductor, TickResult
from conductor.models import CampaignPlan, Step, StepKind, WorkerState
from conductor.receipts import build_receipt, write_receipt
from conductor.store import ConductorStore


class FakeLauncher:
    def __init__(self) -> None:
        self.launches = []
        self.running = set()
        self.cleaned = []

    def launch(self, spec):
        self.launches.append(spec)
        self.running.add(spec.tmux_session)
        return {"pid": 123, "start_marker": "fake-123"}

    def is_running(self, session):
        return session in self.running

    def cleanup(self, session):
        self.running.discard(session)
        self.cleaned.append(session)


def plan(repo: Path, *, steps=None, budgets=None) -> CampaignPlan:
    return CampaignPlan(
        campaign_id="campaign-1",
        cwd=str(repo.resolve()),
        mutable_manifest=["known.txt"],
        steps=steps or [Step("write", StepKind.IMPLEMENTATION, "make known.txt")],
        writer={"command": ["fake-writer"], "provider": "p", "model": "writer"},
        reviewer={"command": ["fake-reviewer"], "provider": "p", "model": "reviewer"},
        budgets=budgets or {},
    )


def make_engine(tmp_path, campaign_plan, launcher=None, now=None):
    store = ConductorStore(tmp_path / "state.sqlite")
    store.create_campaign(campaign_plan)
    return Conductor(store, launcher or FakeLauncher(), now=now), store


def test_implementation_launches_once_and_second_tick_adopts_progress(tmp_path):
    launcher = FakeLauncher()
    engine, store = make_engine(tmp_path, plan(tmp_path), launcher)

    first = engine.tick("campaign-1")
    assert first is TickResult.LAUNCHED_WRITER
    worker = store.active_worker("campaign-1")
    assert worker.state is WorkerState.RUNNING
    assert worker.role == "writer"
    assert worker.cwd == str(tmp_path.resolve())
    assert worker.tmux_session in launcher.running
    assert len(launcher.launches) == 1

    engine.record_progress(worker.worker_id, "heartbeat-1")
    assert engine.tick("campaign-1") is TickResult.ADOPTED_PROGRESSING
    assert len(launcher.launches) == 1


def test_stale_tmux_is_not_progress_and_timeout_never_takes_over(tmp_path):
    clock = [100.0]
    launcher = FakeLauncher()
    engine, store = make_engine(
        tmp_path,
        plan(tmp_path, budgets={"wall_time_seconds": 10, "max_retries": 1}),
        launcher,
        now=lambda: clock[0],
    )
    engine.tick("campaign-1")
    worker = store.active_worker("campaign-1")
    clock[0] = 111.0

    assert engine.tick("campaign-1") is TickResult.WAITING_STALE
    assert store.get_campaign("campaign-1").state == "WAITING_WRITER"
    assert len(launcher.launches) == 1

    launcher.running.clear()
    assert engine.tick("campaign-1") is TickResult.RETRY_BACKOFF
    assert len(launcher.launches) == 1


def test_receipt_validation_rejects_malformed_and_fabricated(tmp_path):
    launcher = FakeLauncher()
    engine, store = make_engine(tmp_path, plan(tmp_path), launcher)
    engine.tick("campaign-1")
    worker = store.active_worker("campaign-1")

    Path(worker.receipt_path).write_text("{}", encoding="utf-8")
    launcher.running.clear()
    assert engine.tick("campaign-1") is TickResult.BLOCKED_INVALID_RECEIPT

    payload = build_receipt(worker, status="COMPLETE", usage={"input_tokens": 1})
    payload["model"] = "impersonator"
    write_receipt(Path(worker.receipt_path), payload)
    assert engine.tick("campaign-1") is TickResult.BLOCKED_INVALID_RECEIPT


def test_budget_counts_cache_reads_and_day_budget_survives_restart(tmp_path):
    budgets = {
        "max_processed_tokens_per_run": 10,
        "max_processed_tokens_per_day": 12,
        "max_runs_per_day": 3,
    }
    launcher = FakeLauncher()
    engine, store = make_engine(tmp_path, plan(tmp_path), launcher)
    engine.tick("campaign-1")
    worker = store.active_worker("campaign-1")
    payload = build_receipt(
        worker,
        status="COMPLETE",
        usage={
            "input_tokens": 2,
            "output_tokens": 1,
            "reasoning_tokens": 1,
            "cache_read_tokens": 6,
            "cache_write_tokens": 9,
        },
    )
    write_receipt(Path(worker.receipt_path), payload)
    launcher.running.clear()
    assert engine.tick("campaign-1") is TickResult.COMPLETE
    usage = store.daily_usage("campaign-1")
    assert usage["processed_tokens"] == 10
    assert usage["cache_read_tokens"] == 6
    assert usage["cache_write_tokens"] == 9

    restarted = Conductor(ConductorStore(tmp_path / "state.sqlite"), launcher)
    assert restarted.store.daily_usage("campaign-1")["processed_tokens"] == 10


def test_concurrent_ticks_have_one_owner_and_one_launch(tmp_path):
    launcher = FakeLauncher()
    engine, _ = make_engine(tmp_path, plan(tmp_path), launcher)
    barrier = threading.Barrier(3)
    results = []

    def run():
        barrier.wait()
        results.append(engine.tick("campaign-1"))

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()
    assert len(launcher.launches) == 1
    assert TickResult.LAUNCHED_WRITER in results


def test_restart_adopts_the_same_campaign_and_worker(tmp_path):
    launcher = FakeLauncher()
    engine, store = make_engine(tmp_path, plan(tmp_path), launcher)
    engine.tick("campaign-1")
    worker = store.active_worker("campaign-1")
    engine.record_progress(worker.worker_id, "after-restart-heartbeat")

    restarted = Conductor(ConductorStore(tmp_path / "state.sqlite"), launcher)
    assert restarted.tick("campaign-1") is TickResult.ADOPTED_PROGRESSING
    assert restarted.store.active_worker("campaign-1").worker_id == worker.worker_id
    assert len(launcher.launches) == 1


def test_step_classification_and_human_block(tmp_path):
    steps = [
        Step("impl", StepKind.IMPLEMENTATION, "x"),
        Step("review", StepKind.JUDGMENT_REVIEW, "x"),
        Step("gate", StepKind.DETERMINISTIC_GATE, "x", command=["true"]),
        Step("observe", StepKind.OBSERVATION, "x", command=["true"]),
        Step("decide", StepKind.HUMAN_DECISION, "choose"),
    ]
    campaign = plan(tmp_path, steps=steps)
    assert [step.kind for step in campaign.steps] == list(StepKind)
