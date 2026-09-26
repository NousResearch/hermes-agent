"""Driver contracts: scripted agent steps, real SessionDB, no model calls."""

from pathlib import Path
from typing import Any, Dict

import pytest

from harness.knowledge import KnowledgeCandidate
from harness.loop import HarnessRunner, StepResult
from harness.state import Outcome, StepStatus, ToolObservation
from harness.store import HarnessStore
from hermes_state import SessionDB


def _runner(tmp_path: Path, make_step):
    db = SessionDB(db_path=tmp_path / "loop-test.db")
    return HarnessRunner(HarnessStore(db), make_step)


def _ok_step(**overrides):
    def run(task, feature, context):
        base: Dict[str, Any] = dict(
            summary="did work",
            status_hint=Outcome.CONTINUE,
            observations=[
                ToolObservation(
                    id="obs-1", tool="read", success=True, summary="read ok"
                )
            ],
            evidence=["file content"],
        )
        base.update(overrides)
        return StepResult(**base)

    return run


def test_full_cycle_completes_with_evidence(tmp_path):
    runner = _runner(tmp_path, lambda task: _ok_step(status_hint=StepStatus.DONE))
    task = runner.create("ship it", ["tests pass"])
    assert runner.step(task.id) == Outcome.COMPLETED
    status = runner.status(task.id)
    assert status["iteration"] == 1 and status["observations"] == 1
    with pytest.raises(RuntimeError):
        runner.step(task.id)


def test_done_without_evidence_does_not_complete(tmp_path):
    runner = _runner(
        tmp_path,
        lambda task: _ok_step(
            status_hint=StepStatus.DONE, observations=[], evidence=[]
        ),
    )
    task = runner.create("ship it", ["tests pass"])
    assert runner.step(task.id) == Outcome.CONTINUE


def test_tool_failure_recovers_without_blind_retry(tmp_path):
    calls = []

    def flaky(task, feature, context):
        calls.append(1)
        if len(calls) == 1:
            return StepResult(
                observations=[
                    ToolObservation(
                        id="obs-1", tool="write", success=False, summary="disk gone"
                    )
                ]
            )
        return StepResult(
            summary="replanned",
            status_hint=Outcome.CONTINUE,
            observations=[
                ToolObservation(
                    id="obs-2", tool="read", success=True, summary="read ok"
                )
            ],
            evidence=["content"],
        )

    runner = _runner(tmp_path, lambda task: flaky)
    task = runner.create("write it", ["file exists"])
    assert runner.step(task.id) == Outcome.CONTINUE
    assert runner.step(task.id) == Outcome.CONTINUE
    assert len(calls) == 2


def test_repeated_identical_failure_stops(tmp_path):
    def broken(task, feature, context):
        return StepResult(
            observations=[
                ToolObservation(
                    id="obs-x", tool="write", success=False, summary="disk gone"
                )
            ]
        )

    runner = _runner(tmp_path, lambda task: broken)
    task = runner.create("write it", ["file exists"])
    outcome = runner.run(task.id, max_rounds=30)
    assert outcome == Outcome.STOPPED


def test_pause_cancel_resume(tmp_path):
    runner = _runner(tmp_path, lambda task: _ok_step())
    task = runner.create("long job", ["done"])
    runner.pause(task.id)
    runner.cancel(task.id)
    assert runner.resume(task.id, max_rounds=2) == Outcome.CONTINUE
    kinds = [
        e["payload"].split(":")[0]
        for e in runner._store.list_events()
        if e["kind"] == "TASK"
    ]
    assert (
        "TASK_PAUSED" in kinds and "TASK_CANCELLED" in kinds and "TASK_RESUMED" in kinds
    )


def test_resume_rebuilds_from_store(tmp_path):
    db_path = tmp_path / "resume.db"
    first = HarnessRunner(
        HarnessStore(SessionDB(db_path=db_path)), lambda task: _ok_step()
    )
    task = first.create("persist me", ["done"])
    assert first.step(task.id) == Outcome.CONTINUE
    second = HarnessRunner(
        HarnessStore(SessionDB(db_path=db_path)), lambda task: _ok_step()
    )
    assert second.status(task.id)["iteration"] == 1
    assert second.resume(task.id, max_rounds=1) == Outcome.CONTINUE


def test_knowledge_stored_only_with_evidence(tmp_path):
    candidate = KnowledgeCandidate(
        type="SOLUTION", content="restart the worker to clear it"
    )
    runner = _runner(tmp_path, lambda task: _ok_step(proposed_knowledge=[candidate]))
    task = runner.create("learn", ["done"])
    runner.step(task.id)
    assert [k.content for k in runner._store.list_knowledge()] == [candidate.content]
