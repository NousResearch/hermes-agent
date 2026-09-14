"""Submitting an objective to a real runtime board, and reading it back.

These tests write to a real ``kanban.db`` through the runtime's own API, because the
properties that matter here — idempotency, dependency ordering, and a refusal writing
nothing — are properties of that interaction and a fake would assert my assumptions about
it rather than its behaviour.

They skip when the runtime is not importable, which is the normal state of NOVA's own test
environment: the layer is meant to install without it.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from nova.apply import apply_bundle
from nova.audit import AuditLog, NullAuditLog
from nova.spec import load_bundle
from nova.supervisor import collect, submit_objective
from nova.supervisor.report import BLOCKED, NOT_STARTED, RUNNING, derive_state
from nova.supervisor.submit import work_key

from .conftest import EXAMPLE_BUNDLE

@pytest.fixture(scope="module", autouse=True)
def _requires_runtime():
    pytest.importorskip(
        "hermes_cli.kanban_db", reason="submission needs the runtime's work-store API"
    )


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    """An applied bundle on a fresh, empty board."""
    from nova.runtime.hermes import HermesRuntime

    home = tmp_path / "home"
    home.mkdir()
    # The runtime's work store resolves from its own environment, not from our paths.
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("NOVA_HOME", str(home))

    bundle = load_bundle(EXAMPLE_BUNDLE)
    runtime = HermesRuntime(home=home, tenant_id=bundle.tenant_id)
    audit = AuditLog(home / "audit.jsonl", tenant_id=bundle.tenant_id)
    apply_bundle(bundle, runtime, audit=audit)
    return bundle, runtime, audit


@pytest.fixture
def objective(deployment):
    bundle, _, _ = deployment
    return bundle.objectives[0]


def submit(deployment, objective, **kwargs):
    bundle, runtime, audit = deployment
    return submit_objective(
        objective, bundle.agents, runtime, audit=audit, tenant_id=bundle.tenant_id, **kwargs
    )


# -- submission -------------------------------------------------------------


def test_every_step_becomes_a_work_item(deployment, objective):
    report = submit(deployment, objective)
    assert not report.refused
    assert len(report.result.created) == len(objective.steps)


def test_independent_steps_are_ready_and_dependent_ones_wait(deployment, objective):
    """Parallelism is the absence of a dependency, and the runtime expresses that as the
    state it gives the task. If this inverts, a plan runs in series and nobody notices."""
    report = submit(deployment, objective)
    states = {item.key.split(":")[1]: item.state for item in report.result.items}
    assert states["pull-ledger"] == "ready"
    assert states["handbook-thresholds"] == "ready"
    assert states["compare"] == "todo"
    assert states["write-findings"] == "todo"


def test_resubmission_creates_nothing(deployment, objective):
    """A supervisor that duplicates a month-end close on a retry is worse than one that
    does nothing."""
    first = submit(deployment, objective)
    second = submit(deployment, objective)
    assert len(second.result.created) == 0
    assert len(second.result.existing) == len(objective.steps)
    # And the same tasks, not look-alikes.
    assert {item.task_id for item in second.result.items} == {
        item.task_id for item in first.result.items
    }


def test_a_dry_run_writes_nothing(deployment, objective):
    _, runtime, _ = deployment
    report = submit(deployment, objective, dry_run=True)
    assert report.result.dry_run
    assert len(report.result.created) == len(objective.steps)
    assert runtime.list_tasks(limit=100) == []


def test_a_refused_objective_creates_no_work_at_all(deployment, objective):
    """Not "submit the valid steps and report the rest": half a month-end close running
    while the other half is refused is worse than none of it."""
    bundle, runtime, audit = deployment
    stripped = tuple(
        replace(spec, delegation=replace(spec.delegation, may_assign_to=()))
        if spec.id == objective.owner
        else spec
        for spec in bundle.agents
    )
    report = submit_objective(objective, stripped, runtime, audit=audit, tenant_id="acme")

    assert report.refused
    assert report.result is None
    assert runtime.list_tasks(limit=100) == []


def test_a_refusal_is_recorded_even_though_nothing_was_written(deployment, objective):
    """An audit log containing only successful submissions cannot answer "did anyone try"."""
    bundle, runtime, audit = deployment
    stripped = tuple(
        replace(spec, delegation=replace(spec.delegation, may_assign_to=()))
        if spec.id == objective.owner
        else spec
        for spec in bundle.agents
    )
    submit_objective(objective, stripped, runtime, audit=audit, tenant_id="acme")

    refused = [event for event in audit.read() if event.kind == "objective.refused"]
    assert refused
    assert refused[-1].detail["refusals"]


def test_submission_is_a_write_ahead_model_visible_change(deployment, objective):
    """A worker reads the title and body this puts on the board as its instructions."""
    _, _, audit = deployment
    submit(deployment, objective)

    events = [event for event in audit.read() if event.kind == "work.submitted"]
    assert [event.phase for event in events] == ["intent", "committed"]
    assert all(event.model_visible for event in events)


def test_the_tenant_is_stamped_on_the_created_work(deployment, objective):
    _, runtime, _ = deployment
    submit(deployment, objective)
    assert all(view.tenant_id == "acme" for view in runtime.list_tasks(limit=100))


# -- collection -------------------------------------------------------------


def test_an_unsubmitted_objective_reports_not_started(deployment, objective):
    _, runtime, _ = deployment
    report = collect(objective, runtime)
    assert report.state == NOT_STARTED
    assert all(not step.submitted for step in report.steps)


def test_steps_are_matched_by_key_not_by_title(deployment, objective):
    """Two objectives with a 'Review findings' step would otherwise report each other's
    progress."""
    _, runtime, _ = deployment
    submit(deployment, objective)
    report = collect(objective, runtime)
    assert all(step.submitted for step in report.steps)
    assert {step.key for step in report.steps} == {
        work_key(objective.id, step.id) for step in objective.steps
    }


def test_a_submitted_objective_reports_running(deployment, objective):
    _, runtime, _ = deployment
    submit(deployment, objective)
    assert collect(objective, runtime).state == RUNNING


def test_progress_counts_every_declared_step_not_only_submitted_ones(deployment, objective):
    _, runtime, _ = deployment
    submit(deployment, objective)
    done, total = collect(objective, runtime).progress
    assert (done, total) == (0, len(objective.steps))


# -- derived state ----------------------------------------------------------


def _step(**overrides):
    from nova.supervisor.report import StepReport

    base = dict(step_id="s", title="T", assignee="a", key="k", submitted=True, state="done")
    base.update(overrides)
    return StepReport(**base)


def test_one_blocked_step_blocks_the_whole_objective():
    """Reporting it as running because most of it is fine is how a stuck process goes
    unnoticed for a week."""
    steps = [_step(step_id=str(n)) for n in range(5)]
    steps.append(_step(step_id="stuck", state="blocked"))
    assert derive_state(steps) == BLOCKED


def test_a_repeatedly_failing_step_blocks_the_objective_even_while_running():
    assert derive_state([_step(state="running", consecutive_failures=2)]) == BLOCKED


def test_an_objective_is_done_only_when_every_declared_step_is():
    """A step declared but never submitted must not count as finished."""
    assert derive_state([_step(), _step(step_id="two", submitted=False, state="")]) != "done"
    assert derive_state([_step(), _step(step_id="two")]) == "done"
