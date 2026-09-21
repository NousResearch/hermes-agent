"""Whether anything is going to run the work NOVA submitted.

The first real AWS workload test submitted an objective, got HTTP 200, four durable
tasks, ``refused: false`` — and then nothing. The control plane was deployed without a
dispatcher, because the NOVA image deliberately is not one: executing an agent needs the
full runtime and a model credential NOVA does not hold. Everything worked exactly as
designed, and the design had no way to say so.

These tests pin the two halves of the fix: NOVA can now prove the gap from the board, and
the deployment can now ship the process that closes it.

They run against a real ``kanban.db`` through the runtime's own API and the runtime's own
dispatcher, because "would a dispatcher have claimed this" is a property of that code and
a fake would assert my reading of it rather than its behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nova.apply import apply_bundle
from nova.audit import AuditLog
from nova.runtime.base import WorkExecutionHealth
from nova.spec import load_bundle
from nova.supervisor import submit_objective

from .conftest import EXAMPLE_BUNDLE

MODULE = Path(__file__).resolve().parents[2] / "deploy" / "aws"


class _Named:
    """Just enough of a runtime for a default contract method to name it."""

    def __init__(self, name: str) -> None:
        self.name = name


@pytest.fixture(scope="module", autouse=True)
def _requires_runtime():
    pytest.importorskip(
        "hermes_cli.kanban_db", reason="this asks the runtime's own dispatcher"
    )


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    from nova.runtime.hermes import HermesRuntime

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("NOVA_HOME", str(home))

    bundle = load_bundle(EXAMPLE_BUNDLE)
    runtime = HermesRuntime(home=home, tenant_id=bundle.tenant_id)
    audit = AuditLog(home / "audit.jsonl", tenant_id=bundle.tenant_id)
    apply_bundle(bundle, runtime, audit=audit)
    return bundle, runtime, audit


def submit(deployment, **kwargs):
    bundle, runtime, audit = deployment
    return submit_objective(
        bundle.objectives[0], bundle.agents, runtime, audit=audit,
        tenant_id=bundle.tenant_id, **kwargs
    )


# ---------------------------------------------------------------------------
# The diagnosis
# ---------------------------------------------------------------------------


def test_an_unobserved_board_is_not_reported_as_broken():
    """The guard that keeps this worth reading.

    A control plane that announced "no dispatcher" every time it had nothing to look at
    would be tuned out by the second week, and then the one real warning goes with it.
    """
    from nova.runtime.hermes import dispatch

    health = dispatch.work_execution_health(Path("/nonexistent-home"))
    assert not health.determined
    assert not health.stalled
    assert health.detail, "an undetermined answer still has to say why"


def test_ready_work_that_has_sat_past_a_poll_interval_proves_nothing_is_dispatching(
    deployment,
):
    """The exact symptom of the first workload test, reproduced against a real board.

    Two steps have no dependencies, so the runtime puts them straight in ``ready``. A
    dispatcher's whole job is that ``ready`` does not sit. Age them past three poll
    intervals and the absence is proven, not guessed.
    """
    from nova.runtime.hermes import dispatch

    _, runtime, _ = deployment
    report = submit(deployment)
    assert not report.refused

    home = runtime.paths.home
    fresh = dispatch.work_execution_health(home)
    assert fresh.ready_waiting >= 1, "the plan's independent steps should be claimable"
    assert not fresh.determined, "brand-new work proves nothing yet"

    import time

    aged = dispatch.work_execution_health(home, now=time.time() + 3600)
    assert aged.determined and not aged.attached
    assert aged.stalled
    assert aged.ready_waiting == fresh.ready_waiting
    assert "Nothing is claiming this board's work" in aged.detail
    assert aged.remedy, "a proven stall must name what the operator does about it"


def test_a_claim_by_the_real_dispatcher_flips_the_verdict_to_attached(deployment):
    """The other direction, and the one that proves the signal is not just an alarm.

    This runs the runtime's own ``dispatch_once`` against the board NOVA submitted to,
    with a spawn function standing in for the worker process. If that claims a task, the
    health reading must say something is attached — otherwise the deployment that fixes
    the outage would still be reported as broken.
    """
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    from nova.runtime.hermes import dispatch

    _, runtime, _ = deployment
    submit(deployment)
    home = runtime.paths.home

    spawned: list[str] = []

    def fake_spawn(task, workspace, *, board=None):
        spawned.append(task.id)
        return 424242  # a PID the dispatcher records; nothing is actually executed

    conn = kbc.connect()
    try:
        result = kbd.dispatch_once(conn, spawn_fn=fake_spawn, max_spawn=1)
        conn.commit()
    finally:
        conn.close()

    assert spawned, (
        "the runtime's own dispatcher claimed nothing from a board of ready work; "
        f"dispatch result: {result}"
    )

    health = dispatch.work_execution_health(home)
    assert health.attached and health.determined
    assert health.mechanism == "board-activity"
    assert not health.stalled


def test_the_default_contract_answers_rather_than_staying_silent():
    """An adapter that cannot see a dispatcher must say so, not return a cheerful zero."""
    from nova.runtime.base import AgentRuntime

    # Called unbound on the contract itself: the point is the default body, and
    # instantiating would require stubbing ten unrelated abstract methods whose shapes
    # this test would then be pinned to.
    health = AgentRuntime.work_execution_health(_Named("bare"))
    assert isinstance(health, WorkExecutionHealth)
    assert not health.attached and not health.determined
    assert "bare" in health.detail


# ---------------------------------------------------------------------------
# Saying it where somebody is looking
# ---------------------------------------------------------------------------


def test_submitting_onto_a_stalled_board_warns_instead_of_reporting_clean_success(
    deployment, monkeypatch,
):
    """The lie this whole change exists to remove.

    The first submission could not have known — the board had no history. The second one
    can, and must: by then two tasks have been sitting unclaimed, which is the proof.
    """
    from nova.runtime.hermes import dispatch as _dispatch

    first = submit(deployment)
    assert not first.refused

    # Age the board rather than sleep through three poll intervals.
    import time

    real = _dispatch.work_execution_health
    monkeypatch.setattr(
        _dispatch, "work_execution_health",
        lambda home, now=None: real(home, now=time.time() + 3600),
    )

    second = submit(deployment)
    assert any("nothing is running this work" in w for w in second.warnings), (
        f"submission onto a proven-stalled board reported clean success: {second.warnings}"
    )


def test_a_dry_run_does_not_warn_about_execution(deployment):
    """Planning is useful on a deployment with no worker; only a real write is a promise."""
    report = submit(deployment, dry_run=True)
    assert not any("nothing is running this work" in w for w in report.warnings)


def test_the_tasks_screen_says_whether_anything_is_running_the_work(deployment):
    """Same duty the Automations screen already discharges with ``scheduler_health``."""
    from nova.control.api import ControlAPI
    from nova.control.auth import Principal

    bundle, runtime, audit = deployment
    submit(deployment)
    api = ControlAPI(bundle=bundle, runtime=runtime, audit=audit)
    response = api.tasks({})
    assert response.status == 200
    execution = response.body["execution"]
    assert set(execution) >= {"attached", "determined", "ready_waiting", "detail"}
    assert execution["detail"]


# ---------------------------------------------------------------------------
# Shipping the process that closes it
# ---------------------------------------------------------------------------


def _user_data() -> str:
    return (MODULE / "user_data.sh.tftpl").read_text(encoding="utf-8")


def test_the_deployment_can_start_a_dispatcher():
    """Before this, deploy/aws had no way to run one at all — the whole outage."""
    body = _user_data()
    assert "nova-worker.service" in body, (
        "the deployment ships no dispatcher unit, so tasks it creates are never claimed"
    )
    assert "kanban daemon --force" in body, (
        "the worker unit does not run the dispatcher; a container that starts and idles "
        "looks healthy and claims nothing"
    )


def test_the_dispatcher_shares_the_control_plane_state_and_user():
    """Two processes, one board. Get either of these wrong and they silently disagree."""
    body = _user_data()
    assert "HERMES_HOME=${state_mount}/home" in body, (
        "the worker would read a different home than the control plane serves, so it "
        "would dispatch from an empty board"
    )
    assert "HERMES_UID=10001" in body, (
        "the runtime image runs as uid 10000 and the state volume is chowned to 10001; "
        "without the remap the worker cannot write the board"
    )


def test_the_dispatcher_is_declared_and_never_assumed():
    """An empty worker_image_uri deploys the control plane alone — the current state of
    the live deployment. It must stay a deliberate choice, not a silent default."""
    body = _user_data()
    assert 'if [ -n "${worker_image_uri}" ]; then' in body
    variables = (MODULE / "variables.tf").read_text(encoding="utf-8")
    assert 'variable "worker_image_uri"' in variables
    assert 'default     = ""' in variables


def test_the_control_plane_image_is_never_used_as_the_worker():
    """They are different images on purpose. Pointing the worker at the control-plane
    image would produce a container that cannot run an agent and says nothing useful."""
    body = _user_data()
    worker = body[body.index("nova-worker.service"):]
    assert "${image_uri}" not in worker, (
        "the worker unit references the control-plane image; it has neither the "
        "runtime's dependencies nor a model credential"
    )


def test_a_control_plane_only_bootstrap_does_not_fail_the_script():
    """`set -e` plus a bare `[ -f ] && ...` as the last command exits 1, and cloud-init
    reports the whole bootstrap failed on every deployment without a worker."""
    body = _user_data()
    assert "[ -f /etc/systemd/system/nova-worker.service ] &&" not in body
    assert "if [ -f /etc/systemd/system/nova-worker.service ]; then" in body
