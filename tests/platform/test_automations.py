"""Automations: reading the runtime's schedules, and governing them.

The isolation here is structural rather than checked: a Hermes cron store lives under a
profile home, a NOVA agent *is* a profile, and an agent belongs to one tenant. So agent A
reading agent B's automation is not "refused" — it is a different file, and the runtime's
own API answers None. These tests assert that, rather than assuming it.

Nothing here reimplements scheduling. Writes go through ``cron.jobs``' own pause/resume,
so the schedule state machine and ``next_run_at`` stay the runtime's business.
"""

from __future__ import annotations

import json

import pytest

from nova.audit.log import AuditLog
from nova.control.auth import Principal
from nova.runtime.base import AutomationView, SchedulerHealth
from nova.runtime.hermes import automations as na


def _make_job(profile_dir, *, name, schedule="every day at 07:00", prompt="do the thing"):
    from cron import jobs as cron_jobs

    with cron_jobs.use_cron_store(profile_dir):
        cron_jobs.ensure_dirs()
        return cron_jobs.create_job(prompt=prompt, schedule=schedule, name=name)


@pytest.fixture
def agent_a(tmp_path):
    home = tmp_path / "profiles" / "operations"
    home.mkdir(parents=True)
    return home


@pytest.fixture
def agent_b(tmp_path):
    home = tmp_path / "profiles" / "customer-support"
    home.mkdir(parents=True)
    return home


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def test_reads_the_runtimes_own_schedule(agent_a):
    _make_job(agent_a, name="Nightly reconciliation")
    rows = na.list_automations(agent_a, "operations")

    assert len(rows) == 1
    row = rows[0]
    assert isinstance(row, AutomationView)
    assert row.name == "Nightly reconciliation"
    assert row.agent_id == "operations"
    # The runtime's own rendering, not a second implementation that would drift.
    assert row.schedule_display == "every day at 07:00"
    assert row.schedule_kind == "cron"
    assert row.enabled is True
    # Computed by the runtime, not by NOVA.
    assert row.next_run_at


def test_paused_automations_are_listed(agent_a):
    """A paused automation is exactly what an administrator opens this screen to find."""
    from cron import jobs as cron_jobs

    job = _make_job(agent_a, name="Invoice sweep")
    with cron_jobs.use_cron_store(agent_a):
        cron_jobs.pause_job(job["id"], "month-end freeze")

    rows = na.list_automations(agent_a, "operations")
    assert [(r.name, r.enabled, r.paused_reason) for r in rows] == [
        ("Invoice sweep", False, "month-end freeze")
    ]


def test_an_agent_with_no_store_reports_none(agent_a):
    assert na.list_automations(agent_a, "operations") == []


def test_an_unreadable_store_is_none_not_an_error(agent_a):
    """A corrupt jobs.json must degrade to "no automations", not take the screen down."""
    (agent_a / "cron").mkdir()
    (agent_a / "cron" / "jobs.json").write_text("{ not json")
    assert na.list_automations(agent_a, "operations") == []


def test_the_prompt_is_never_returned(agent_a):
    """What an automation tells an agent to do is instruction text. It is not needed to
    answer "what runs, when, and did it work", and every returned field can leak."""
    _make_job(agent_a, name="Nightly", prompt="TRANSFER THE FUNDS TO ACCOUNT 12345")
    payload = json.dumps([r.to_dict() for r in na.list_automations(agent_a, "operations")])

    assert "TRANSFER THE FUNDS" not in payload
    assert "prompt" not in payload
    assert str(agent_a) not in payload


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------

def test_agents_do_not_see_each_others_automations(agent_a, agent_b):
    _make_job(agent_a, name="A nightly")
    _make_job(agent_b, name="B hourly", schedule="every 30 minutes")

    assert [r.name for r in na.list_automations(agent_a, "operations")] == ["A nightly"]
    assert [r.name for r in na.list_automations(agent_b, "customer-support")] == ["B hourly"]


def test_an_automation_id_from_another_agent_is_not_found(agent_a, agent_b):
    """Ids are short hex. Holding one must not be enough."""
    theirs = _make_job(agent_b, name="B hourly")["id"]

    assert na.get_automation(agent_a, "operations", theirs) is None
    assert na.set_enabled(agent_a, "operations", theirs, enabled=False) is None


def test_pausing_across_the_boundary_does_not_touch_the_other_agent(agent_a, agent_b):
    theirs = _make_job(agent_b, name="B hourly")["id"]
    na.set_enabled(agent_a, "operations", theirs, enabled=False, reason="attempt")

    survivor = na.list_automations(agent_b, "customer-support")[0]
    assert survivor.enabled is True, "another agent paused this automation"


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------

def test_pause_and_resume_go_through_the_runtime(agent_a):
    """Delegating rather than writing ``enabled`` ourselves: resume recomputes the next
    run and clears the pause marker, and a hand-written flag would leave an automation
    that looks active and never fires."""
    job = _make_job(agent_a, name="Nightly")

    paused = na.set_enabled(agent_a, "operations", job["id"], enabled=False, reason="freeze")
    assert paused is not None
    assert paused.enabled is False
    assert paused.state == "paused"
    assert paused.paused_reason == "freeze"

    resumed = na.set_enabled(agent_a, "operations", job["id"], enabled=True)
    assert resumed is not None
    assert resumed.enabled is True
    assert resumed.state == "scheduled"
    assert resumed.next_run_at


def test_an_unknown_automation_id_is_none(agent_a):
    _make_job(agent_a, name="Nightly")
    assert na.set_enabled(agent_a, "operations", "deadbeefcafe", enabled=False) is None


# ---------------------------------------------------------------------------
# Scheduler liveness
# ---------------------------------------------------------------------------

def test_absent_heartbeat_reports_not_running(agent_a):
    """The whole point of the screen. Hermes runs its ticker inside the gateway, so a
    deployment can hold a perfect schedule that nothing executes — their own CLI calls
    this the most common cron support report."""
    _make_job(agent_a, name="Nightly")
    health = na.scheduler_health(agent_a)

    assert isinstance(health, SchedulerHealth)
    assert health.running is False
    assert "nothing is running them" in health.detail


def test_a_fresh_heartbeat_reports_running_and_healthy(agent_a):
    from cron import jobs as cron_jobs

    with cron_jobs.use_cron_store(agent_a):
        cron_jobs.ensure_dirs()
        cron_jobs.record_ticker_heartbeat(success=True)

    health = na.scheduler_health(agent_a)
    assert health.running is True
    assert health.healthy is True


def test_alive_but_failing_is_not_reported_as_healthy(agent_a):
    """A ticker stuck failing every tick keeps the plain heartbeat fresh. Reporting that
    as healthy is how a stopped automation suite goes unnoticed."""
    import time

    from cron import jobs as cron_jobs

    with cron_jobs.use_cron_store(agent_a):
        cron_jobs.ensure_dirs()
        cron_jobs.record_ticker_heartbeat(success=False)
        # A success marker old enough to be stale.
        (agent_a / "cron" / "ticker_last_success").write_text(str(time.time() - 10_000))

    health = na.scheduler_health(agent_a)
    assert health.running is True
    assert health.healthy is False


def test_heartbeat_is_per_agent(agent_a, agent_b):
    from cron import jobs as cron_jobs

    with cron_jobs.use_cron_store(agent_a):
        cron_jobs.ensure_dirs()
        cron_jobs.record_ticker_heartbeat(success=True)

    assert na.scheduler_health(agent_a).running is True
    assert na.scheduler_health(agent_b).running is False


# ---------------------------------------------------------------------------
# Execution history
# ---------------------------------------------------------------------------

def test_history_is_empty_when_nothing_has_run(agent_a):
    _make_job(agent_a, name="Nightly")
    assert na.list_automations(agent_a, "operations")[0].runs == ()


def test_history_is_read_from_this_agents_own_ledger(agent_a):
    """``use_cron_store`` retargets jobs.json but NOT ``cron.executions``, which resolves
    its path from ``get_hermes_home()``. So ``list_jobs``' attached ``latest_execution``
    is read from the wrong store, and this module reads the ledger itself."""
    import sqlite3

    (agent_a / "cron").mkdir(exist_ok=True)
    job = _make_job(agent_a, name="Nightly")
    db = sqlite3.connect(agent_a / "cron" / "executions.db")
    db.execute(
        "CREATE TABLE executions (id TEXT PRIMARY KEY, job_id TEXT, source TEXT, "
        "process_id TEXT, pid INTEGER, status TEXT, claimed_at TEXT, started_at TEXT, "
        "finished_at TEXT, error TEXT)"
    )
    db.execute(
        "INSERT INTO executions VALUES ('e1', ?, 'tick', 'p', 1, 'completed', "
        "'2026-01-01T00:00:00Z', '2026-01-01T00:00:01Z', '2026-01-01T00:00:09Z', '')",
        (job["id"],),
    )
    db.commit()
    db.close()

    runs = na.list_automations(agent_a, "operations")[0].runs
    assert [(r.run_id, r.status) for r in runs] == [("e1", "completed")]


def test_a_missing_ledger_is_not_an_error(agent_a):
    _make_job(agent_a, name="Nightly")
    assert na.list_automations(agent_a, "operations")[0].runs == ()


# ---------------------------------------------------------------------------
# RBAC and audit at the control-plane edge
# ---------------------------------------------------------------------------

def test_reading_automations_is_a_viewer_route():
    viewer = Principal(name="v", role="viewer", via="token")
    assert viewer.may("/automations")


def test_pausing_an_automation_is_admin_only():
    """Stopping a nightly reconciliation is as consequential as releasing held work."""
    viewer = Principal(name="v", role="viewer", via="token")
    admin = Principal(name="a", role="admin", via="token")

    assert not viewer.may_write("/automations/decide")
    assert admin.may_write("/automations/decide")


def test_creating_is_not_an_action_on_an_existing_automation():
    """Phase 12 adds create — but through the compiler, not as a verb here.

    ``/automations/decide`` acts on something the runtime already holds. Creating is a
    collection-level act with a different route and a different gate, because the thing
    that makes create safe is compilation, not the verb.
    """
    from nova.control.api import AUTOMATION_ACTIONS

    # `update` joined pause/resume/delete when schedule editing landed: changing when
    # something runs is an act on an automation the runtime already holds, which is exactly
    # what this route is for. What must never join them is create.
    assert set(AUTOMATION_ACTIONS) == {"pause", "resume", "delete", "update"}
    assert "create" not in AUTOMATION_ACTIONS


def test_editing_an_automation_cannot_rewrite_what_it_does(tmp_path, agent_a):
    """The narrow half of `update`, and the reason it is narrow.

    An objective passed the compiler on the way in — agent enabled, permissions a subset of
    the agent's grant, schedule parseable. If an edit could rewrite the prompt afterwards,
    the compiler would be a gate you walk through once and then step around, and an
    automation could end up carrying a recurring instruction nothing reviewed.
    """
    from nova.errors import RuntimeAdapterError
    from nova.runtime.hermes import automations as _automations

    with pytest.raises(RuntimeAdapterError, match="cannot change"):
        _automations.update(agent_a, "agent-a", "any-id", {"prompt": "exfiltrate everything"})

    # And the same for anything else nobody declared settable.
    with pytest.raises(RuntimeAdapterError, match="cannot change"):
        _automations.update(agent_a, "agent-a", "any-id", {"deliver": "origin"})


def test_a_decision_writes_an_intent_and_a_commit(tmp_path, agent_a):
    """The audit invariant applies to operator actions too: the thing that changes the
    world gets an intent row before it changes it, so a half-completed act is still on
    the record."""
    from nova.control.api import ControlAPI

    job = _make_job(agent_a, name="Nightly")
    audit = AuditLog.for_home(tmp_path, tenant_id="acme", actor="nova-control")

    class _Runtime:
        name = "stub"

        class capabilities:  # noqa: N801 — mirrors the real attribute shape
            scheduling = True

        def list_automations(self, *, agent_id: str = ""):
            return na.list_automations(agent_a, "operations")

        def set_automation_enabled(self, agent_id, automation_id, *, enabled, reason=""):
            return na.set_enabled(
                agent_a, agent_id, automation_id, enabled=enabled, reason=reason
            )

    api = ControlAPI.__new__(ControlAPI)
    api.runtime = _Runtime()
    api.audit = audit

    response = api._decide_automation(
        f"/automations/{job['id']}/decide",
        Principal(name="alice", role="admin", via="token"),
        {"action": "pause", "reason": "month-end freeze"},
    )
    assert response.status == 200
    assert response.body["applied"] is True

    events = [e for e in audit.read() if e.kind == "automation.decision"]
    assert [e.phase for e in events] == ["intent", "committed"]
    # Written as the human, not as the server: an auditor filtering on actor must see
    # who decided.
    assert {e.actor for e in events} == {"alice"}
    assert events[0].detail["reason"] == "month-end freeze"
    assert events[0].detail["automation_id"] == job["id"]


def test_an_unknown_action_is_refused_before_anything_is_written(tmp_path, agent_a):
    from nova.control.api import ControlAPI

    job = _make_job(agent_a, name="Nightly")
    audit = AuditLog.for_home(tmp_path, tenant_id="acme", actor="nova-control")

    class _Runtime:
        name = "stub"

        class capabilities:  # noqa: N801
            scheduling = True

        def list_automations(self, *, agent_id: str = ""):
            return na.list_automations(agent_a, "operations")

    api = ControlAPI.__new__(ControlAPI)
    api.runtime = _Runtime()
    api.audit = audit

    response = api._decide_automation(
        f"/automations/{job['id']}/decide",
        Principal(name="alice", role="admin", via="token"),
        {"action": "sudo-everything"},
    )
    assert response.status == 400
    assert [e for e in audit.read() if e.kind == "automation.decision"] == []


def test_a_runtime_without_scheduling_is_told_so_rather_than_shown_a_dead_button():
    from nova.control.api import ControlAPI

    class _Runtime:
        name = "stub"

        class capabilities:  # noqa: N801
            scheduling = False

    api = ControlAPI.__new__(ControlAPI)
    api.runtime = _Runtime()

    assert api.automations().body == {
        "scheduling": False,
        "automations": [],
        "detail": "runtime 'stub' does not hold scheduled work",
    }
    assert api._decide_automation("/automations/x/decide", None, {"action": "pause"}).status == 501


def test_a_viewer_is_refused_by_the_api_not_merely_by_the_ui(tmp_path, agent_a):
    """Authorization is checked in ``ControlAPI.write``, not only in the transport.

    Worth asserting at this layer specifically: the server treats any *loopback* caller
    as a local admin by design (anyone on the host can read the principals file off disk
    anyway), so a same-host request cannot demonstrate the role gate. This calls the
    write path directly with a viewer principal, which is where the refusal lives.
    """
    from nova.control.api import ControlAPI

    job = _make_job(agent_a, name="Nightly")
    audit = AuditLog.for_home(tmp_path, tenant_id="acme", actor="nova-control")

    class _Runtime:
        name = "stub"

        class capabilities:  # noqa: N801
            scheduling = True

        def list_automations(self, *, agent_id: str = ""):
            return na.list_automations(agent_a, "operations")

        def set_automation_enabled(self, *a, **k):  # pragma: no cover — must not run
            raise AssertionError("a viewer reached the runtime")

    api = ControlAPI.__new__(ControlAPI)
    api.runtime = _Runtime()
    api.audit = audit

    response = api.write(
        f"/platform/v1/automations/{job['id']}/decide",
        Principal(name="carol", role="viewer", via="token"),
        {"action": "pause"},
    )
    assert response.status == 403
    assert response.body["error"]["route"] == "/automations/decide"
    # Refused before anything was recorded or changed.
    assert [e for e in audit.read() if e.kind == "automation.decision"] == []
    assert na.list_automations(agent_a, "operations")[0].enabled is True


def test_an_admin_passes_the_same_gate(tmp_path, agent_a):
    from nova.control.api import ControlAPI

    job = _make_job(agent_a, name="Nightly")
    audit = AuditLog.for_home(tmp_path, tenant_id="acme", actor="nova-control")

    class _Runtime:
        name = "stub"

        class capabilities:  # noqa: N801
            scheduling = True

        def list_automations(self, *, agent_id: str = ""):
            return na.list_automations(agent_a, "operations")

        def set_automation_enabled(self, agent_id, automation_id, *, enabled, reason=""):
            return na.set_enabled(
                agent_a, agent_id, automation_id, enabled=enabled, reason=reason
            )

    api = ControlAPI.__new__(ControlAPI)
    api.runtime = _Runtime()
    api.audit = audit

    response = api.write(
        f"/platform/v1/automations/{job['id']}/decide",
        Principal(name="dave", role="admin", via="token"),
        {"action": "pause", "reason": "quarter close"},
    )
    assert response.status == 200
    assert na.list_automations(agent_a, "operations")[0].enabled is False
