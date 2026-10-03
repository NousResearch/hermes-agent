"""Unit tests for the Chronos NAS-mediated cron provider (Phase 4D).

All NAS calls are mocked — ZERO live network. These prove:
  - is_available is config-only (no network), false without config.
  - one-shot arming sends the right provision payload (incl. sub-minute fires —
    the agent owns the time, so there's no 1-minute floor).
  - reconcile arms missing, cancels orphaned, skips paused.
  - fire_due re-arms the next one-shot after a successful run, and repeat-N
    (job gone) stops re-arming.
"""

import pytest


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


def _future(**kw):
    """A timestamp safely in the future.

    These tests used to hardcode 2026-06-18 fire times. Arming now (correctly)
    refuses a fire_at in the past, so a frozen date would make them fail purely
    with the passage of time. The property under test is "a FUTURE fire is
    armed", so derive it rather than freezing it.
    """
    from datetime import datetime, timedelta, timezone

    kw = kw or {"hours": 6}
    return (datetime.now(timezone.utc) + timedelta(**kw)).isoformat()


FUTURE_A = _future(hours=6)
FUTURE_B = _future(hours=7)
FUTURE_C = _future(hours=8)


@pytest.fixture
def chronos(monkeypatch):
    """A ChronosCronScheduler with a fake NAS client capturing calls."""
    from plugins.cron_providers.chronos import ChronosCronScheduler

    class FakeClient:
        def __init__(self):
            self.provisions = []
            self.cancels = []
            self._armed = []

        def provision(self, *, job_id, fire_at, agent_callback_url, dedup_key):
            self.provisions.append({
                "job_id": job_id, "fire_at": fire_at,
                "agent_callback_url": agent_callback_url, "dedup_key": dedup_key,
            })
            return {"schedule_id": f"sched-{job_id}"}

        def cancel(self, *, job_id):
            self.cancels.append(job_id)
            return {}

        def list_armed(self):
            return list(self._armed)

    prov = ChronosCronScheduler()
    fake = FakeClient()
    prov._client = fake
    # callback_url is read via _cfg; patch the module helper to avoid config.
    monkeypatch.setattr("plugins.cron_providers.chronos._cfg",
                        lambda *k, default="": "https://agent.example/" if k[-1] == "callback_url" else "https://portal.test")
    return prov, fake


# -- is_available -------------------------------------------------------------

def test_is_available_false_without_config(temp_home, monkeypatch):
    from plugins.cron_providers.chronos import ChronosCronScheduler

    monkeypatch.setattr("plugins.cron_providers.chronos._cfg", lambda *k, default="": "")
    assert ChronosCronScheduler().is_available() is False


# -- arming -------------------------------------------------------------------

def test_arm_one_shot_sends_provision(chronos):
    prov, fake = chronos
    prov._arm_one_shot({"id": "j1", "next_run_at": FUTURE_A})

    assert len(fake.provisions) == 1
    p = fake.provisions[0]
    assert p["job_id"] == "j1"
    assert p["fire_at"] == FUTURE_A
    assert p["dedup_key"] == "j1:" + FUTURE_A
    assert p["agent_callback_url"] == "https://agent.example/"


def test_register_job_arms_only_the_created_job(chronos):
    prov, fake = chronos
    job = {"id": "created", "next_run_at": FUTURE_A}

    prov.register_job(job)

    assert [p["job_id"] for p in fake.provisions] == ["created"]


def test_register_job_propagates_provision_failure(chronos):
    prov, fake = chronos

    def fail_provision(**kwargs):
        raise RuntimeError("provision rejected")

    fake.provision = fail_provision

    with pytest.raises(RuntimeError, match="provision rejected"):
        prov.register_job(
            {"id": "created", "next_run_at": FUTURE_A}
        )


# -- reconcile ----------------------------------------------------------------

def test_identity_rejection_hands_fires_to_the_builtin_ticker(temp_home, chronos, monkeypatch, caplog):
    """Regression for #97494: NAS answering 403 invalid_client is a deterministic identity rejection
    (the auth.json token is not this instance's provisioned agent), not a transient. The provider
    must say so ONCE with the remedy, stop calling NAS, and start the built-in ticker so jobs still
    fire on time instead of only via the late misfire sweep."""
    import threading

    from plugins.cron_providers.chronos._nas_client import NasCronClientError

    prov, fake = chronos
    calls = []

    def rejected(**kw):
        calls.append(kw["job_id"])
        raise NasCronClientError(
            "POST /api/agent-cron/provision returned 403: invalid_client",
            status=403, error_code="invalid_client")

    fake.provision = rejected
    jobs = [
        {"id": "a", "enabled": True, "next_run_at": FUTURE_A, "state": "scheduled"},
        {"id": "b", "enabled": True, "next_run_at": FUTURE_B, "state": "scheduled"},
    ]
    monkeypatch.setattr("cron.jobs.load_jobs", lambda: jobs)
    monkeypatch.setattr("cron.jobs.get_job", lambda jid: next(j for j in jobs if j["id"] == jid))
    monkeypatch.setattr("cron.executions.recover_interrupted_executions", lambda: 0)
    ticker_started = threading.Event()
    monkeypatch.setattr(
        "cron.scheduler_provider.InProcessCronScheduler.start",
        lambda self, stop_event, **kw: ticker_started.set())

    stop = threading.Event()
    with caplog.at_level("WARNING", logger="cron.chronos"):
        prov.start(stop, adapters={"x": 1}, loop=None, interval=7)
    assert ticker_started.wait(2.0), "the built-in ticker must take over this process's fires"
    assert calls == ["a"], "after the first rejection NAS is left alone"
    identity_msgs = [r.message for r in caplog.records if "re-login" in r.message]
    assert len(identity_msgs) == 1 and "built-in cron ticker" in identity_msgs[0]

    # Job creation and re-arms no longer reach NAS (and no longer fail the create).
    prov.register_job({"id": "c", "next_run_at": FUTURE_C})
    prov.on_jobs_changed()
    assert calls == ["a"]


def test_transient_provision_failure_does_not_degrade(temp_home, chronos, monkeypatch):
    """A 5xx / transport error is retried on the next reconcile; only 403 invalid_client degrades."""
    from plugins.cron_providers.chronos._nas_client import NasCronClientError

    prov, fake = chronos
    attempts = []

    def flaky(**kw):
        attempts.append(kw["job_id"])
        raise NasCronClientError("POST /api/agent-cron/provision returned 502: upstream", status=502)

    fake.provision = flaky
    jobs = [{"id": "a", "enabled": True, "next_run_at": FUTURE_A, "state": "scheduled"}]
    monkeypatch.setattr("cron.jobs.load_jobs", lambda: jobs)
    monkeypatch.setattr("cron.jobs.get_job", lambda jid: jobs[0])

    prov.reconcile()
    prov.reconcile()
    assert attempts == ["a", "a"] and prov._identity_rejected is False


def test_reconcile_arms_all_enabled(temp_home, chronos, monkeypatch):
    prov, fake = chronos
    jobs = [
        {"id": "a", "enabled": True, "next_run_at": FUTURE_A, "state": "scheduled"},
        {"id": "b", "enabled": True, "next_run_at": FUTURE_B, "state": "scheduled"},
    ]
    monkeypatch.setattr("cron.jobs.load_jobs", lambda: jobs)
    monkeypatch.setattr("cron.jobs.get_job", lambda jid: next(j for j in jobs if j["id"] == jid))

    prov.reconcile()
    assert {p["job_id"] for p in fake.provisions} == {"a", "b"}
    assert fake.cancels == []


def test_reconcile_advances_a_missed_recurring_job_before_arming(temp_home, chronos):
    """Skipping a stale fire must not permanently unschedule the recurring job."""
    from datetime import datetime, timedelta, timezone

    from cron.jobs import create_job, get_job, load_jobs, save_jobs

    prov, fake = chronos
    job = create_job(prompt="Recurring check", schedule="every 8h")
    jobs = load_jobs()
    jobs[0]["next_run_at"] = (
        datetime.now(timezone.utc) - timedelta(hours=3)
    ).isoformat()
    save_jobs(jobs)

    before = datetime.now(timezone.utc)
    prov.reconcile()

    assert len(fake.provisions) == 1
    fire_at = fake.provisions[0]["fire_at"]
    assert datetime.fromisoformat(fire_at) > before
    stored = get_job(job["id"])
    assert stored is not None
    assert stored["next_run_at"] == fire_at


# -- fire_due re-arm ----------------------------------------------------------

def test_fire_due_rearms_next_oneshot(chronos, monkeypatch):
    prov, fake = chronos
    # Keep the two-phase provider flow intact while stubbing durable admission
    # and the shared runner body.
    monkeypatch.setattr(
        "cron.scheduler_provider.CronScheduler.claim_fire",
        lambda self, jid, **kw: {"id": jid, "execution_id": "exec-1"},
    )
    monkeypatch.setattr(
        "cron.scheduler_provider.CronScheduler.fire_claimed",
        lambda self, job, **kw: True,
    )
    monkeypatch.setattr("cron.jobs.get_job",
                        lambda jid: {"id": jid, "enabled": True, "next_run_at": FUTURE_B})

    assert prov.fire_due("j1") is True
    assert [p["job_id"] for p in fake.provisions] == ["j1"]
    assert fake.provisions[0]["fire_at"] == FUTURE_B


def test_fire_due_rearms_after_claimed_job_failure(chronos, monkeypatch, tmp_path):
    """A claimed attempt is consumed even when the job pipeline reports failure."""
    import cron.executions as executions

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "executions.db")
    prov, fake = chronos
    claimed = {"id": "j1", "fire_claim": {"by": "owner-1"}}
    persisted = {
        "id": "j1",
        "enabled": True,
        "next_run_at": FUTURE_B,
    }

    monkeypatch.setattr("cron.jobs.claim_job_for_fire", lambda jid, **kw: claimed)
    monkeypatch.setattr("cron.scheduler.run_one_job", lambda *args, **kwargs: False)
    monkeypatch.setattr("cron.jobs.get_job", lambda jid: persisted)

    assert prov.fire_due("j1") is True
    assert [provision["job_id"] for provision in fake.provisions] == ["j1"]


def test_fire_due_forwards_manual_force_to_claim(chronos, monkeypatch):
    """A manual force fire must reach the store claim as force=True."""
    prov, _fake = chronos
    seen = []
    monkeypatch.setattr(
        "cron.jobs.claim_job_for_fire",
        lambda jid, **kw: seen.append(kw) or False,
    )
    monkeypatch.setattr(
        "cron.executions.create_execution",
        lambda jid, source: {"id": "exec-1"},
    )

    assert prov.fire_due("j1", force=True) is False
    assert seen == [{"return_job": True, "force": True}]


def test_fire_due_no_rearm_when_job_gone(chronos, monkeypatch):
    """repeat-N exhausted / one-shot completed → mark_job_run deleted the job →
    get_job None → no re-arm (the schedule stops cleanly)."""
    prov, fake = chronos
    monkeypatch.setattr("cron.scheduler_provider.CronScheduler.fire_due",
                        lambda self, jid, **kw: True)
    monkeypatch.setattr("cron.jobs.get_job", lambda jid: None)

    assert prov.fire_due("j1") is True
    assert fake.provisions == []


def test_fire_due_no_rearm_when_claim_lost(chronos, monkeypatch):
    """If the run didn't happen (claim lost), don't re-arm."""
    prov, fake = chronos
    monkeypatch.setattr("cron.scheduler_provider.CronScheduler.fire_due",
                        lambda self, jid, **kw: False)

    assert prov.fire_due("j1") is False
    assert fake.provisions == []


# -- provider capability classification ----------------------------------------

def test_chronos_is_split_fire_capable(chronos):
    """Regression: Chronos must be classified as a split-aware provider so the
    fire webhook uses durable claim admission (not the legacy fire_due path).
    Chronos deliberately has NO fire_due override — its re-arm logic lives in
    fire_claimed, which the split path invokes."""
    from cron.scheduler_provider import provider_supports_force_fire, provider_supports_split_fire

    prov, _fake = chronos
    assert provider_supports_split_fire(prov) is True
    assert provider_supports_force_fire(prov) is True


def test_fire_claimed_no_rearm_when_run_failed(chronos, monkeypatch):
    prov, fake = chronos
    monkeypatch.setattr(
        "cron.scheduler_provider.CronScheduler.fire_claimed",
        lambda self, job, **kw: False,
    )

    assert prov.fire_claimed({"id": "j1"}) is False
    assert fake.provisions == []


def test_fire_claimed_no_rearm_when_job_gone(chronos, monkeypatch):
    prov, fake = chronos
    monkeypatch.setattr(
        "cron.scheduler_provider.CronScheduler.fire_claimed",
        lambda self, job, **kw: True,
    )
    monkeypatch.setattr("cron.jobs.get_job", lambda jid: None)

    assert prov.fire_claimed({"id": "j1"}) is True
    assert fake.provisions == []
# -- past fire_at must never be armed (re-fire-on-restart guard) --------------


def test_arm_one_shot_refuses_a_fire_at_in_the_past(chronos):
    """A next_run_at already in the PAST must not be armed.

    ROOT CAUSE (2026-08-11): reconcile() runs on gateway boot and arms every
    enabled job at its stored next_run_at, with no check that the time is still
    in the future. A job whose next_run_at had already passed got armed with a
    past fire_at, and the external scheduler fired it IMMEDIATELY -- so a
    scheduled job re-ran minutes after a restart. Observed live: a `0 */8` job
    delivered 7 messages in ~17h, 3 of them re-fires whose own "in the last
    0.1h/0.4h" delta proved they followed a real run.

    A past one-shot is a MISSED fire, not a due one. The next legitimate fire is
    already represented by the schedule; arming the stale one only duplicates it.
    """
    from datetime import datetime, timedelta, timezone

    prov, fake = chronos
    past = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
    prov._arm_one_shot({"id": "stale", "next_run_at": past})

    assert fake.provisions == [], (
        "armed a one-shot whose fire_at is in the past -- the external scheduler "
        "fires those immediately, which is the re-fire-on-restart bug"
    )
    assert "stale" not in prov._armed


def test_arm_one_shot_still_arms_a_future_fire_at(chronos):
    """The guard must not break normal arming."""
    from datetime import datetime, timedelta, timezone

    prov, fake = chronos
    future = (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat()
    prov._arm_one_shot({"id": "ok", "next_run_at": future})

    assert [p["job_id"] for p in fake.provisions] == ["ok"]
    assert prov._armed["ok"] == future


def test_arm_one_shot_arms_a_fire_at_inside_the_grace_window(chronos):
    """A fire_at a few seconds in the past is CLOCK SKEW, not a missed fire.

    Arming must stay tolerant there, or a job whose fire_at passed while the
    provision request was in flight would be silently dropped.
    """
    from datetime import datetime, timedelta, timezone

    prov, fake = chronos
    just_now = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
    prov._arm_one_shot({"id": "skew", "next_run_at": just_now})

    assert [p["job_id"] for p in fake.provisions] == ["skew"]


def test_arm_one_shot_tolerates_an_unparseable_fire_at(chronos):
    """A malformed timestamp must ARM (fail-open), never crash reconcile.

    Dropping it would silently unschedule the job; reconcile is the only thing
    that re-arms, so a raised exception there stops every later job in the loop.
    """
    prov, fake = chronos
    prov._arm_one_shot({"id": "weird", "next_run_at": "not-a-timestamp"})
    assert [p["job_id"] for p in fake.provisions] == ["weird"]
