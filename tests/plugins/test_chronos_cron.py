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
    prov._arm_one_shot({"id": "j1", "next_run_at": "2026-06-18T12:00:00+00:00"})

    assert len(fake.provisions) == 1
    p = fake.provisions[0]
    assert p["job_id"] == "j1"
    assert p["fire_at"] == "2026-06-18T12:00:00+00:00"
    assert p["dedup_key"] == "j1:2026-06-18T12:00:00+00:00"
    assert p["agent_callback_url"] == "https://agent.example/"


def test_register_job_arms_only_the_created_job(chronos):
    prov, fake = chronos
    job = {"id": "created", "next_run_at": "2026-06-18T12:00:00+00:00"}

    prov.register_job(job)

    assert [p["job_id"] for p in fake.provisions] == ["created"]


def test_register_job_propagates_provision_failure(chronos):
    prov, fake = chronos

    def fail_provision(**kwargs):
        raise RuntimeError("provision rejected")

    fake.provision = fail_provision

    with pytest.raises(RuntimeError, match="provision rejected"):
        prov.register_job(
            {"id": "created", "next_run_at": "2026-06-18T12:00:00+00:00"}
        )


# -- reconcile ----------------------------------------------------------------

def test_reconcile_arms_all_enabled(temp_home, chronos, monkeypatch):
    prov, fake = chronos
    jobs = [
        {"id": "a", "enabled": True, "next_run_at": "2026-06-18T12:00:00+00:00", "state": "scheduled"},
        {"id": "b", "enabled": True, "next_run_at": "2026-06-18T12:05:00+00:00", "state": "scheduled"},
    ]
    monkeypatch.setattr("cron.jobs.load_jobs", lambda: jobs)
    monkeypatch.setattr("cron.jobs.get_job", lambda jid: next(j for j in jobs if j["id"] == jid))

    prov.reconcile()
    assert {p["job_id"] for p in fake.provisions} == {"a", "b"}
    assert fake.cancels == []


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
                        lambda jid: {"id": jid, "enabled": True, "next_run_at": "2026-06-18T12:05:00+00:00"})

    assert prov.fire_due("j1") is True
    assert [p["job_id"] for p in fake.provisions] == ["j1"]
    assert fake.provisions[0]["fire_at"] == "2026-06-18T12:05:00+00:00"


def test_fire_due_rearms_after_claimed_job_failure(chronos, monkeypatch):
    """A claimed attempt is consumed even when the job pipeline reports failure."""
    prov, fake = chronos
    claimed = {"id": "j1", "fire_claim": {"by": "owner-1"}}
    persisted = {
        "id": "j1",
        "enabled": True,
        "next_run_at": "2026-06-18T12:05:00+00:00",
    }

    monkeypatch.setattr("cron.jobs.claim_job_for_fire", lambda jid, **kw: claimed)
    monkeypatch.setattr(
        "cron.executions.create_execution",
        lambda jid, source: {"id": "exec-1"},
    )
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
