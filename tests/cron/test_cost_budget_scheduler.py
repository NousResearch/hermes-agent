"""Cost budget enforcement in the cron scheduler (fork-only).

The scheduler itself skips a discretionary job while its profile is over
budget — before the job's script or agent runs — and records the skip in the
executions ledger. Non-discretionary work keeps running. Every over-budget
preflight raises the one deduplicated over-budget alert for the period.
"""
from datetime import datetime, timedelta, timezone

import pytest

import cron.scheduler as s
from agent import cost_budget
from cron import executions, jobs
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture(autouse=True)
def _reset():
    cost_budget._reset_for_tests()
    yield
    cost_budget._reset_for_tests()


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


def _budget(home, extra=""):
    (home / "config.yaml").write_text(
        "cost_budget:\n  daily_usd: 10\n  monthly_usd: 100\n  alert_channel: slack:C1\n"
        "  discretionary_jobs: [abc]\n" + extra,
        encoding="utf-8",
    )


def _spend(monkeypatch, *, by_home=None, day=0.0, month=0.0, calls=None):
    """Stub the scan. Keys follow the ``now`` it is given, so the cache is really used."""

    def fake(home, now):
        if calls is not None:
            calls.append(str(home))
        d, m = (by_home or {}).get(str(home), (day, month))
        day_key, month_key = now.strftime("%Y-%m-%d"), now.strftime("%Y-%m")
        return cost_budget.SpendSnapshot(
            day_key=day_key, month_key=month_key, day_usd=d, month_usd=m,
            day_unpriced_tokens=0, month_unpriced_tokens=0, unpriced_models=(), runs=(),
            as_of=now.timestamp())

    monkeypatch.setattr(cost_budget, "compute_spend", fake)


@pytest.fixture()
def pipeline(monkeypatch):
    calls = {"run_job": [], "deliver": [], "mark": [], "skipped": []}

    def fake_run_job(job, *, defer_agent_teardown=None, **kw):
        calls["run_job"].append(job["id"])
        return True, "out", "final response", None

    def fake_deliver(job, content, adapters=None, loop=None):
        calls["deliver"].append({"job": job, "content": content, "adapters": adapters})
        return None

    def fake_mark(jid, ok, err=None, delivery_error=None, **kw):
        calls["mark"].append((jid, ok, kw.get("status")))
        return True

    def fake_skipped(jid, status, reason, **kw):
        calls["skipped"].append((jid, status, reason, kw.get("retry_at")))
        return True

    monkeypatch.setattr(s, "run_job", fake_run_job)
    monkeypatch.setattr(s, "save_job_output", lambda jid, out: f"/tmp/{jid}.txt")
    monkeypatch.setattr(s, "_deliver_result", fake_deliver)
    monkeypatch.setattr(s, "_resolve_delivery_targets",
                        lambda job: [{"platform": "slack", "chat_id": "C1"}])
    monkeypatch.setattr(s, "mark_job_run", fake_mark)
    monkeypatch.setattr(jobs, "mark_job_skipped", fake_skipped)
    return calls


def _alerts(pipeline):
    cost_budget._wait_for_sends()
    return [d for d in pipeline["deliver"] if d["job"]["id"] == "cost_budget"]


def test_discretionary_job_skipped_when_over_budget(home, monkeypatch, pipeline):
    _budget(home)
    _spend(monkeypatch, day=12, month=40)

    assert s.run_one_job({"id": "j-exp", "name": "x", "discretionary": True}) is True

    assert pipeline["run_job"] == []
    assert pipeline["mark"] == []
    ((jid, status, reason, retry_at),) = pipeline["skipped"]
    assert (jid, status) == ("j-exp", "skipped_budget")
    assert "daily $12.00 of $10.00" in reason
    assert retry_at is not None
    record = executions.latest_execution("j-exp")
    assert record["status"] == "completed"
    assert record["error"].startswith("skipped: over budget")
    # The skip is never silent: the over-budget alert goes out, naming the job.
    (alert,) = _alerts(pipeline)
    assert "`x` was skipped" in alert["content"]


def test_listed_job_id_skipped_on_monthly(home, monkeypatch, pipeline):
    _budget(home)
    _spend(monkeypatch, day=1, month=150)

    s.run_one_job({"id": "abc", "name": "explore"})

    assert pipeline["run_job"] == []
    assert "monthly $150.00 of $100.00" in executions.latest_execution("abc")["error"]


def test_name_does_not_make_a_job_discretionary(home, monkeypatch, pipeline):
    _budget(home)
    _spend(monkeypatch, day=12, month=40)

    s.run_one_job({"id": "zzz", "name": "abc"})

    assert pipeline["run_job"] == ["zzz"]


def test_over_budget_alert_once_per_period_across_jobs(home, monkeypatch, pipeline):
    _budget(home, "  alert_mention: '<!subteam^S1>'\n")
    _spend(monkeypatch, day=12, month=40)
    adapters = {"slack": object()}

    s.run_one_job({"id": "rel", "name": "release-watch"}, adapters=adapters)
    s.run_one_job({"id": "rel2", "name": "other-release"}, adapters=adapters)
    s.run_one_job({"id": "e", "name": "explore", "discretionary": True}, adapters=adapters)
    cost_budget._wait_for_sends()
    cost_budget._last_watch.clear()
    s.tick(verbose=False, adapters=adapters)

    assert pipeline["run_job"] == ["rel", "rel2"]
    alerts = _alerts(pipeline)
    assert len(alerts) == 1
    assert alerts[0]["job"]["deliver"] == "slack:C1"
    assert alerts[0]["adapters"] is adapters
    assert alerts[0]["content"].startswith("<!subteam^S1> 🚨")
    assert "release-watch" in alerts[0]["content"]


def test_under_budget_runs(home, monkeypatch, pipeline):
    _budget(home)
    _spend(monkeypatch, day=2, month=20)

    s.run_one_job({"id": "abc", "name": "explore"})

    assert pipeline["run_job"] == ["abc"]
    assert pipeline["mark"] == [("abc", True, None)]


def test_spend_cached_across_preflights(home, monkeypatch, pipeline):
    _budget(home)
    calls = []
    _spend(monkeypatch, day=2, month=20, calls=calls)

    for _ in range(3):
        s.run_one_job({"id": "abc", "name": "explore"})

    assert len(calls) == 1


def test_no_block_unchanged(home, monkeypatch, pipeline):
    monkeypatch.setattr(cost_budget, "compute_spend", lambda *a, **k: pytest.fail("spend queried"))

    s.run_one_job({"id": "e", "name": "explore", "discretionary": True})

    assert pipeline["run_job"] == ["e"]
    assert not (home / "cost_budget").exists()


def test_preflight_error_fails_open(home, monkeypatch, pipeline):
    _budget(home)

    def boom(*a, **k):
        raise RuntimeError("store exploded")

    monkeypatch.setattr(cost_budget, "compute_spend", boom)

    s.run_one_job({"id": "abc", "name": "explore"})

    assert pipeline["run_job"] == ["abc"]


def test_lost_fire_claim_on_skip_is_not_recorded_as_skip(home, monkeypatch, pipeline):
    _budget(home)
    _spend(monkeypatch, day=12, month=40)
    monkeypatch.setattr(jobs, "mark_job_skipped", lambda *a, **k: False)

    s.run_one_job({"id": "abc", "name": "explore"})

    record = executions.latest_execution("abc")
    assert record["status"] == "failed"
    assert "ownership lost" in record["error"]


def test_tick_calls_watch_with_adapters(home, monkeypatch):
    seen = []
    monkeypatch.setattr(cost_budget, "watch", lambda **kw: seen.append(kw))
    monkeypatch.setattr(s, "get_due_jobs", lambda: [])
    adapters, loop = {"slack": object()}, object()

    s.tick(verbose=False, adapters=adapters, loop=loop)

    assert seen == [{"adapters": adapters, "loop": loop}]


def test_tick_survives_watch_error(home, monkeypatch):
    def boom(**_kw):
        raise RuntimeError("nope")

    monkeypatch.setattr(cost_budget, "watch", boom)
    monkeypatch.setattr(s, "get_due_jobs", lambda: [])

    assert s.tick(verbose=False) == 0


def test_two_profiles_each_use_own_budget_store_and_bot(tmp_path, monkeypatch, pipeline):
    """Multiplex: each profile's tick reads its own spend and alerts through its own adapters."""
    a, b = tmp_path / "default", tmp_path / "educator"
    for h in (a, b):
        h.mkdir()
    (a / "config.yaml").write_text("cost_budget:\n  daily_usd: 100\n  alert_channel: slack:CA00000001\n",
                                   encoding="utf-8")
    _budget(b)
    monkeypatch.setenv("HERMES_HOME", str(a))
    _spend(monkeypatch, by_home={str(a.resolve()): (5.0, 5.0), str(b.resolve()): (12.0, 12.0)})
    monkeypatch.setattr(s, "get_due_jobs", lambda: [])
    adapters_a, adapters_b = {"slack": "bot-A"}, {"slack": "bot-B"}

    for home, adapters in ((a, adapters_a), (b, adapters_b)):
        token = set_hermes_home_override(str(home))
        try:
            s.tick(verbose=False, adapters=adapters)
            s.run_one_job({"id": "abc", "name": "explore"}, adapters=adapters)
        finally:
            reset_hermes_home_override(token)

    alerts = _alerts(pipeline)
    assert [(al["adapters"], al["job"]["deliver"]) for al in alerts] == [(adapters_b, "slack:C1")]
    assert pipeline["run_job"] == ["abc"]  # profile A is under budget; B skipped it
    assert (b / "cost_budget" / "alerts.json").exists()
    assert not (a / "cost_budget" / "alerts.json").exists()


# ── mark_job_skipped: a skip is not a run ─────────────────────────────────

def test_skip_keeps_streak_and_repeat_of_recurring_job(home):
    job = jobs.create_job(prompt="explore", schedule="every 1h", name="explore", repeat=5)
    jobs.update_job(job["id"], {"failure_streak": 2})

    assert jobs.mark_job_skipped(job["id"], "skipped_budget", "skipped: over budget")

    after = jobs.get_job(job["id"])
    assert after["last_status"] == "skipped_budget"
    assert after["last_skip_reason"] == "skipped: over budget"
    assert after["failure_streak"] == 2
    assert after["repeat"]["completed"] == 0
    assert after["next_run_at"]
    assert after["enabled"] is not False


def test_skip_hands_back_one_shot_dispatch_and_rearms_at_reset(home):
    job = jobs.create_job(prompt="draft", schedule="2026-12-01T09:00:00", name="once")
    assert job["schedule"]["kind"] == "once"
    assert jobs.claim_dispatch(job["id"])
    retry = datetime(2026, 11, 18, 0, 0, tzinfo=timezone.utc)

    assert jobs.mark_job_skipped(job["id"], "skipped_budget", "r", retry_at=retry)

    after = jobs.get_job(job["id"])
    assert after["repeat"]["completed"] == 0
    assert after["next_run_at"] == retry.isoformat()
    assert after["enabled"] is not False and after.get("state") != "completed"


def test_skip_refused_for_other_fire_owner(home):
    job = jobs.create_job(prompt="explore", schedule="every 1h")
    jobs.update_job(job["id"], {"fire_claim": {"by": "other", "at": "x"}})

    assert not jobs.mark_job_skipped(job["id"], "skipped_budget", "r", expected_fire_owner="me")
    assert jobs.get_job(job["id"]).get("last_status") != "skipped_budget"


def test_cronjob_tool_discretionary_is_set_only(home, monkeypatch):
    import json

    from tools import cronjob_tools

    created = json.loads(cronjob_tools.cronjob(action="create", prompt="explore the KB",
                                               schedule="every 4h", name="explore",
                                               discretionary=True))
    assert created["success"], created
    job_id = created["job_id"]
    assert jobs.get_job(job_id)["discretionary"] is True

    refused = json.loads(cronjob_tools.cronjob(action="update", job_id=job_id, discretionary=False))
    assert refused.get("success") is False and "operator" in refused["error"]
    assert jobs.get_job(job_id)["discretionary"] is True
