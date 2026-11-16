"""Cost budget enforcement in the cron scheduler (fork-only).

The scheduler itself skips a discretionary job while its profile is over
budget — before the job's script or agent runs — and records the skip in the
executions ledger. Non-discretionary work keeps running with a loud alert.
"""
from datetime import datetime, timezone

import pytest

import cron.scheduler as s
from agent import cost_budget
from cron import executions

NOW = datetime(2026, 11, 16, 15, 0, tzinfo=timezone.utc)


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
        "  discretionary_jobs: [explore]\n" + extra,
        encoding="utf-8",
    )


def _spend(monkeypatch, *, day=0.0, month=0.0):
    snap = cost_budget.SpendSnapshot(
        day_key="2026-11-16", month_key="2026-11", day_usd=day, month_usd=month,
        day_unpriced_tokens=0, month_unpriced_tokens=0, unpriced_models=(), runs=(),
        as_of=NOW.timestamp())
    monkeypatch.setattr(cost_budget, "compute_spend", lambda *_a, **_k: snap)


@pytest.fixture()
def pipeline(monkeypatch):
    calls = {"run_job": [], "deliver": [], "mark": []}

    def fake_run_job(job, *, defer_agent_teardown=None, **kw):
        calls["run_job"].append(job["id"])
        return True, "out", "final response", None

    def fake_deliver(job, content, adapters=None, loop=None):
        calls["deliver"].append({"job": job, "content": content, "adapters": adapters})
        return None

    def fake_mark(jid, ok, err=None, delivery_error=None, **kw):
        calls["mark"].append((jid, ok, kw.get("status")))
        return True

    monkeypatch.setattr(s, "run_job", fake_run_job)
    monkeypatch.setattr(s, "save_job_output", lambda jid, out: f"/tmp/{jid}.txt")
    monkeypatch.setattr(s, "_deliver_result", fake_deliver)
    monkeypatch.setattr(s, "mark_job_run", fake_mark)
    return calls


def test_discretionary_job_skipped_when_over_budget(home, monkeypatch, pipeline):
    _budget(home)
    _spend(monkeypatch, day=12, month=40)

    assert s.run_one_job({"id": "j-exp", "name": "x", "discretionary": True}) is True

    assert pipeline["run_job"] == []
    assert pipeline["deliver"] == []
    assert pipeline["mark"] == [("j-exp", True, "skipped_budget")]
    record = executions.latest_execution("j-exp")
    assert record["status"] == "completed"
    assert record["error"].startswith("skipped: over budget")
    assert "daily $12.00 of $10.00" in record["error"]


def test_named_discretionary_job_skipped_on_monthly(home, monkeypatch, pipeline):
    _budget(home)
    _spend(monkeypatch, day=1, month=150)

    s.run_one_job({"id": "abc", "name": "explore"})

    assert pipeline["run_job"] == []
    assert "monthly $150.00 of $100.00" in executions.latest_execution("abc")["error"]


def test_non_discretionary_job_runs_and_alerts_loudly_once(home, monkeypatch, pipeline):
    _budget(home, "  alert_mention: '<!subteam^S1>'\n")
    _spend(monkeypatch, day=12, month=40)
    adapters = {"slack": object()}

    s.run_one_job({"id": "rel", "name": "release-watch", "deliver": "local"}, adapters=adapters)
    s.run_one_job({"id": "rel", "name": "release-watch", "deliver": "local"}, adapters=adapters)

    assert pipeline["run_job"] == ["rel", "rel"]
    alerts = [d for d in pipeline["deliver"] if d["job"]["id"] == "cost_budget"]
    assert len(alerts) == 1
    assert alerts[0]["job"]["deliver"] == "slack:C1"
    assert alerts[0]["adapters"] is adapters
    assert alerts[0]["content"].startswith("<!subteam^S1> 🚨")
    assert "release-watch" in alerts[0]["content"]
    assert executions.latest_execution("rel")["error"] is None


def test_under_budget_runs(home, monkeypatch, pipeline):
    _budget(home)
    _spend(monkeypatch, day=2, month=20)

    s.run_one_job({"id": "e", "name": "explore"})

    assert pipeline["run_job"] == ["e"]
    assert pipeline["mark"] == [("e", True, None)]


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

    s.run_one_job({"id": "e", "name": "explore"})

    assert pipeline["run_job"] == ["e"]


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
