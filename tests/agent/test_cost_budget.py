"""Cost budget + alerts (fork-only): config, spend snapshot, status, alerts."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent import cost_budget
from agent.cost_budget import (
    BudgetConfig,
    budget_status,
    compute_spend,
    get_spend,
    is_discretionary,
    load_budget_config,
)
from hermes_state import SessionDB

NOW = datetime(2026, 11, 16, 15, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _clear_caches():
    cost_budget._reset_for_tests()
    yield
    cost_budget._reset_for_tests()


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


def _seed(home: Path, session_id: str, *, started_at: datetime, cost=None, status="estimated",
          parent=None, model="gpt-5.4-mini", input_tokens=1000, output_tokens=100):
    db = SessionDB(db_path=home / "state.db")
    try:
        db.create_session(session_id=session_id, source="slack", model=model, parent_session_id=parent)
        db.update_token_counts(
            session_id, input_tokens=input_tokens, output_tokens=output_tokens, model=model,
            billing_provider="openai", estimated_cost_usd=cost, cost_status=status,
            cost_source="user_override" if cost is not None else "none", api_call_count=1,
        )
        db.append_message(session_id, role="user", content="hi")
        db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?",
                         (started_at.timestamp(), session_id))
        db._conn.commit()
    finally:
        db.close()


def _cfg(**kw) -> BudgetConfig:
    raw = {"daily_usd": 10, "monthly_usd": 100, "alert_channel": "slack:C1"}
    raw.update(kw)
    cfg = load_budget_config({"cost_budget": raw})
    assert cfg is not None
    return cfg


# ── config ────────────────────────────────────────────────────────────────

class TestConfig:
    def test_absent_block_is_none(self):
        assert load_budget_config({}) is None
        assert load_budget_config({"cost_budget": None}) is None
        assert load_budget_config({"cost_budget": "yes"}) is None

    def test_block_without_any_limit_is_none(self):
        assert load_budget_config({"cost_budget": {"alert_channel": "slack:C1"}}) is None
        assert load_budget_config({"cost_budget": {"daily_usd": 0}}) is None

    def test_parses_full_block(self):
        cfg = load_budget_config({"cost_budget": {
            "daily_usd": 5, "monthly_usd": "100", "alert_at": [1.0, 0.5, 0.8],
            "per_run_usd": 1.5, "alert_channel": "C0123ABCD", "alert_mention": "<!here>",
            "discretionary_jobs": ["explore", "abc123"], "cache_seconds": 120,
        }})
        assert cfg == BudgetConfig(
            daily_usd=5.0, monthly_usd=100.0, alert_at=(0.5, 0.8, 1.0), per_run_usd=1.5,
            alert_channel="slack:C0123ABCD", alert_mention="<!here>",
            discretionary_jobs=frozenset({"explore", "abc123"}), cache_seconds=120,
        )

    def test_defaults(self):
        cfg = load_budget_config({"cost_budget": {"monthly_usd": 50}})
        assert cfg.daily_usd is None
        assert cfg.alert_at == (0.5, 0.8, 1.0)
        assert cfg.alert_channel is None
        assert cfg.discretionary_jobs == frozenset()
        assert cfg.cache_seconds == 300

    def test_bad_values_dropped(self):
        cfg = load_budget_config({"cost_budget": {
            "daily_usd": "x", "monthly_usd": 20, "alert_at": [2, -1, 0.5, "q"],
            "discretionary_jobs": "explore", "cache_seconds": 1,
        }})
        assert cfg.daily_usd is None
        assert cfg.alert_at == (0.5, 2.0)
        assert cfg.discretionary_jobs == frozenset({"explore"})
        assert cfg.cache_seconds == 30

    def test_reads_profile_config_by_default(self, home):
        (home / "config.yaml").write_text("cost_budget:\n  daily_usd: 3\n", encoding="utf-8")
        assert load_budget_config().daily_usd == 3.0


# ── spend ─────────────────────────────────────────────────────────────────

class TestSpend:
    def test_day_and_month(self, home):
        _seed(home, "s_today", started_at=NOW - timedelta(hours=2), cost=2.0)
        _seed(home, "s_month", started_at=NOW - timedelta(days=5), cost=3.0)
        _seed(home, "s_last_month", started_at=NOW - timedelta(days=40), cost=10.0)
        snap = compute_spend(home, NOW)
        assert snap.day_key == "2026-11-16"
        assert snap.month_key == "2026-11"
        assert snap.day_usd == pytest.approx(2.0)
        assert snap.month_usd == pytest.approx(5.0)
        assert snap.day_unpriced_tokens == 0

    def test_unpriced_tokens_counted_not_zeroed(self, home):
        _seed(home, "s_unpriced", started_at=NOW - timedelta(hours=1), cost=None,
              status="unknown", model="mystery-model")
        snap = compute_spend(home, NOW)
        assert snap.day_usd == 0
        assert snap.day_unpriced_tokens == 1100
        assert snap.month_unpriced_tokens == 1100
        assert snap.unpriced_models == ("mystery-model",)

    def test_cron_runs_grouped_by_root(self, home):
        root = "cron_job1_20261116_080000"
        _seed(home, root, started_at=NOW - timedelta(hours=7), cost=1.0)
        _seed(home, "child", started_at=NOW - timedelta(hours=6), cost=0.75, parent=root)
        _seed(home, "slack_turn", started_at=NOW - timedelta(hours=6), cost=5.0)
        snap = compute_spend(home, NOW)
        assert len(snap.runs) == 1
        run = snap.runs[0]
        assert (run.root_id, run.job_id) == (root, "job1")
        assert run.cost_usd == pytest.approx(1.75)

    def test_missing_store_zero_snapshot(self, home):
        snap = compute_spend(home, NOW)
        assert (snap.day_usd, snap.month_usd, snap.runs) == (0.0, 0.0, ())
        assert not (home / "state.db").exists()

    def test_get_spend_caches_per_home(self, tmp_path, monkeypatch):
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir(), b.mkdir()
        _seed(a, "sa", started_at=NOW - timedelta(hours=1), cost=1.0)
        _seed(b, "sb", started_at=NOW - timedelta(hours=1), cost=4.0)
        cfg = _cfg()
        monkeypatch.setenv("HERMES_HOME", str(a))
        assert get_spend(cfg, now=NOW).day_usd == pytest.approx(1.0)
        monkeypatch.setenv("HERMES_HOME", str(b))
        assert get_spend(cfg, now=NOW).day_usd == pytest.approx(4.0)

        calls = []
        monkeypatch.setattr(cost_budget, "compute_spend",
                            lambda *a, **k: calls.append(1) or pytest.fail("cache miss"))
        assert get_spend(cfg, now=NOW).day_usd == pytest.approx(4.0)
        assert calls == []

    def test_get_spend_refreshes_on_new_day(self, home, monkeypatch):
        _seed(home, "s1", started_at=NOW - timedelta(hours=1), cost=1.0)
        cfg = _cfg()
        assert get_spend(cfg, now=NOW).day_key == "2026-11-16"
        assert get_spend(cfg, now=NOW + timedelta(days=1)).day_key == "2026-11-17"


# ── status ────────────────────────────────────────────────────────────────

def _snap(day=0.0, month=0.0, **kw):
    base = dict(day_key="2026-11-16", month_key="2026-11", day_usd=day, month_usd=month,
                day_unpriced_tokens=0, month_unpriced_tokens=0, unpriced_models=(),
                runs=(), as_of=NOW.timestamp())
    base.update(kw)
    return cost_budget.SpendSnapshot(**base)


class TestStatus:
    def test_under_budget(self):
        st = budget_status(_cfg(), _snap(day=4, month=40))
        assert st.day_fraction == pytest.approx(0.4)
        assert st.month_fraction == pytest.approx(0.4)
        assert not st.over_budget

    def test_over_daily(self):
        st = budget_status(_cfg(), _snap(day=10, month=40))
        assert st.over_daily and not st.over_monthly and st.over_budget

    def test_over_monthly(self):
        st = budget_status(_cfg(), _snap(day=1, month=120))
        assert st.over_monthly and st.over_budget

    def test_no_limits_never_over(self):
        st = budget_status(_cfg(daily_usd=None, monthly_usd=None, per_run_usd=1), _snap(day=99, month=999))
        assert st.day_fraction is None and not st.over_budget


class TestDiscretionary:
    def test_by_field_id_and_name(self):
        cfg = _cfg(discretionary_jobs=["explore", "abc123"])
        assert is_discretionary({"id": "zzz", "name": "x", "discretionary": True}, cfg)
        assert is_discretionary({"id": "abc123", "name": "x"}, cfg)
        assert is_discretionary({"id": "zzz", "name": "explore"}, cfg)
        assert not is_discretionary({"id": "zzz", "name": "release-watch"}, cfg)
        assert not is_discretionary({"id": "zzz", "name": "x", "discretionary": "yes"}, cfg)


# ── alerts ────────────────────────────────────────────────────────────────

from agent.cost_budget import Alert, pending_alerts, send_once, watch  # noqa: E402


@pytest.fixture()
def sent(monkeypatch):
    """Record deliveries made through the cron delivery path."""
    import cron.scheduler as scheduler

    calls = []

    def fake_deliver(job, content, adapters=None, loop=None):
        calls.append({"job": job, "content": content, "adapters": adapters, "loop": loop})
        return None

    monkeypatch.setattr(scheduler, "_deliver_result", fake_deliver)
    return calls


def _alerts(cfg, snap):
    return pending_alerts(budget_status(cfg, snap))


class TestPendingAlerts:
    def test_under_first_threshold_no_alert(self):
        assert _alerts(_cfg(), _snap(day=4, month=40)) == []

    def test_single_alert_for_multiple_crossings(self):
        alerts = _alerts(_cfg(), _snap(day=11, month=40))
        assert [a.key for a in alerts] == ["daily:2026-11-16:1"]
        assert alerts[0].loud
        assert set(alerts[0].covers) == {"daily:2026-11-16:0.5", "daily:2026-11-16:0.8"}
        assert "$11.00" in alerts[0].text and "$10.00" in alerts[0].text
        assert "Discretionary jobs are skipped" in alerts[0].text

    def test_monthly_threshold_not_loud_below_one(self):
        alerts = _alerts(_cfg(), _snap(day=1, month=85))
        assert [(a.key, a.loud) for a in alerts] == [("monthly:2026-11:0.8", False)]

    def test_run_overspend(self):
        run = cost_budget.RunSpend(root_id="cron_j1_20261116_080000", job_id="j1",
                                   started_at=NOW.timestamp(), cost_usd=2.5, unpriced_tokens=0)
        cheap = cost_budget.RunSpend(root_id="cron_j1_20261116_090000", job_id="j1",
                                     started_at=NOW.timestamp(), cost_usd=0.5, unpriced_tokens=0)
        alerts = _alerts(_cfg(per_run_usd=2), _snap(day=3, month=3, runs=(run, cheap)))
        assert [a.key for a in alerts] == ["run:cron_j1_20261116_080000"]
        assert "$2.50" in alerts[0].text and "j1" in alerts[0].text

    def test_unpriced_alert(self):
        alerts = _alerts(_cfg(), _snap(day=1, month=1, day_unpriced_tokens=5000,
                                       month_unpriced_tokens=5000, unpriced_models=("m1",)))
        assert [a.key for a in alerts] == ["unpriced:2026-11-16"]
        assert "pricing.overrides" in alerts[0].text and "m1" in alerts[0].text

    def test_threshold_alert_marks_spend_as_lower_bound(self):
        alerts = _alerts(_cfg(), _snap(day=6, month=6, day_unpriced_tokens=10,
                                       month_unpriced_tokens=10))
        daily = next(a for a in alerts if a.key.startswith("daily:"))
        assert "at least" in daily.text


class TestSendOnce:
    def test_delivers_through_cron_path_with_callers_adapters(self, home, sent):
        adapters, loop = {"slack": object()}, object()
        n = send_once([Alert(key="k1", text="hello")], _cfg(), adapters=adapters, loop=loop)
        assert n == 1
        assert sent[0]["job"]["deliver"] == "slack:C1"
        assert sent[0]["adapters"] is adapters and sent[0]["loop"] is loop
        assert sent[0]["content"] == "hello"

    def test_not_resent(self, home, sent):
        send_once([Alert(key="k1", text="a")], _cfg())
        assert send_once([Alert(key="k1", text="a")], _cfg()) == 0
        assert len(sent) == 1

    def test_covers_marked(self, home, sent):
        send_once([Alert(key="d:1", text="a", covers=("d:0.5",))], _cfg())
        assert send_once([Alert(key="d:0.5", text="b")], _cfg()) == 0

    def test_new_period_alerts_again(self, home, sent):
        cfg = _cfg()
        send_once(_alerts(cfg, _snap(day=11)), cfg)
        send_once(_alerts(cfg, _snap(day=11, day_key="2026-11-17")), cfg)
        assert len(sent) == 2

    def test_delivery_failure_not_recorded(self, home, monkeypatch):
        import cron.scheduler as scheduler

        results = iter(["boom", None])
        monkeypatch.setattr(scheduler, "_deliver_result", lambda *a, **k: next(results))
        assert send_once([Alert(key="k1", text="a")], _cfg()) == 0
        assert send_once([Alert(key="k1", text="a")], _cfg()) == 1

    def test_delivery_exception_not_recorded(self, home, monkeypatch):
        import cron.scheduler as scheduler

        def boom(*a, **k):
            raise RuntimeError("adapter gone")

        monkeypatch.setattr(scheduler, "_deliver_result", boom)
        assert send_once([Alert(key="k1", text="a")], _cfg()) == 0

    def test_loud_alert_gets_mention(self, home, sent):
        cfg = _cfg(alert_mention="<!subteam^S1>")
        send_once([Alert(key="q", text="quiet"), Alert(key="l", text="loud", loud=True)], cfg)
        assert [c["content"] for c in sent] == ["quiet", "<!subteam^S1> loud"]

    def test_no_alert_channel_logs_only(self, home, sent, caplog):
        cfg = _cfg(alert_channel=None)
        with caplog.at_level("WARNING", logger="agent.cost_budget"):
            assert send_once([Alert(key="k1", text="over!")], cfg) == 0
        assert sent == []
        assert "over!" in caplog.text
        assert not (home / "cost_budget").exists()

    def test_state_is_per_home(self, tmp_path, monkeypatch, sent):
        for name in ("a", "b"):
            h = tmp_path / name
            h.mkdir()
            monkeypatch.setenv("HERMES_HOME", str(h))
            assert send_once([Alert(key="k1", text="a")], _cfg()) == 1
        assert len(sent) == 2


class TestWatch:
    def test_without_block_touches_nothing(self, home, monkeypatch, sent):
        monkeypatch.setattr(cost_budget, "compute_spend", lambda *a, **k: pytest.fail("queried"))
        watch(adapters={}, loop=None)
        assert sent == []
        assert not (home / "cost_budget").exists()

    def test_sends_threshold_alert(self, home, sent):
        (home / "config.yaml").write_text(
            "cost_budget:\n  daily_usd: 1\n  alert_channel: slack:C1\n", encoding="utf-8")
        _seed(home, "s1", started_at=datetime.now(timezone.utc) - timedelta(minutes=1), cost=2.0)
        adapters = {"slack": object()}
        watch(adapters=adapters, loop=None)
        assert len(sent) == 1 and sent[0]["adapters"] is adapters
        assert "Daily spend" in sent[0]["content"]

    def test_throttled(self, home, monkeypatch, sent):
        (home / "config.yaml").write_text("cost_budget:\n  daily_usd: 1\n", encoding="utf-8")
        calls = []
        real = cost_budget.compute_spend
        monkeypatch.setattr(cost_budget, "compute_spend", lambda *a, **k: calls.append(1) or real(*a, **k))
        watch()
        cost_budget._spend_cache.clear()
        watch()
        assert len(calls) == 1

    def test_never_raises(self, home, monkeypatch):
        (home / "config.yaml").write_text("cost_budget:\n  daily_usd: 1\n", encoding="utf-8")

        def boom(*a, **k):
            raise RuntimeError("store exploded")

        monkeypatch.setattr(cost_budget, "compute_spend", boom)
        watch()
