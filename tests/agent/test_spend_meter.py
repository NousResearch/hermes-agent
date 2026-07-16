"""Tests for agent.spend_meter — pricing, watermark accrual, thresholds, throttle."""

from __future__ import annotations

import json
import sqlite3
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo
from datetime import datetime

import pytest

from agent import spend_meter
from agent.spend_meter import (
    SpendConfig,
    accrue,
    compute_and_write_throttle,
    cumulative_cost,
    day_window,
    evaluate_thresholds,
    is_profile_paused,
    read_throttle,
    set_override,
)

TZ = "America/Montevideo"
# 2026-07-15 12:00:00 local (UTC-3)
NOON = datetime(2026, 7, 15, 12, 0, 0, tzinfo=ZoneInfo(TZ)).timestamp()


def make_cfg(**kwargs) -> SpendConfig:
    cfg = SpendConfig(
        timezone=TZ,
        lane_overrides={"workerA": "api_key", "workerB": "personal_oauth"},
        throttle_enabled=True,
    )
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return cfg


def make_db(path: Path, rows: list[dict]) -> Path:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE sessions (id TEXT PRIMARY KEY, model TEXT, billing_provider TEXT,"
        " input_tokens INT, output_tokens INT, cache_read_tokens INT,"
        " cache_write_tokens INT, started_at REAL, ended_at REAL)"
    )
    for row in rows:
        conn.execute(
            "INSERT OR REPLACE INTO sessions VALUES"
            " (:id, :model, :billing_provider, :input_tokens, :output_tokens,"
            "  :cache_read_tokens, :cache_write_tokens, :started_at, :ended_at)",
            {
                "billing_provider": None,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "ended_at": None,
                **row,
            },
        )
    conn.commit()
    conn.close()
    return path


def update_tokens(path: Path, session_id: str, **cols) -> None:
    conn = sqlite3.connect(path)
    sets = ", ".join(f"{col} = :{col}" for col in cols)
    conn.execute(f"UPDATE sessions SET {sets} WHERE id = :id", {"id": session_id, **cols})
    conn.commit()
    conn.close()


@pytest.fixture(autouse=True)
def clear_caches():
    spend_meter._lane_cache.clear()
    spend_meter._throttle_cache.clear()
    yield
    spend_meter._lane_cache.clear()
    spend_meter._throttle_cache.clear()


# ─── Pricing ─────────────────────────────────────────────────────────────────


def test_sonnet5_exact_cost():
    cost, status = cumulative_cost(
        {
            "model": "claude-sonnet-5",
            "input_tokens": 1_000_000,
            "output_tokens": 100_000,
            "cache_read_tokens": 500_000,
            "cache_write_tokens": 200_000,
        }
    )
    # 2.00 + 1.00 + 0.10 + 0.50
    assert cost == Decimal("3.60")
    assert status == "estimated"


def test_fable5_priced():
    cost, status = cumulative_cost(
        {"model": "claude-fable-5", "input_tokens": 100_000, "output_tokens": 10_000}
    )
    # 1.00 + 0.50
    assert cost == Decimal("1.50")
    assert status == "estimated"


def test_reasoning_tokens_not_double_billed():
    base = {"model": "claude-sonnet-5", "input_tokens": 1000, "output_tokens": 5000}
    with_reasoning = dict(base, reasoning_tokens=4000)
    assert cumulative_cost(base) == cumulative_cost(with_reasoning)


def test_unpriced_model_reports_gap(monkeypatch):
    monkeypatch.setattr(spend_meter, "_models_dev_fallback_entry", lambda model: None)
    cost, status = cumulative_cost(
        {"model": "claude-imaginary-9", "input_tokens": 1000, "output_tokens": 1000}
    )
    assert cost == Decimal("0")
    assert status == "pricing_gap"


# ─── Accrual / watermarks ────────────────────────────────────────────────────


def test_accrue_bootstrap_and_delta(tmp_path):
    cfg = make_cfg()
    window_start, _, _ = day_window(NOON, TZ)
    db = make_db(
        tmp_path / "a.db",
        [
            {
                "id": "s1",
                "model": "claude-sonnet-5",
                "input_tokens": 1_000_000,
                "output_tokens": 0,
                "started_at": window_start + 60,
            }
        ],
    )
    dbs = [("workerA", db)]

    ledger = accrue(None, NOON, cfg, dbs=dbs)
    assert ledger["lanes"]["api_key"]["usd"] == pytest.approx(2.0)
    assert ledger["profiles"]["workerA"]["usd"] == pytest.approx(2.0)

    # Second poll, no growth → no double counting.
    ledger = accrue(ledger, NOON + 300, cfg, dbs=dbs)
    assert ledger["lanes"]["api_key"]["usd"] == pytest.approx(2.0)

    # Growth → only the delta accrues.
    update_tokens(db, "s1", input_tokens=1_500_000)
    ledger = accrue(ledger, NOON + 600, cfg, dbs=dbs)
    assert ledger["lanes"]["api_key"]["usd"] == pytest.approx(3.0)


def test_accrue_old_session_starts_at_watermark(tmp_path):
    cfg = make_cfg()
    window_start, _, _ = day_window(NOON, TZ)
    db = make_db(
        tmp_path / "a.db",
        [
            {
                "id": "old",
                "model": "claude-sonnet-5",
                "input_tokens": 10_000_000,
                "output_tokens": 0,
                "started_at": window_start - 3600,  # yesterday
            }
        ],
    )
    dbs = [("workerA", db)]
    ledger = accrue(None, NOON, cfg, dbs=dbs)
    # Pre-existing cumulative total must NOT accrue on bootstrap...
    assert ledger.get("lanes") == {}
    # ...but growth after bootstrap does.
    update_tokens(db, "old", input_tokens=11_000_000)
    ledger = accrue(ledger, NOON + 300, cfg, dbs=dbs)
    assert ledger["lanes"]["api_key"]["usd"] == pytest.approx(2.0)


def test_day_rollover_archives_history(tmp_path):
    cfg = make_cfg()
    window_start, window_end, date_str = day_window(NOON, TZ)
    db = make_db(
        tmp_path / "a.db",
        [
            {
                "id": "s1",
                "model": "claude-sonnet-5",
                "input_tokens": 1_000_000,
                "output_tokens": 0,
                "started_at": window_start + 60,
            }
        ],
    )
    dbs = [("workerA", db)]
    ledger = accrue(None, NOON, cfg, dbs=dbs)

    tomorrow_noon = window_end + 12 * 3600
    ledger = accrue(ledger, tomorrow_noon, cfg, dbs=dbs)
    assert ledger["date"] != date_str
    assert ledger["history"][date_str]["lanes"]["api_key"]["usd"] == pytest.approx(2.0)
    assert ledger.get("lanes") == {}  # fresh day
    # Watermarks survive the rollover: growth still accrues as a delta.
    update_tokens(db, "s1", input_tokens=2_000_000)
    ledger = accrue(ledger, tomorrow_noon + 300, cfg, dbs=dbs)
    assert ledger["lanes"]["api_key"]["usd"] == pytest.approx(2.0)


# ─── Thresholds ──────────────────────────────────────────────────────────────


def test_thresholds_fire_once(tmp_path):
    cfg = make_cfg()
    cfg.lanes["api_key"]["daily_cap_usd"] = 4.0
    ledger = spend_meter.empty_ledger("2026-07-15", TZ)
    ledger["lanes"] = {"api_key": {"usd": 2.2}}  # 55%

    alerts = evaluate_thresholds(ledger, cfg)
    assert [a.threshold for a in alerts] == [0.5]
    assert evaluate_thresholds(ledger, cfg) == []  # deduped

    ledger["lanes"]["api_key"]["usd"] = 4.5  # >100%
    alerts = evaluate_thresholds(ledger, cfg)
    assert [a.threshold for a in alerts] == [0.8, 1.0]


def test_profile_cap_threshold():
    cfg = make_cfg(profile_caps={"workerA": 1.0})
    ledger = spend_meter.empty_ledger("2026-07-15", TZ)
    ledger["profiles"] = {"workerA": {"usd": 1.5, "lane": "api_key"}}
    alerts = evaluate_thresholds(ledger, cfg)
    assert [a.target for a in alerts] == ["profile:workerA"] * 3


# ─── Throttle ────────────────────────────────────────────────────────────────


def test_throttle_pause_and_override(tmp_path):
    cfg = make_cfg()
    cfg.lanes["api_key"]["daily_cap_usd"] = 1.0
    ledger = spend_meter.empty_ledger("2026-07-15", TZ)
    ledger["lanes"] = {"api_key": {"usd": 1.5}}
    path = tmp_path / "throttle.json"

    compute_and_write_throttle(ledger, cfg, path=path, now=NOON)
    state = read_throttle(path)
    assert "api_key" in state.paused_lanes
    assert is_profile_paused("workerA", state, cfg, now=NOON)
    assert is_profile_paused("workerB", state, cfg, now=NOON) is None  # other lane

    # Exempt profile never paused even in the paused lane.
    cfg.exempt_profiles = ["workerA"]
    assert is_profile_paused("workerA", state, cfg, now=NOON) is None
    cfg.exempt_profiles = []

    # Resume override lifts the pause and survives recompute while active.
    set_override("api_key", "resume", NOON + 3600, path=path)
    state = read_throttle(path)
    assert is_profile_paused("workerA", state, cfg, now=NOON) is None
    state = compute_and_write_throttle(ledger, cfg, path=path, now=NOON + 60)
    assert "api_key" not in state.paused_lanes
    # Expired override → pause returns.
    state = compute_and_write_throttle(ledger, cfg, path=path, now=NOON + 7200)
    assert "api_key" in state.paused_lanes


def test_throttle_disabled_writes_nothing_paused(tmp_path):
    cfg = make_cfg(throttle_enabled=False)
    cfg.lanes["api_key"]["daily_cap_usd"] = 1.0
    ledger = spend_meter.empty_ledger("2026-07-15", TZ)
    ledger["lanes"] = {"api_key": {"usd": 99.0}}
    path = tmp_path / "throttle.json"
    state = compute_and_write_throttle(ledger, cfg, path=path, now=NOON)
    assert state.paused_lanes == {}
    assert json.loads(path.read_text())["paused"] == {}


def test_manual_pause_override(tmp_path):
    cfg = make_cfg()
    ledger = spend_meter.empty_ledger("2026-07-15", TZ)
    path = tmp_path / "throttle.json"
    set_override("workerA", "pause", None, path=path)
    compute_and_write_throttle(ledger, cfg, path=path, now=NOON)
    state = read_throttle(path)
    assert is_profile_paused("workerA", state, cfg, now=NOON) == "profile workerA manually paused"


# ─── Billing swap (api_key cap → personal_oauth failover) ────────────────────


def swap_cfg(**kwargs):
    cfg = make_cfg(**kwargs)
    cfg.lanes["api_key"]["swap_to"] = "personal_oauth"
    return cfg


def test_over_cap_swaps_instead_of_pausing(tmp_path):
    cfg = swap_cfg()
    cfg.lanes["api_key"]["daily_cap_usd"] = 1.0
    ledger = spend_meter.empty_ledger("2026-07-16", TZ)
    ledger["lanes"] = {"api_key": {"usd": 1.5}}
    path = tmp_path / "throttle.json"
    state = compute_and_write_throttle(ledger, cfg, path=path, now=NOON)
    assert state.paused_lanes == {}
    assert state.swapped_lanes["api_key"]["to"] == "personal_oauth"
    # Swapped lane's profiles keep dispatching.
    assert is_profile_paused("workerA", read_throttle(path), cfg, now=NOON) is None


def test_swap_falls_back_to_pause_when_target_exhausted(tmp_path):
    cfg = swap_cfg()
    cfg.lanes["api_key"]["daily_cap_usd"] = 1.0
    cfg.lanes["personal_oauth"]["daily_cap_usd"] = 2.0
    ledger = spend_meter.empty_ledger("2026-07-16", TZ)
    ledger["lanes"] = {"api_key": {"usd": 1.5}, "personal_oauth": {"usd": 2.5}}
    path = tmp_path / "throttle.json"
    state = compute_and_write_throttle(ledger, cfg, path=path, now=NOON)
    assert state.swapped_lanes == {}
    assert "api_key" in state.paused_lanes
    assert "personal_oauth" in state.paused_lanes


def test_swapped_profiles_pause_when_target_lane_pauses(tmp_path):
    """Escalation ladder: api_key swapped, then personal_oauth hits its cap
    → api_key profiles (billing through personal) pause too."""
    cfg = swap_cfg()
    path = tmp_path / "throttle.json"
    spend_meter._save_throttle_raw(
        {
            "swapped": {"api_key": {"to": "personal_oauth", "usd": 200.0, "cap": 200.0}},
            "paused": {"personal_oauth": {"usd": 25.0, "cap": 25.0}},
            "paused_profiles": {},
            "overrides": {},
        },
        path,
    )
    state = read_throttle(path)
    reason = is_profile_paused("workerA", state, cfg, now=NOON)
    assert reason is not None and "swapped to personal_oauth" in reason
    assert is_profile_paused("workerB", state, cfg, now=NOON) is not None  # native lane paused


def test_swap_accrual_attributes_to_target_lane(tmp_path):
    cfg = swap_cfg()
    window_start, _, _ = day_window(NOON, TZ)
    db = make_db(
        tmp_path / "a.db",
        [
            {
                "id": "s1",
                "model": "claude-sonnet-5",
                "input_tokens": 1_000_000,
                "output_tokens": 0,
                "started_at": window_start + 60,
            }
        ],
    )
    dbs = [("workerA", db)]
    throttle_path = tmp_path / "throttle.json"
    spend_meter._save_throttle_raw(
        {
            "swapped": {"api_key": {"to": "personal_oauth"}},
            "routing": {"mode": "personal_only", "personal_share": 1.0},
            "paused": {},
            "overrides": {},
        },
        throttle_path,
    )
    ledger = accrue(None, NOON, cfg, dbs=dbs, throttle_path=throttle_path)
    assert "api_key" not in ledger["lanes"]
    assert ledger["lanes"]["personal_oauth"]["usd"] == pytest.approx(2.0)
    # Profile keeps its .env lane for display; the USD went to the target lane.
    assert ledger["profiles"]["workerA"]["lane"] == "api_key"


def test_resume_override_clears_swap(tmp_path):
    cfg = swap_cfg()
    cfg.lanes["api_key"]["daily_cap_usd"] = 1.0
    ledger = spend_meter.empty_ledger("2026-07-16", TZ)
    ledger["lanes"] = {"api_key": {"usd": 1.5}}
    path = tmp_path / "throttle.json"
    compute_and_write_throttle(ledger, cfg, path=path, now=NOON)
    assert "api_key" in read_throttle(path).swapped_lanes
    set_override("api_key", "resume", NOON + 3600, path=path)
    assert "api_key" not in read_throttle(path).swapped_lanes
    state = compute_and_write_throttle(ledger, cfg, path=path, now=NOON + 60)
    assert state.swapped_lanes == {}  # override honored on recompute


# ─── Account-first routing ladder ────────────────────────────────────────────


def test_routing_default_split(tmp_path):
    cfg = swap_cfg()
    ledger = spend_meter.empty_ledger("2026-07-16", TZ)
    path = tmp_path / "throttle.json"
    state = compute_and_write_throttle(ledger, cfg, path=path, now=NOON,
                                       account={"max_pct": 12.0})
    assert state.routing["mode"] == "split"
    assert state.routing["personal_share"] == pytest.approx(0.8)


def test_routing_account_hot_switches_to_api_key_only_with_hysteresis(tmp_path):
    cfg = swap_cfg()
    ledger = spend_meter.empty_ledger("2026-07-16", TZ)
    path = tmp_path / "throttle.json"
    state = compute_and_write_throttle(ledger, cfg, path=path, now=NOON,
                                       account={"max_pct": 85.0})
    assert state.routing["mode"] == "api_key_only"
    # Hysteresis: 70% is below the 80% entry but above the 60% resume → stay.
    state = compute_and_write_throttle(ledger, cfg, path=path, now=NOON + 300,
                                       account={"max_pct": 70.0})
    assert state.routing["mode"] == "api_key_only"
    # Below resume threshold → back to split.
    state = compute_and_write_throttle(ledger, cfg, path=path, now=NOON + 600,
                                       account={"max_pct": 45.0})
    assert state.routing["mode"] == "split"


def test_routing_api_key_capped_goes_personal_only(tmp_path):
    cfg = swap_cfg()
    cfg.lanes["api_key"]["daily_cap_usd"] = 1.0
    ledger = spend_meter.empty_ledger("2026-07-16", TZ)
    ledger["lanes"] = {"api_key": {"usd": 1.5}}
    path = tmp_path / "throttle.json"
    state = compute_and_write_throttle(ledger, cfg, path=path, now=NOON,
                                       account={"max_pct": 30.0})
    assert state.routing["mode"] == "personal_only"
    assert state.swapped_lanes["api_key"]["to"] == "personal_oauth"
    assert state.paused_lanes == {}


def test_routing_both_exhausted_pauses_dispatch(tmp_path):
    cfg = swap_cfg()
    cfg.lanes["api_key"]["daily_cap_usd"] = 1.0
    ledger = spend_meter.empty_ledger("2026-07-16", TZ)
    ledger["lanes"] = {"api_key": {"usd": 1.5}}
    path = tmp_path / "throttle.json"
    state = compute_and_write_throttle(ledger, cfg, path=path, now=NOON,
                                       account={"max_pct": 90.0})
    assert state.routing["mode"] == "api_key_only"
    assert "api_key" in state.paused_lanes
    assert is_profile_paused("workerA", read_throttle(path), cfg, now=NOON) is not None


def test_split_accrual_attributes_fractionally(tmp_path):
    cfg = swap_cfg()
    window_start, _, _ = day_window(NOON, TZ)
    db = make_db(
        tmp_path / "a.db",
        [
            {
                "id": "s1",
                "model": "claude-sonnet-5",
                "input_tokens": 1_000_000,
                "output_tokens": 0,
                "started_at": window_start + 60,
            }
        ],
    )
    dbs = [("workerA", db)]
    throttle_path = tmp_path / "throttle.json"
    spend_meter._save_throttle_raw(
        {"routing": {"mode": "split", "personal_share": 0.8}}, throttle_path
    )
    ledger = accrue(None, NOON, cfg, dbs=dbs, throttle_path=throttle_path)
    assert ledger["lanes"]["personal_oauth"]["usd"] == pytest.approx(1.6)
    assert ledger["lanes"]["api_key"]["usd"] == pytest.approx(0.4)
    assert ledger["profiles"]["workerA"]["usd"] == pytest.approx(2.0)


# ─── Credits-aware failover ──────────────────────────────────────────────────


def _acct(five=0.0, weekly=0.0, credits=None):
    return {
        "five_hour_pct": five,
        "weekly_pct": weekly,
        "max_pct": max(five, weekly),
        "credits": credits,
        "windows": {"five_hour": {"pct": five}, "seven_day": {"pct": weekly}},
    }


def test_five_hour_burst_with_credit_headroom_stays_split(tmp_path):
    """5h at 100% spills into credits — no failover while headroom exists."""
    cfg = swap_cfg()
    ledger = spend_meter.empty_ledger("2026-07-16", TZ)
    path = tmp_path / "throttle.json"
    account = _acct(five=100.0, weekly=7.0,
                    credits={"used_usd": 50.0, "limit_usd": 100.0,
                             "remaining_usd": 50.0, "pct": 50.0})
    state = compute_and_write_throttle(ledger, cfg, path=path, now=NOON, account=account)
    assert state.routing["mode"] == "split"


def test_five_hour_burst_without_credits_fails_over(tmp_path):
    cfg = swap_cfg()
    ledger = spend_meter.empty_ledger("2026-07-16", TZ)
    path = tmp_path / "throttle.json"
    state = compute_and_write_throttle(
        ledger, cfg, path=path, now=NOON, account=_acct(five=100.0, weekly=7.0, credits=None)
    )
    assert state.routing["mode"] == "api_key_only"
    assert "5h" in state.routing["reason"]


def test_credits_near_exhaustion_fails_over(tmp_path):
    cfg = swap_cfg()
    ledger = spend_meter.empty_ledger("2026-07-16", TZ)
    path = tmp_path / "throttle.json"
    account = _acct(five=10.0, weekly=7.0,
                    credits={"used_usd": 97.0, "limit_usd": 100.0,
                             "remaining_usd": 3.0, "pct": 97.0})
    state = compute_and_write_throttle(ledger, cfg, path=path, now=NOON, account=account)
    assert state.routing["mode"] == "api_key_only"
    assert "credits" in state.routing["reason"]


def test_weekly_window_always_gates_even_with_credits(tmp_path):
    cfg = swap_cfg()
    ledger = spend_meter.empty_ledger("2026-07-16", TZ)
    path = tmp_path / "throttle.json"
    account = _acct(five=10.0, weekly=85.0,
                    credits={"used_usd": 10.0, "limit_usd": 100.0,
                             "remaining_usd": 90.0, "pct": 10.0})
    state = compute_and_write_throttle(ledger, cfg, path=path, now=NOON, account=account)
    assert state.routing["mode"] == "api_key_only"
    assert "weekly" in state.routing["reason"]
