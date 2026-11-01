"""Tests for backfill_unknown_session_costs: re-pricing sessions the cost writer
left at ``cost_status='unknown'``/NULL (transient /models pricing-fetch failures)."""

import sqlite3
from types import SimpleNamespace

import pytest

from hermes_state_usage import backfill_unknown_session_costs


_SESSION_COLS = (
    "id TEXT PRIMARY KEY, model TEXT, billing_provider TEXT, billing_base_url TEXT, "
    "billing_mode TEXT, input_tokens INTEGER, output_tokens INTEGER, cache_read_tokens INTEGER, "
    "cache_write_tokens INTEGER, reasoning_tokens INTEGER, estimated_cost_usd REAL, "
    "actual_cost_usd REAL, cost_status TEXT, cost_source TEXT, pricing_version TEXT, ended_at TEXT"
)


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "state.db"
    conn = sqlite3.connect(path)
    conn.execute(f"CREATE TABLE sessions ({_SESSION_COLS})")
    conn.commit()
    conn.close()
    return str(path)


def _insert(db_path, sid, **kw):
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO sessions (id) VALUES (?)", (sid,))
    kw = dict(kw)
    kw.setdefault("ended_at", "2026-01-01T00:00:00")  # ended by default; pass None for active
    cols = ", ".join(f"{k}=?" for k in kw)
    conn.execute(f"UPDATE sessions SET {cols} WHERE id=?", (*kw.values(), sid))
    conn.commit()
    conn.close()


def _fake_estimate(monkeypatch, amount=0.1234, status="estimated"):
    def _fake(model, usage, provider=None, base_url=None, api_key=""):
        return SimpleNamespace(
            amount_usd=amount, status=status, source="provider_models_api",
            pricing_version="openai-compatible-models-api",
        )
    monkeypatch.setattr("agent.usage_pricing.estimate_usage_cost", _fake)


def _rows(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    out = [dict(r) for r in conn.execute("SELECT * FROM sessions ORDER BY id")]
    conn.close()
    return out


def test_backfill_prices_unknown_and_skips_priced(db_path, monkeypatch):
    _fake_estimate(monkeypatch)
    _insert(db_path, "cron", model="z-ai/glm-5.2", billing_provider="nous",
            input_tokens=70142, output_tokens=4069, cache_read_tokens=51520,
            cache_write_tokens=0, reasoning_tokens=3101,
            cost_status="unknown", cost_source="none", estimated_cost_usd=0.0)
    _insert(db_path, "priced", model="z-ai/glm-5.2", billing_provider="nous",
            input_tokens=100, output_tokens=10, cost_status="estimated",
            cost_source="provider_models_api", estimated_cost_usd=0.0005)
    _insert(db_path, "subagent", input_tokens=0, output_tokens=0, cache_read_tokens=0,
            cache_write_tokens=0, reasoning_tokens=0, cost_status=None)

    report = backfill_unknown_session_costs(db_path)

    assert report["found"] == 2          # 'cron' + 'subagent'; 'priced' excluded
    assert report["fixed"] == 2
    assert report["skipped"] == 0
    rows = {r["id"]: r for r in _rows(db_path)}

    # cron re-priced via estimate_usage_cost
    assert rows["cron"]["cost_status"] == "estimated"
    assert rows["cron"]["estimated_cost_usd"] == 0.1234
    assert rows["cron"]["cost_source"] == "provider_models_api"
    assert rows["cron"]["actual_cost_usd"] is None  # estimate, not actual
    # zero-token session finalized as included $0 (a fact, not an estimate)
    assert rows["subagent"]["cost_status"] == "included"
    assert rows["subagent"]["estimated_cost_usd"] == 0.0
    # already-priced row untouched
    assert rows["priced"]["estimated_cost_usd"] == 0.0005


def test_backfill_dry_run_writes_nothing(db_path, monkeypatch):
    _fake_estimate(monkeypatch)
    _insert(db_path, "cron", model="z-ai/glm-5.2", billing_provider="nous",
            input_tokens=10, output_tokens=1, cost_status="unknown",
            cost_source="none", estimated_cost_usd=0.0)

    report = backfill_unknown_session_costs(db_path, dry_run=True)
    assert report["fixed"] == 1
    rows = _rows(db_path)
    assert rows[0]["cost_status"] == "unknown"  # unchanged
    assert rows[0]["estimated_cost_usd"] == 0.0


def test_backfill_skips_no_model_nonzero_tokens(db_path, monkeypatch):
    _fake_estimate(monkeypatch)
    _insert(db_path, "orphan", model=None, billing_provider=None,
            input_tokens=100, output_tokens=10, cost_status="unknown")

    report = backfill_unknown_session_costs(db_path)
    assert report["found"] == 1
    assert report["fixed"] == 0
    assert report["skipped"] == 1
    assert report["details"][0]["result"] == "no-model"


def test_backfill_skips_still_unpriced(db_path, monkeypatch):
    # estimate_usage_cost returns no amount -> route still has no pricing -> skip, not $0.
    def _fake(model, usage, provider=None, base_url=None, api_key=""):
        return SimpleNamespace(amount_usd=None, status="unknown", source="none", pricing_version=None)
    monkeypatch.setattr("agent.usage_pricing.estimate_usage_cost", _fake)
    _insert(db_path, "stuck", model="some/model", billing_provider="weird",
            input_tokens=50, output_tokens=5, cost_status="unknown")

    report = backfill_unknown_session_costs(db_path)
    assert report["found"] == 1
    assert report["fixed"] == 0
    assert report["skipped"] == 1
    assert report["details"][0]["result"] == "still-unpriced"


def test_backfill_ignores_active_sessions(db_path, monkeypatch):
    # An in-flight session legitimately has cost_status NULL; ended_at IS NULL means
    # it must be left untouched for the normal finalization path.
    _fake_estimate(monkeypatch)
    _insert(db_path, "active", model="z-ai/glm-5.2", billing_provider="nous",
            input_tokens=500, output_tokens=50, cost_status=None, ended_at=None)

    report = backfill_unknown_session_costs(db_path)
    assert report["found"] == 0
    assert report["fixed"] == 0
    rows = _rows(db_path)
    assert rows[0]["cost_status"] is None  # untouched
    assert rows[0]["estimated_cost_usd"] is None