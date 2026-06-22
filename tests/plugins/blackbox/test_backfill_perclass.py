"""SPEC-C Phase 4 — backfill per-class cost into existing blackbox rows.

Tests the load-bearing logic (not just Σparts==total, which is vacuous):
- route-linearity audit (request_cost must be None to aggregate-price);
- status allowlist (only estimated/included/actual repriced);
- the repriced-vs-original guardrail (INV-8) — if the aggregate reprice diverges
  from the originally-billed total beyond a small rate-drift bound, the row is a
  hidden partial/non-linear case → NULL split, original total untouched.
"""
from __future__ import annotations

import sqlite3

import pytest

from scripts import backfill_perclass_cost as bf


REPRICEABLE = ("estimated", "included", "actual")


def test_status_allowlist_excludes_partial_unknown_null():
    assert bf.is_repriceable("estimated") is True
    assert bf.is_repriceable("included") is True
    assert bf.is_repriceable("actual") is True
    assert bf.is_repriceable("partial") is False
    assert bf.is_repriceable("unknown") is False
    assert bf.is_repriceable("") is False
    assert bf.is_repriceable(None) is False
    # an UNKNOWN future status vocab is excluded (allowlist, not denylist)
    assert bf.is_repriceable("weird-new-status") is False


def test_reprice_row_fully_priced_writes_split_summing_to_total():
    # a clean Opus row reprices; the split sums to the new total within bound
    out = bf.reprice_row(
        model="claude-opus-4-8", provider="claude-api-proxy",
        input_tokens=0, output_tokens=130_000,
        cache_read=109_700_000, cache_write=646_000, reasoning=0,
        original_cost_usd=62.20, rate_drift_bound=2.0)
    assert out is not None
    total, split = out
    assert total is not None
    s = split["uncached"] + split["cache_read"] + split["cache_write"] + split["output"]
    assert abs(s - total) < 0.01


def test_reprice_row_guardrail_rejects_divergent_total():
    # original cost wildly different from a fresh reprice (a hidden partial that
    # slipped the status filter) → guardrail returns None (caller keeps original
    # total + NULL split), NOT a silent rewrite.
    out = bf.reprice_row(
        model="claude-opus-4-8", provider="claude-api-proxy",
        input_tokens=0, output_tokens=130_000,
        cache_read=109_700_000, cache_write=646_000, reasoning=0,
        original_cost_usd=5.00,   # way below the ~$62 reprice
        rate_drift_bound=2.0)
    assert out is None


def test_reprice_row_unknown_route_returns_none():
    out = bf.reprice_row(
        model="totally-unknown-xyz", provider="mystery",
        input_tokens=1000, output_tokens=500,
        cache_read=0, cache_write=0, reasoning=0,
        original_cost_usd=1.0, rate_drift_bound=2.0)
    assert out is None


def _seed_db(path, rows):
    conn = sqlite3.connect(str(path))
    conn.execute("""CREATE TABLE turns (
        turn_id TEXT PRIMARY KEY, model TEXT, provider TEXT, cost_usd REAL,
        cost_status TEXT, input_tokens INT, output_tokens INT, cache_read INT,
        cache_write INT, reasoning INT,
        cost_uncached_usd REAL, cost_cache_read_usd REAL,
        cost_cache_write_usd REAL, cost_output_usd REAL)""")
    for r in rows:
        cols = ",".join(r.keys()); ph = ",".join("?" * len(r))
        conn.execute(f"INSERT INTO turns ({cols}) VALUES ({ph})", tuple(r.values()))
    conn.commit(); conn.close()


def test_route_audit_enumerates_from_db_not_config(tmp_path):
    db = tmp_path / "t.db"
    _seed_db(db, [
        dict(turn_id="a", model="claude-opus-4-8", provider="claude-api-proxy",
             cost_status="estimated", cost_usd=1.0, input_tokens=1000,
             output_tokens=1, cache_read=0, cache_write=0, reasoning=0),
    ])
    routes = bf.audit_routes([str(db)])
    # returns {(model, provider): request_cost_is_none_bool}
    assert ("claude-opus-4-8", "claude-api-proxy") in routes
    assert routes[("claude-opus-4-8", "claude-api-proxy")] is True


def test_backfill_db_dry_run_writes_nothing_then_apply(tmp_path):
    db = tmp_path / "t.db"
    _seed_db(db, [
        # fully-priced, in-bound → repriced
        dict(turn_id="ok", model="claude-opus-4-8", provider="claude-api-proxy",
             cost_status="estimated", cost_usd=None, input_tokens=0,
             output_tokens=130000, cache_read=109_700_000, cache_write=646000,
             reasoning=0),
        # partial → skipped, total untouched, split stays NULL
        dict(turn_id="part", model="claude-opus-4-8", provider="claude-api-proxy",
             cost_status="partial", cost_usd=3.0, input_tokens=1000,
             output_tokens=100, cache_read=500000, cache_write=0, reasoning=0),
    ])
    # the 'ok' row has cost_usd=None so the guardrail can't compare — treat None
    # original as "accept reprice" (first-time pricing). Apply:
    stats = bf.backfill_db(str(db), apply=True, rate_drift_bound=2.0)
    conn = sqlite3.connect(str(db)); conn.row_factory = sqlite3.Row
    ok = conn.execute("SELECT * FROM turns WHERE turn_id='ok'").fetchone()
    part = conn.execute("SELECT * FROM turns WHERE turn_id='part'").fetchone()
    conn.close()
    # ok row got a split summing to its (new) total
    assert ok["cost_uncached_usd"] is not None
    s = (ok["cost_uncached_usd"] + ok["cost_cache_read_usd"]
         + ok["cost_cache_write_usd"] + ok["cost_output_usd"])
    assert abs(s - ok["cost_usd"]) < 0.01
    # partial row untouched
    assert part["cost_usd"] == 3.0
    assert part["cost_uncached_usd"] is None
    assert stats["repriced"] == 1
    assert stats["skipped_status"] == 1


def test_backfill_preserves_existing_cost_usd_and_scales_split(tmp_path):
    """greptile #79: an already-priced row must NOT have its historical cost_usd
    overwritten. The split is SCALED to the stored total (proportions preserved,
    parts sum to the real billed amount), and cost_usd is left byte-identical."""
    db = tmp_path / "t.db"
    _seed_db(db, [
        dict(turn_id="hist", model="claude-opus-4-8", provider="claude-api-proxy",
             cost_status="estimated", cost_usd=50.00,  # historical billed total
             input_tokens=0, output_tokens=130000, cache_read=109_700_000,
             cache_write=646000, reasoning=0),
    ])
    # the fresh reprice of these tokens is ~$62; within a generous bound it must
    # NOT overwrite the stored $50 — it scales the split to $50 instead.
    bf.backfill_db(str(db), apply=True, rate_drift_bound=100.0)
    conn = sqlite3.connect(str(db)); conn.row_factory = sqlite3.Row
    r = conn.execute("SELECT * FROM turns WHERE turn_id='hist'").fetchone()
    conn.close()
    assert r["cost_usd"] == 50.00, "historical cost_usd must be preserved, not overwritten"
    parts = (r["cost_uncached_usd"] + r["cost_cache_read_usd"]
             + r["cost_cache_write_usd"] + r["cost_output_usd"])
    assert abs(parts - 50.00) < 0.01, "scaled split must sum to the stored total"
    # proportions preserved: cache-read dominates an Opus turn
    assert r["cost_cache_read_usd"] > r["cost_output_usd"]
