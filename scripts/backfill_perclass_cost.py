#!/usr/bin/env python3
"""SPEC-C Phase 4 — backfill per-class cost into existing blackbox turn rows.

Existing rows predate the per-class columns, and the per-CALL billing tokens were
never persisted (comp_calls_json carries only output/reasoning/composition). So
this prices from each row's AGGREGATE token columns in ONE estimate_usage_cost
call. That reproduces the live per-call total because pricing is LINEAR on our
routes (request_cost=None) — SPEC-C INV-7.

Safety (SPEC-C D-4 / INV-8), in order:
  1. Route audit: enumerate DISTINCT (model, provider) from the DBs (NOT config)
     and require request_cost is None — a per-request fee can't be reconstructed
     from aggregates, so such a route's turns are left NULL-split.
  2. Status allowlist: only reprice estimated/included/actual rows. partial/
     unknown/NULL/unknown-vocab → left untouched (NULL split).
  3. Repriced-vs-original guardrail: if the fresh aggregate reprice diverges from
     the originally-billed cost_usd beyond a small rate-drift bound, the row is a
     hidden partial/non-linear case → leave its total, NULL split, log it. (When
     the original cost_usd is NULL — never priced — accept the reprice.)

Backs up each DB before any write; --dry-run shows the plan; idempotent within a
pricing epoch.

Usage:
  PYTHONPATH=<repo-root> python scripts/backfill_perclass_cost.py --dry-run DB [DB ...]
  PYTHONPATH=<repo-root> python scripts/backfill_perclass_cost.py --apply  DB [DB ...]

(PYTHONPATH pins ``agent.usage_pricing`` to THIS checkout's engine — the one with
the per-class breakdown — rather than a system-installed hermes-agent.)
"""
from __future__ import annotations

import argparse
import glob
import os
import sqlite3
import sys
import time
from decimal import Decimal

from agent.usage_pricing import (
    CanonicalUsage,
    estimate_usage_cost,
    get_pricing_entry,
)

_REPRICEABLE = frozenset({"estimated", "included", "actual"})
DEFAULT_KEEP_BACKUPS = 2


def _safe_backup(path: str, *, keep: int = DEFAULT_KEEP_BACKUPS) -> str:
    """Checkpoint-consistent backup of ``path`` → ``<path>.bak-perclass-<ts>``,
    then prune older ``.bak-perclass-*`` for this DB to the newest ``keep``.

    Consistency (SPEC-D INV-2): uses ``VACUUM INTO`` (folds the WAL into one
    defragmented file, no ``-wal``/``-shm`` sidecars), falling back to the
    SQLite online-backup API for very old SQLite. NEVER a raw ``copy2`` of a
    live WAL-mode DB (which can tear / strand sidecars).

    Bounded (INV-3): after writing the new backup, delete older backups beyond
    ``keep`` (newest-first by mtime), plus any stranded ``-wal``/``-shm``
    sidecar from a legacy ``copy2`` backup (D-5). The just-written backup is
    NEVER a prune candidate (D-4). Write-then-prune ordering (D-3) guarantees
    ≥1 valid backup at every instant. Prune failures are swallowed (INV-4) —
    a cleanup error must never abort the reprice or drop the current backup.
    """
    backup = f"{path}.bak-perclass-{int(time.time())}"
    # avoid clobbering a same-second prior backup
    if os.path.exists(backup):
        backup = f"{path}.bak-perclass-{int(time.time())}-{os.getpid()}"
    src = sqlite3.connect(path)
    try:
        try:
            # quote the destination (VACUUM INTO won't take a bound param)
            src.execute("VACUUM INTO ?", (backup,))
        except sqlite3.OperationalError:
            # SQLite < 3.27: online-backup API (also checkpoint-consistent)
            dst = sqlite3.connect(backup)
            try:
                src.backup(dst)
            finally:
                dst.close()
    finally:
        src.close()

    # prune older backups to `keep`, never the one just written.
    try:
        candidates = [
            p for p in glob.glob(f"{path}.bak-perclass-*")
            if not p.endswith(("-wal", "-shm")) and p != backup
        ]
        candidates.sort(key=os.path.getmtime, reverse=True)
        # keep the newest (keep-1) priors alongside the just-written backup
        for stale in candidates[max(0, keep - 1):]:
            for victim in (stale, stale + "-wal", stale + "-shm"):
                try:
                    os.unlink(victim)
                except OSError:
                    pass
    except Exception:
        pass  # INV-4: prune is best-effort; never abort the reprice
    return backup

_PERCLASS_COLS = (
    "cost_uncached_usd", "cost_cache_read_usd",
    "cost_cache_write_usd", "cost_output_usd",
)
DEFAULT_RATE_DRIFT_BOUND = 5.0  # $ — generous; catches structural divergence, not cents


def is_repriceable(status) -> bool:
    """Allowlist (not denylist): only known clean statuses reprice."""
    return status in _REPRICEABLE


def route_request_cost_is_none(model: str, provider: str) -> bool:
    """True if the route has no per-request fee (safe to aggregate-price)."""
    try:
        entry = get_pricing_entry(model, provider=provider or None)
    except Exception:
        return False
    if entry is None:
        # unpriceable route — not aggregate-safe (it'll fail to reprice anyway)
        return False
    return entry.request_cost is None


def audit_routes(db_paths: list[str]) -> dict[tuple[str, str], bool]:
    """Enumerate DISTINCT (model, provider) from the DBs and map each to whether
    request_cost is None (aggregate-priceable). From the DATA, not config."""
    out: dict[tuple[str, str], bool] = {}
    for path in db_paths:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "SELECT DISTINCT model, COALESCE(provider,'') FROM turns"
            ).fetchall()
        finally:
            conn.close()
        for model, provider in rows:
            key = (model or "", provider or "")
            if key not in out:
                out[key] = route_request_cost_is_none(model or "", provider or "")
    return out


def reprice_row(
    *,
    model: str,
    provider: str,
    input_tokens: int,
    output_tokens: int,
    cache_read: int,
    cache_write: int,
    reasoning: int,
    original_cost_usd,
    rate_drift_bound: float = DEFAULT_RATE_DRIFT_BOUND,
):
    """Reprice one row from its aggregate tokens.

    Returns ``(total_float, perclass_dict)`` on success, or ``None`` when the row
    must be left untouched (unknown route, or the reprice diverges from the
    original billed total beyond ``rate_drift_bound`` — INV-8 guardrail).
    """
    usage = CanonicalUsage(
        input_tokens=int(input_tokens or 0),
        output_tokens=int(output_tokens or 0),
        cache_read_tokens=int(cache_read or 0),
        cache_write_tokens=int(cache_write or 0),
        reasoning_tokens=int(reasoning or 0),
    )
    try:
        r = estimate_usage_cost(model, usage, provider=provider or None)
    except Exception:
        return None
    if r.amount_usd is None or r.status == "unknown":
        return None
    if r.cost_input_usd is None:  # not a priced breakdown (shouldn't happen here)
        return None
    total = float(r.amount_usd)
    # INV-8 guardrail: compare to the originally-billed total (when present).
    if original_cost_usd is not None:
        if abs(total - float(original_cost_usd)) > rate_drift_bound:
            return None
    split = {
        "uncached": float(r.cost_input_usd),
        "cache_read": float(r.cost_cache_read_usd or Decimal("0")),
        "cache_write": float(r.cost_cache_write_usd or Decimal("0")),
        "output": float(r.cost_output_usd or Decimal("0")),
    }
    return total, split


def backfill_db(path: str, *, apply: bool, rate_drift_bound: float = DEFAULT_RATE_DRIFT_BOUND,
                routes: dict | None = None, keep_backups: int = DEFAULT_KEEP_BACKUPS) -> dict:
    """Backfill one DB. Returns a stats dict. Backs up before writing when apply."""
    if routes is None:
        routes = audit_routes([path])
    stats = {
        "total_rows": 0, "repriced": 0, "skipped_status": 0,
        "skipped_route": 0, "skipped_guardrail": 0,
    }
    if apply:
        backup = _safe_backup(path, keep=keep_backups)
        stats["backup"] = backup

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT turn_id, model, COALESCE(provider,'') AS provider, cost_usd,
                      cost_status, input_tokens, output_tokens, cache_read,
                      cache_write, reasoning
               FROM turns
               WHERE (COALESCE(input_tokens,0)+COALESCE(output_tokens,0)
                      +COALESCE(cache_read,0)+COALESCE(cache_write,0)) > 0"""
        ).fetchall()
        for row in rows:
            stats["total_rows"] += 1
            if not is_repriceable(row["cost_status"]):
                stats["skipped_status"] += 1
                continue
            key = (row["model"] or "", row["provider"] or "")
            if not routes.get(key, False):
                stats["skipped_route"] += 1
                continue
            out = reprice_row(
                model=row["model"] or "", provider=row["provider"] or "",
                input_tokens=row["input_tokens"], output_tokens=row["output_tokens"],
                cache_read=row["cache_read"], cache_write=row["cache_write"],
                reasoning=row["reasoning"], original_cost_usd=row["cost_usd"],
                rate_drift_bound=rate_drift_bound)
            if out is None:
                stats["skipped_guardrail"] += 1
                continue
            total, split = out
            stats["repriced"] += 1
            if apply:
                # NEVER overwrite an existing historical cost_usd (greptile #79):
                # the stored total was billed at record-time rates; the split was
                # priced at current rates. For an already-priced row, keep the
                # billed total and SCALE the split to it (single factor preserves
                # each class's proportion exactly AND keeps parts summing to the
                # real billed amount — INV-2). Only WRITE cost_usd when the row
                # was never priced (original is None → first-time pricing).
                orig = row["cost_usd"]
                if orig is None:
                    new_cost = total
                    s = split
                else:
                    new_cost = orig
                    factor = (float(orig) / total) if total else 0.0
                    s = {k: (v * factor if v is not None else None)
                         for k, v in split.items()}
                conn.execute(
                    """UPDATE turns SET cost_usd=?, cost_uncached_usd=?,
                       cost_cache_read_usd=?, cost_cache_write_usd=?,
                       cost_output_usd=? WHERE turn_id=?""",
                    (new_cost, s["uncached"], s["cache_read"],
                     s["cache_write"], s["output"], row["turn_id"]))
        if apply:
            conn.commit()
    finally:
        conn.close()
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description="Backfill per-class cost (SPEC-C Phase 4).")
    ap.add_argument("dbs", nargs="+", help="turns.db path(s)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="show the plan, write nothing")
    g.add_argument("--apply", action="store_true", help="back up + write")
    ap.add_argument("--rate-drift-bound", type=float, default=DEFAULT_RATE_DRIFT_BOUND)
    ap.add_argument("--keep-backups", type=int, default=DEFAULT_KEEP_BACKUPS,
                    help=f"per-DB .bak-perclass-* files to retain (default {DEFAULT_KEEP_BACKUPS})")
    a = ap.parse_args(argv)

    routes = audit_routes(a.dbs)
    bad = sorted(k for k, ok in routes.items() if not ok)
    print(f"route audit: {len(routes)} distinct (model,provider); "
          f"{len(bad)} non-aggregate-safe (NULL-split): {bad}", flush=True)

    grand = {}
    for path in a.dbs:
        st = backfill_db(path, apply=a.apply, rate_drift_bound=a.rate_drift_bound,
                         routes=routes, keep_backups=a.keep_backups)
        print(f"{path}: {st}", flush=True)
        for k, v in st.items():
            if isinstance(v, int):
                grand[k] = grand.get(k, 0) + v
    print(f"TOTAL: {grand}  ({'APPLIED' if a.apply else 'DRY-RUN'})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
