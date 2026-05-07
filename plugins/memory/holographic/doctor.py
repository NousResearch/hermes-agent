"""Memory store health checks.

Surfaces invariants that downstream code relies on but cannot enforce
locally:

- ``schema_version`` matches the code's ``_CURRENT_SCHEMA_VERSION``
- every fact with an ``hrr_vector`` was encoded under the current
  ``_CURRENT_ENCODING_VERSION`` (drift here means probe results can be
  stale until the row is re-encoded)
- ``hrr_vector`` byte-length matches ``hrr_dim * 8`` (float64 phases)
- no orphan rows in ``fact_entities`` (a fact_id or entity_id pointing
  at a deleted row)
- a smoke probe completes in well under the 5s budget against the live
  corpus and recovers signal above the noise floor

The intent is a single command, fast, definitive: green means callers
can trust probe/related/reason without running diagnostics. The doctor
does not mutate state — it reports. Repairs are explicit and called out.
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from typing import Any


_NOISE_FLOOR = 0.10  # raw_sim above this is "signal" given dim=2048 crosstalk math
_BUDGET_MS = 5000


def _check(name: str, status: str, detail: str = "", **extra: Any) -> dict:
    out = {"name": name, "status": status, "detail": detail}
    out.update(extra)
    return out


def check_memory_health(
    db_path: "str | Path | None" = None,
    hrr_dim: int = 2048,
    *,
    smoke_entity: "str | None" = None,
) -> dict:
    """Run all memory-store health checks. Read-only.

    Returns ``{"status": "ok"|"warn"|"error", "checks": [...],
    "elapsed_ms": int, "db_path": str}``. Each check is a dict with
    ``name``, ``status``, ``detail`` and optional context fields.

    ``status`` rolls up: any ``error`` → top-level error; any ``warn``
    without errors → warn; otherwise ok. The doctor never raises — a
    truly broken DB still returns a structured report.
    """
    t0 = time.monotonic()
    checks: list[dict] = []

    if db_path is None:
        try:
            from hermes_constants import get_hermes_home
            db_path = str(get_hermes_home() / "memory_store.db")
        except Exception as exc:
            checks.append(_check(
                "db_path", "error",
                f"could not resolve default db_path: {exc}",
            ))
            return _finalize(checks, t0, db_path="<unresolved>")

    db_path_str = str(Path(db_path).expanduser())
    if not os.path.exists(db_path_str):
        checks.append(_check(
            "db_exists", "error",
            f"memory_store.db not found at {db_path_str}",
        ))
        return _finalize(checks, t0, db_path=db_path_str)

    try:
        conn = sqlite3.connect(db_path_str, timeout=2.0)
        conn.row_factory = sqlite3.Row
    except sqlite3.OperationalError as exc:
        checks.append(_check("db_open", "error", f"could not open DB: {exc}"))
        return _finalize(checks, t0, db_path=db_path_str)

    try:
        from .store import _CURRENT_ENCODING_VERSION, _CURRENT_SCHEMA_VERSION
        checks.append(_check_schema_version(conn, _CURRENT_SCHEMA_VERSION))
        checks.append(_check_encoding_version(conn, _CURRENT_ENCODING_VERSION))
        checks.append(_check_vector_shape(conn, hrr_dim))
        checks.append(_check_orphans(conn))
        checks.append(_check_smoke_probe(conn, hrr_dim, smoke_entity))
    finally:
        conn.close()

    return _finalize(checks, t0, db_path=db_path_str)


def _finalize(checks: list[dict], t0: float, *, db_path: str) -> dict:
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    if any(c["status"] == "error" for c in checks):
        rolled = "error"
    elif any(c["status"] == "warn" for c in checks):
        rolled = "warn"
    else:
        rolled = "ok"
    return {
        "status":     rolled,
        "checks":     checks,
        "elapsed_ms": elapsed_ms,
        "db_path":    db_path,
    }


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_schema_version(conn: sqlite3.Connection, expected: int) -> dict:
    try:
        row = conn.execute(
            "SELECT version FROM schema_version LIMIT 1"
        ).fetchone()
    except sqlite3.OperationalError as exc:
        return _check("schema_version", "error",
                      f"schema_version table missing: {exc}")
    if row is None:
        return _check("schema_version", "error",
                      "schema_version table exists but is empty")
    current = row[0]
    if current != expected:
        return _check("schema_version", "warn",
                      f"DB at v{current}, code expects v{expected} — "
                      f"open the DB once via MemoryStore() to upgrade",
                      current=current, expected=expected)
    return _check("schema_version", "ok", f"v{current}")


def _check_encoding_version(conn: sqlite3.Connection, expected: int) -> dict:
    """Count facts with hrr_vector encoded under an older algebra version."""
    try:
        stale = conn.execute(
            "SELECT COUNT(*) FROM facts "
            "WHERE hrr_vector IS NOT NULL AND encoding_version < ?",
            (expected,),
        ).fetchone()[0]
    except sqlite3.OperationalError as exc:
        return _check("encoding_version", "error",
                      f"encoding_version column missing: {exc}")
    total = conn.execute(
        "SELECT COUNT(*) FROM facts WHERE hrr_vector IS NOT NULL"
    ).fetchone()[0]
    if stale > 0:
        return _check(
            "encoding_version", "warn",
            f"{stale}/{total} facts encoded under an older version "
            f"(current=v{expected}). Run MemoryStore.rebuild_all_vectors() "
            f"to repair.",
            stale=stale, total=total, expected=expected,
        )
    return _check(
        "encoding_version", "ok",
        f"all {total} encoded facts at v{expected}",
        total=total,
    )


def _check_vector_shape(conn: sqlite3.Connection, hrr_dim: int) -> dict:
    """Every hrr_vector should be exactly hrr_dim * 8 bytes (float64 phases)."""
    expected_bytes = hrr_dim * 8
    rows = conn.execute(
        "SELECT fact_id, length(hrr_vector) AS n FROM facts "
        "WHERE hrr_vector IS NOT NULL"
    ).fetchall()
    if not rows:
        return _check("vector_shape", "ok",
                      "no encoded facts (skipped)", total=0)
    bad = [(r["fact_id"], r["n"]) for r in rows if r["n"] != expected_bytes]
    if bad:
        sample = ", ".join(f"fact_id={fid}: {n} bytes" for fid, n in bad[:3])
        return _check(
            "vector_shape", "error",
            f"{len(bad)}/{len(rows)} vectors at wrong byte-length "
            f"(expected {expected_bytes}). Sample: {sample}",
            bad_count=len(bad), total=len(rows), expected_bytes=expected_bytes,
        )
    return _check("vector_shape", "ok",
                  f"all {len(rows)} vectors at {expected_bytes} bytes",
                  total=len(rows))


def _check_orphans(conn: sqlite3.Connection) -> dict:
    """fact_entities rows whose fact_id or entity_id no longer exists."""
    orphan_facts = conn.execute(
        "SELECT COUNT(*) FROM fact_entities fe "
        "WHERE NOT EXISTS (SELECT 1 FROM facts f WHERE f.fact_id = fe.fact_id)"
    ).fetchone()[0]
    orphan_entities = conn.execute(
        "SELECT COUNT(*) FROM fact_entities fe "
        "WHERE NOT EXISTS (SELECT 1 FROM entities e WHERE e.entity_id = fe.entity_id)"
    ).fetchone()[0]
    total = orphan_facts + orphan_entities
    if total == 0:
        return _check("orphans", "ok", "fact_entities is clean")
    return _check(
        "orphans", "warn",
        f"{orphan_facts} dangling fact refs, {orphan_entities} dangling "
        f"entity refs in fact_entities",
        orphan_fact_refs=orphan_facts, orphan_entity_refs=orphan_entities,
    )


def _check_smoke_probe(
    conn: sqlite3.Connection,
    hrr_dim: int,
    smoke_entity: "str | None",
) -> dict:
    """Pick a high-link-count entity, run a Strategy-A probe, assert signal.

    The doctor only fails this check on hard breakage (numpy missing,
    no facts at all, exception). Low signal on a real DB is reported as
    a warn, not an error — corpus content varies.
    """
    try:
        from . import holographic as hrr
    except Exception as exc:
        return _check("smoke_probe", "error", f"holographic import failed: {exc}")

    if not getattr(hrr, "_HAS_NUMPY", False):
        return _check("smoke_probe", "warn",
                      "numpy unavailable — probe path is FTS-only")

    # Pick the entity to probe by, if not explicitly supplied.
    entity_name: "str | None" = smoke_entity
    if entity_name is None:
        row = conn.execute(
            "SELECT e.name FROM entities e "
            "JOIN fact_entities fe ON fe.entity_id = e.entity_id "
            "GROUP BY e.entity_id "
            "ORDER BY COUNT(*) DESC LIMIT 1"
        ).fetchone()
        if row is None or not row["name"]:
            return _check("smoke_probe", "ok",
                          "no entities to probe (empty store)")
        entity_name = str(row["name"])

    # Read every encoded fact's vector once. Strategy A end-to-end.
    rows = conn.execute(
        "SELECT fact_id, trust_score, helpful_count, hrr_vector "
        "FROM facts WHERE hrr_vector IS NOT NULL"
    ).fetchall()
    if not rows:
        return _check("smoke_probe", "ok", "no encoded facts to probe")

    t0 = time.monotonic()
    try:
        role_entity = hrr.encode_atom("__hrr_role_entity__", hrr_dim)
        target_atom = hrr.encode_atom(entity_name.lower(), hrr_dim)
        best_sim = -1.0
        for r in rows:
            try:
                v = hrr.bytes_to_phases(r["hrr_vector"])
            except Exception:
                # vector_shape check will have flagged this already
                continue
            residual = hrr.unbind(v, role_entity)
            sim = hrr.similarity(residual, target_atom)
            if sim > best_sim:
                best_sim = sim
    except Exception as exc:
        return _check("smoke_probe", "error", f"probe raised: {exc}")
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    if elapsed_ms > _BUDGET_MS:
        return _check(
            "smoke_probe", "warn",
            f"probe('{entity_name}') took {elapsed_ms}ms over {len(rows)} "
            f"facts — budget is {_BUDGET_MS}ms",
            entity=entity_name, elapsed_ms=elapsed_ms, n_facts=len(rows),
            best_sim=round(best_sim, 4),
        )
    if best_sim < _NOISE_FLOOR:
        return _check(
            "smoke_probe", "warn",
            f"probe('{entity_name}') best raw_sim={best_sim:+.4f} below "
            f"noise floor {_NOISE_FLOOR:.2f} — corpus may not contain this "
            f"entity, or vectors are stale",
            entity=entity_name, elapsed_ms=elapsed_ms, n_facts=len(rows),
            best_sim=round(best_sim, 4),
        )
    return _check(
        "smoke_probe", "ok",
        f"probe('{entity_name}') best raw_sim={best_sim:+.4f} in "
        f"{elapsed_ms}ms over {len(rows)} facts",
        entity=entity_name, elapsed_ms=elapsed_ms, n_facts=len(rows),
        best_sim=round(best_sim, 4),
    )
