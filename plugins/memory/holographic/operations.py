"""Production operations for stock Holographic memory.

Deterministic, local-only, zero LLM, zero network. Provides:

- health_check(): read-only CHECK across 11 categories
- repair(): CHECK (default) / REPAIR_DERIVED / REPAIR_SAFE / REPORT
- create_backup() / verify_backup() / restore_to_scratch() / rotate_backups()
- run_maintenance(): NORMAL / LIGHT / DEEP, budgeted, idempotent, resumable
- migration_status() / ensure_migration(): additive, idempotent state machine
- circuit breaker, bounded event log, redacted stats/metrics
- network_dependencies(): introspectable offline contract (always [])

Design rules (R6 governing principle):
- CHECK paths never write. Mutation only via explicit repair/maintenance/backup.
- Backups are verbatim snapshots (fidelity); metadata/events/stats never carry
  fact content. This module stores NO secrets and invents NO config syntax.
- Restore NEVER touches the production DB (scratch path only).
- The provider is NOT auto-wired: startup/session hooks stay untouched so the
  stock baseline cannot drift. Call startup_check() explicitly where wanted.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

OPS_SCHEMA_VERSION = 1
PROVIDER_VERSION = "holographic-stock"
EVENT_LIMIT = 200
CIRCUIT_THRESHOLD = 3
DEFAULT_KEEP = 5

_HEALTHY = "HEALTHY"
_DEGRADED = "DEGRADED"
_NEEDS_REPAIR = "NEEDS_REPAIR"
_BLOCKED = "BLOCKED"

_EXPECTED_TABLES = ("facts", "entities", "fact_entities", "memory_banks", "facts_fts")

# Minimal local suspicious-content shapes for the *security* health section
# (diagnostic counting only; stock write path stays unscreened by design).
_SECRET_RES = (
    re.compile(r"sk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._~+/-]{8,}", re.IGNORECASE),
    re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
    re.compile(r"(?i)(?:api[_-]?key|secret|token|password)\s*[:=]\s*\S{4,}"),
)
_INSTRUCTION_RES = (
    re.compile(r"(?i)ignore\s+(all\s+)?previous\s+instructions"),
    re.compile(r"(?i)^system\s*:"),
    re.compile(r"(?i)disregard\s+.*(policy|instructions|safety)"),
)

# Event detail keys that must never be persisted (defense in depth; callers
# already pass count-only payloads).
_FORBIDDEN_DETAIL_KEYS = ("content", "text", "secret", "token", "password", "api_key")


def network_dependencies() -> list:
    """Introspectable offline contract: operations needs no network."""
    return []


def hrr_available() -> bool:
    """Whether the HRR vector layer exists (numpy present). Bank checks gate on this."""
    try:
        from . import holographic as hrr
        return bool(hrr._HAS_NUMPY)
    except Exception:
        return False


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- ops tables

_OPS_SQL = """
CREATE TABLE IF NOT EXISTS ops_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ops_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       TEXT NOT NULL,
    type     TEXT NOT NULL,
    detail   TEXT DEFAULT '{}',
    ok       INTEGER DEFAULT 1
);
"""


def _tables(conn: sqlite3.Connection) -> set:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view')").fetchall()
    return {r[0] for r in rows}


def ensure_migration(store) -> dict:
    """Create ops tables + record version. Additive, idempotent, transactional."""
    with store._lock:
        before = _tables(store._conn)
        with store._conn:  # transaction: both tables + version land together
            store._conn.executescript(_OPS_SQL)
            store._conn.execute(
                "INSERT OR IGNORE INTO ops_meta(key, value) VALUES ('ops_schema_version', ?)",
                (str(OPS_SCHEMA_VERSION),))
            store._conn.execute(
                "INSERT OR IGNORE INTO ops_meta(key, value) VALUES ('migration_state', 'COMPLETE')",
                )
        after = _tables(store._conn)
        created = sorted(after - before)
    return {"state": "COMPLETE", "additive": True, "created": created,
            "schema_version": OPS_SCHEMA_VERSION, "llm_calls": 0}


def migration_status(store) -> dict:
    """Read-only migration state machine readout (never writes)."""
    try:
        with store._lock:
            if "ops_meta" not in _tables(store._conn):
                return {"state": "READY", "schema_version": OPS_SCHEMA_VERSION,
                        "additive": True, "detail": "ops tables not yet created",
                        "llm_calls": 0}
            rows = dict(store._conn.execute("SELECT key, value FROM ops_meta").fetchall())
    except Exception as e:
        return {"state": "BLOCKED", "schema_version": OPS_SCHEMA_VERSION,
                "additive": True, "detail": f"unreadable: {e}", "llm_calls": 0}
    state = rows.get("migration_state", "READY")
    if state not in ("NOT_REQUIRED", "READY", "RUNNING", "COMPLETE", "FAILED", "BLOCKED"):
        state = "BLOCKED"
    return {"state": state, "schema_version": rows.get("ops_schema_version", "?"),
            "additive": True, "detail": "recorded", "llm_calls": 0}


# ---------------------------------------------------------------- events

def _scrub(detail: dict) -> dict:
    clean = {}
    for k, v in dict(detail).items():
        if k.lower() in _FORBIDDEN_DETAIL_KEYS:
            continue
        if isinstance(v, dict):  # one-level nested scrub for future callers
            v = {nk: nv for nk, nv in v.items()
                 if nk.lower() not in _FORBIDDEN_DETAIL_KEYS}
        clean[k] = v
    return clean


def _record(store, etype: str, detail: dict | None = None, ok: bool = True) -> None:
    try:
        with store._lock:
            if "ops_events" not in _tables(store._conn):
                return  # CHECK paths / fresh DBs stay write-free
            store._conn.execute(
                "INSERT INTO ops_events(ts, type, detail, ok) VALUES (?, ?, ?, ?)",
                (_utcnow(), etype, json.dumps(_scrub(detail or {})), 1 if ok else 0))
            store._conn.execute(
                "DELETE FROM ops_events WHERE event_id NOT IN "
                "(SELECT event_id FROM ops_events ORDER BY event_id DESC LIMIT ?)",
                (EVENT_LIMIT,))
            store._conn.commit()
    except Exception:
        pass  # diagnostics must never break production paths


def recent_events(store, limit: int = 50) -> list:
    try:
        with store._lock:
            if "ops_events" not in _tables(store._conn):
                return []
            rows = store._conn.execute(
                "SELECT ts, type, detail, ok FROM ops_events "
                "ORDER BY event_id DESC LIMIT ?", (max(1, min(limit, EVENT_LIMIT)),)).fetchall()
    except Exception:
        return []
    out = []
    for ts, etype, detail, ok in rows:
        try:
            d = json.loads(detail or "{}")
        except Exception:
            d = {}
        out.append({"ts": ts, "type": etype, "detail": d, "ok": bool(ok)})
    return [{"ts": t, "type": e} | ({"detail": d} if d else {}) for t, e, d in
            [(o["ts"], o["type"], o["detail"]) for o in out]]


# ---------------------------------------------------------------- health

def _suspicious(text: str) -> bool:
    t = text or ""
    return any(r.search(t) for r in _SECRET_RES + _INSTRUCTION_RES)


def health_check(store, config: dict | None = None) -> dict:
    """Read-only health across 11 categories. Never writes. Zero LLM."""
    rep: dict = {"llm_calls": 0, "checked_at": _utcnow()}
    try:
        with store._lock:
            conn = store._conn
            conn.execute("SELECT 1").fetchone()
            rep["database"] = {"readable": True, "path": str(store.db_path)}
            try:
                rep["database"]["journal_mode"] = conn.execute(
                    "PRAGMA journal_mode").fetchone()[0]
                rep["database"]["page_size"] = conn.execute("PRAGMA page_size").fetchone()[0]
                rep["database"]["page_count"] = conn.execute("PRAGMA page_count").fetchone()[0]
            except Exception:
                pass
            tables = _tables(conn)
            missing = [t for t in _EXPECTED_TABLES if t not in tables]
            rep["schema"] = {"valid": not missing, "missing": missing}
            if missing:
                rep["status"] = _NEEDS_REPAIR
                return _finalize(rep)
            cats = [r[0] for r in conn.execute("SELECT DISTINCT category FROM facts").fetchall()]
            banks = {r[0]: r[1] for r in conn.execute(
                "SELECT bank_name, fact_count FROM memory_banks").fetchall()}
            hrr_avail = bool(getattr(store, "_hrr_available", True))
            expected_banks = {f"cat:{c}" for c in cats} if hrr_avail else set()
            missing_banks = sorted(expected_banks - set(banks))
            stale_banks = sorted(set(banks) - expected_banks)
            rep["indexes"] = {"consistent": not missing_banks and not stale_banks,
                              "missing_banks": missing_banks,
                              "stale_banks": stale_banks, "hrr_available": hrr_avail,
                              "bank_count": len(banks), "category_count": len(cats)}
            try:
                fts_n = conn.execute("SELECT COUNT(*) FROM facts_fts").fetchone()[0]
            except Exception:
                fts_n = -1
            fact_n = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            rep["indexes"]["fts_rows"] = fts_n
            rep["indexes"]["fact_rows"] = fact_n
            # Unreadable FTS with a present table = corruption until proven
            # otherwise; a missing table is already caught by the schema check.
            if fts_n < 0 or fts_n != fact_n:
                rep["indexes"]["consistent"] = False
            orphans = conn.execute(
                "SELECT COUNT(*) FROM fact_entities fe LEFT JOIN facts f "
                "ON fe.fact_id = f.fact_id WHERE f.fact_id IS NULL").fetchone()[0]
            orphan_e = conn.execute(
                "SELECT COUNT(*) FROM fact_entities fe LEFT JOIN entities e "
                "ON fe.entity_id = e.entity_id WHERE e.entity_id IS NULL").fetchone()[0]
            rep["lineage"] = {"orphan_count": orphans + orphan_e,
                              "fact_links": conn.execute("SELECT COUNT(*) FROM fact_entities").fetchone()[0]}
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            bad_time = conn.execute(
                "SELECT COUNT(*) FROM facts WHERE updated_at < created_at "
                "OR created_at > ? OR updated_at > ?", (now, now)).fetchone()[0]
            rep["temporal"] = {"invalid_transition_count": bad_time}
            bad_trust = conn.execute(
                "SELECT COUNT(*) FROM facts WHERE trust_score IS NULL "
                "OR trust_score < 0 OR trust_score > 1").fetchone()[0]
            rep["trust"] = {"out_of_range_count": bad_trust}
            unsafe = sum(1 for (c,) in conn.execute("SELECT content FROM facts") if _suspicious(c))
            rep["security"] = {"unsafe_context_count": unsafe}
            rep["cache"] = {"bounded": True,
                            "note": "stock store holds no in-memory fact cache"}
            cfg = dict(config or {})
            known = ("db_path", "auto_extract", "default_trust", "hrr_dim",
                     "hrr_weight", "min_trust_threshold", "temporal_decay_half_life")
            rep["config"] = {"known_keys": [k for k in known if k in cfg],
                             "unknown_keys": sorted(set(cfg) - set(known))}
            mig = "COMPLETE" if "ops_meta" in tables else "READY"
            rep["migration"] = {"state": mig, "ops_tables": "ops_meta" in tables}
            try:
                size = Path(store.db_path).stat().st_size
            except OSError:
                size = -1
            fb = conn.execute("SELECT COUNT(*) FROM fact_feedback").fetchone()[0] \
                if "fact_feedback" in tables else 0
            rep["storage"] = {"db_bytes": size, "facts": fact_n,
                              "entities": conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0],
                              "feedback_rows": fb}
    except Exception as e:
        rep["database"] = {"readable": False, "error": type(e).__name__}
        rep["status"] = _BLOCKED
        return _finalize(rep)
    return _finalize(rep)


def _finalize(rep: dict) -> dict:
    if rep.get("status"):
        return rep
    idx_ok = rep.get("indexes", {}).get("consistent", True)
    schema_ok = rep.get("schema", {}).get("valid", True)
    if not schema_ok or not idx_ok:
        rep["status"] = _NEEDS_REPAIR
    elif (rep.get("lineage", {}).get("orphan_count", 0) > 0
          or rep.get("temporal", {}).get("invalid_transition_count", 0) > 0
          or rep.get("trust", {}).get("out_of_range_count", 0) > 0
          or rep.get("security", {}).get("unsafe_context_count", 0) > 0):
        rep["status"] = _DEGRADED
    else:
        rep["status"] = _HEALTHY
    rep["warnings"] = [f"{k} needs attention" for k in
                       ("lineage", "temporal", "trust", "security")
                       if rep.get(k) and list(rep[k].values())[0] > 0]
    return rep


def startup_check(store, config: dict | None = None) -> dict:
    """Explicit startup self-check entry point (pure health; caller wires it)."""
    return health_check(store, config=config)


# ---------------------------------------------------------------- repair

def repair(store, mode: str = "CHECK") -> dict:
    """CHECK (default, read-only) / REPAIR_DERIVED / REPAIR_SAFE / REPORT."""
    mode = (mode or "CHECK").upper()
    if mode not in ("CHECK", "REPAIR_DERIVED", "REPAIR_SAFE", "REPORT"):
        raise ValueError(f"unknown repair mode: {mode}")
    before = health_check(store)
    out: dict = {"mode": mode, "before": before["status"], "repaired": [],
                 "ok": True, "llm_calls": 0}
    if mode in ("REPAIR_DERIVED", "REPAIR_SAFE"):
        ensure_migration(store)
        for bank in before.get("indexes", {}).get("missing_banks", []):
            cat = bank[4:] if bank.startswith("cat:") else bank
            try:
                with store._lock:
                    store._rebuild_bank(cat)
                out["repaired"].append(bank)
            except Exception as e:
                out["ok"] = False
                out["error"] = f"{bank}: {type(e).__name__}"
                _record(store, "REPAIR_FAILED", {"bank": bank}, ok=False)
                break
        if out["ok"]:
            # Stale banks are derived garbage (rebuild recreates them if facts
            # return); pruning keeps "consistent" honest.
            for bank in before.get("indexes", {}).get("stale_banks", []):
                try:
                    with store._lock:
                        store._conn.execute("DELETE FROM memory_banks WHERE bank_name = ?",
                                            (bank,))
                        store._conn.commit()
                    out["repaired"].append(f"pruned:{bank}")
                except Exception as e:
                    out["ok"] = False
                    out["error"] = f"{bank}: {type(e).__name__}"
                    _record(store, "REPAIR_FAILED", {"bank": bank}, ok=False)
                    break
        if mode == "REPAIR_SAFE" and out["ok"]:
            try:
                with store._lock:
                    store._conn.execute("PRAGMA incremental_vacuum").fetchone()
                out["repaired"].append("incremental_vacuum")
            except Exception as e:
                out["ok"] = False
                out["error"] = type(e).__name__
        if out["repaired"] and out["ok"]:
            _record(store, "REPAIR_COMPLETED", {"repaired": out["repaired"]})
    out["after"] = health_check(store)["status"]
    if mode == "REPORT":
        storage = before.get("storage") or {}
        lineage = before.get("lineage") or {}
        security = before.get("security") or {}
        out["report"] = (f"status={out['after']} facts={storage.get('facts', '?')} "
                         f"orphans={lineage.get('orphan_count', '?')} "
                         f"unsafe={security.get('unsafe_context_count', '?')}")
    return out


# ---------------------------------------------------------------- backup

def _counts(conn: sqlite3.Connection) -> dict:
    return {
        "facts": conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0],
        "entities": conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0],
        "links": conn.execute("SELECT COUNT(*) FROM fact_entities").fetchone()[0],
        "banks": conn.execute("SELECT COUNT(*) FROM memory_banks").fetchone()[0],
    }


def create_backup(store, backup_dir) -> dict:
    """Consistent snapshot via SQLite online backup + metadata sidecar."""
    ensure_migration(store)
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = backup_dir / f"holographic-{stamp}.db"
    i = 1
    while dest.exists():
        dest = backup_dir / f"holographic-{stamp}-{i}.db"
        i += 1
    with store._lock:
        counts = _counts(store._conn)
        src = sqlite3.connect(f"file:{store.db_path}?mode=ro", uri=True, timeout=30)
        try:
            dst = sqlite3.connect(str(dest), timeout=30)
            try:
                src.backup(dst)
            finally:
                dst.close()
        finally:
            src.close()
    sha = hashlib.sha256(dest.read_bytes()).hexdigest()
    meta = {"path": str(dest), "sha256": sha, "created_at": _utcnow(),
            "schema_version": OPS_SCHEMA_VERSION, "provider": PROVIDER_VERSION,
            "scope": str(store.db_path), "fact_count": counts["facts"],
            "counts": counts, "llm_calls": 0,
            "note": "verbatim snapshot; may contain user-stored secrets — protect accordingly"}
    dest.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    _record(store, "BACKUP_CREATED", {"facts": counts["facts"], "sha256": sha[:12]})
    return meta


def verify_backup(backup_path: str) -> dict:
    """Checksum (if sidecar present) + readability + schema + integrity. Fail closed."""
    p = Path(backup_path)
    if not p.exists():
        return {"ok": False, "reason": "missing file", "llm_calls": 0}
    meta_path = p.with_suffix(".meta.json")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {"ok": False, "reason": "unreadable sidecar", "llm_calls": 0}
        if meta.get("sha256") != hashlib.sha256(p.read_bytes()).hexdigest():
            return {"ok": False, "reason": "checksum mismatch", "llm_calls": 0}
    try:
        conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=30)
        try:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view')").fetchall()}
            missing = [t for t in _EXPECTED_TABLES if t not in tables]
            if missing:
                return {"ok": False, "reason": f"schema missing: {missing}", "llm_calls": 0}
            integ = conn.execute("PRAGMA integrity_check").fetchone()[0]
            if integ != "ok":
                return {"ok": False, "reason": f"integrity: {integ}", "llm_calls": 0}
            n = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        finally:
            conn.close()
    except Exception as e:
        return {"ok": False, "reason": f"unreadable: {type(e).__name__}", "llm_calls": 0}
    return {"ok": True, "fact_count": n, "llm_calls": 0}


def restore_to_scratch(backup_path: str, dest_path: str) -> dict:
    """Verify, then copy to a scratch path. NEVER touches production."""
    v = verify_backup(backup_path)
    if not v["ok"]:
        return {"ok": False, "reason": v["reason"], "llm_calls": 0}
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(Path(backup_path).read_bytes())
    v2 = verify_backup(str(dest))
    if not v2["ok"]:
        try:
            dest.unlink()
        except OSError:
            pass
        return {"ok": False, "reason": f"scratch copy invalid: {v2['reason']}", "llm_calls": 0}
    return {"ok": True, "fact_count": v2["fact_count"], "path": str(dest), "llm_calls": 0}


def rotate_backups(backup_dir, keep: int = DEFAULT_KEEP) -> dict:
    """Keep newest `keep` verified-good backups. Never deletes the last good one."""
    d = Path(backup_dir)
    cands = sorted(d.glob("holographic-*.db"))
    good = [c for c in cands if verify_backup(str(c))["ok"]]
    good.sort(key=lambda c: c.stat().st_mtime)
    removable = good[:max(0, len(good) - max(1, keep))]
    # never remove if it would leave zero verified-good backups
    if len(good) - len(removable) < 1:
        removable = []
    removed = 0
    for c in removable:
        try:
            c.unlink()
            side = c.with_suffix(".meta.json")
            if side.exists():
                side.unlink()
            removed += 1
        except OSError:
            pass
    return {"kept": len(good) - removed, "removed": removed,
            "total": len(cands), "llm_calls": 0}


# ---------------------------------------------------------------- maintenance

_MAINTENANCE_STEPS = {
    "NORMAL": ("health", "stats", "prune_events"),
    "LIGHT": ("health", "stats", "prune_events", "ensure_migration", "bank_consistency"),
    "DEEP": ("health", "stats", "prune_events", "ensure_migration", "bank_consistency",
             "integrity", "fts_consistency"),
}


def _over(budget_ms: int, t0: float) -> bool:
    # NOTE: >= (not >) so budget_ms=0 trips immediately even on coarse
    # Windows clocks (~15ms granularity); checks remain per-step.
    return (time.monotonic() - t0) * 1000 >= budget_ms


def run_maintenance(store, mode: str = "NORMAL", budget_ms: int = 60000) -> dict:
    """Bounded, idempotent maintenance. Resumable by rerun. Zero LLM."""
    mode = (mode or "NORMAL").upper()
    if mode not in _MAINTENANCE_STEPS:
        raise ValueError(f"unknown maintenance mode: {mode}")
    t0 = time.monotonic()
    out: dict = {"mode": mode, "ok": True, "completed": [], "repaired": [],
                 "llm_calls": 0}
    _record(store, "MAINTENANCE_STARTED", {"mode": mode})
    try:
        for step in _MAINTENANCE_STEPS[mode]:
            if _over(budget_ms, t0):
                out["ok"] = False
                out["status"] = "BUDGET_EXCEEDED"
                out["resume"] = "rerun (idempotent)"
                break
            if step == "health":
                out["health"] = health_check(store)["status"]
                out["completed"].append(step)
            elif step == "stats":
                out["stats"] = get_stats(store)
                out["completed"].append(step)
            elif step == "prune_events":
                _record(store, "PRUNE", {})  # insert triggers bound rotation
                out["completed"].append(step)
            elif step == "ensure_migration":
                ensure_migration(store)
                out["completed"].append(step)
            elif step == "bank_consistency":
                h = health_check(store)
                for bank in h.get("indexes", {}).get("missing_banks", []):
                    cat = bank[4:] if bank.startswith("cat:") else bank
                    with store._lock:
                        store._rebuild_bank(cat)
                    out["repaired"].append(bank)
                for bank in h.get("indexes", {}).get("stale_banks", []):
                    with store._lock:
                        store._conn.execute("DELETE FROM memory_banks WHERE bank_name = ?",
                                            (bank,))
                        store._conn.commit()
                    out["repaired"].append(f"pruned:{bank}")
                out["completed"].append(step)
            elif step == "integrity":
                with store._lock:
                    integ = store._conn.execute("PRAGMA integrity_check").fetchone()[0]
                out["integrity"] = integ
                if integ != "ok":
                    out["ok"] = False
                out["completed"].append(step)
            elif step == "fts_consistency":
                h = health_check(store)
                out["fts"] = {"rows": h["indexes"].get("fts_rows"),
                              "facts": h["indexes"].get("fact_rows")}
                out["completed"].append(step)
            else:  # health/stats handled above; anything else just records
                out["completed"].append(step)
        out["elapsed_ms"] = round((time.monotonic() - t0) * 1000, 2)
        _record(store, "MAINTENANCE_COMPLETED",
                {"mode": mode, "repaired": len(out["repaired"])}, ok=out["ok"])
        if out["ok"]:
            _record_success(store, f"maintenance-{mode}")
        else:
            # Any incomplete run — including BUDGET_EXCEEDED — is a failure for
            # circuit purposes; a starved loop must trip the breaker, not reset it.
            _record_failure(store, f"maintenance-{mode}")
    except Exception as e:
        out["ok"] = False
        out["error"] = type(e).__name__
        out["elapsed_ms"] = round((time.monotonic() - t0) * 1000, 2)
        _record(store, "MAINTENANCE_COMPLETED", {"mode": mode, "error": out["error"]}, ok=False)
        _record_failure(store, f"maintenance-{mode}")
    return out


# ---------------------------------------------------------------- circuit

def _fail_count(store, op: str) -> int:
    try:
        with store._lock:
            if "ops_meta" not in _tables(store._conn):
                return 0
            row = store._conn.execute("SELECT value FROM ops_meta WHERE key = ?",
                                      (f"circuit_{op}",)).fetchone()
            return int(row[0]) if row else 0
    except Exception:
        return 0


def _record_failure(store, op: str) -> None:
    try:
        ensure_migration(store)
        with store._lock:
            n = _fail_count(store, op) + 1
            store._conn.execute("INSERT OR REPLACE INTO ops_meta(key, value) VALUES (?, ?)",
                                (f"circuit_{op}", str(n)))
            store._conn.commit()
        if n >= CIRCUIT_THRESHOLD:
            _record(store, "CIRCUIT_OPENED", {"op": op, "failures": n})
    except Exception:
        pass


def _record_success(store, op: str) -> None:
    try:
        with store._lock:
            if "ops_meta" not in _tables(store._conn):
                return
            store._conn.execute("DELETE FROM ops_meta WHERE key = ?", (f"circuit_{op}",))
            store._conn.commit()
    except Exception:
        pass


def circuit_state(store, op: str) -> str:
    """CLOSED normally; OPEN after CIRCUIT_THRESHOLD consecutive failures."""
    return "OPEN" if _fail_count(store, op) >= CIRCUIT_THRESHOLD else "CLOSED"


# ---------------------------------------------------------------- stats/metrics

def get_stats(store) -> dict:
    """Redacted operational stats (counts/sizes only, never content)."""
    with store._lock:
        counts = _counts(store._conn)
        try:
            size = Path(store.db_path).stat().st_size
        except OSError:
            size = -1
        tables = _tables(store._conn)
        events = store._conn.execute("SELECT COUNT(*) FROM ops_events").fetchone()[0] \
            if "ops_events" in tables else 0
    out = {"facts": counts["facts"], "entities": counts["entities"],
           "links": counts["links"], "banks": counts["banks"],
           "db_bytes": size, "events": events, "llm_calls": 0}
    try:
        mig = migration_status(store)
        out["migration_state"] = mig["state"]
    except Exception:
        out["migration_state"] = "BLOCKED"
    return out


def get_metrics(store) -> dict:
    """Local metrics incl. LLM counter (always 0) and repair/backup tallies."""
    st = get_stats(store)
    evts = recent_events(store, limit=EVENT_LIMIT)
    m = {"memory_count": st["facts"], "active_count": st["facts"],
         "stale_count": 0, "superseded_count": 0, "quarantine_count": 0,
         "conflict_count": 0, "orphan_count": 0,
         "repair_success": sum(1 for e in evts if e["type"] == "REPAIR_COMPLETED"),
         "repair_failure": sum(1 for e in evts if e["type"] == "REPAIR_FAILED"),
         "backup_success": sum(1 for e in evts if e["type"] == "BACKUP_CREATED"),
         "backup_failure": 0, "llm_calls": 0}
    try:
        h = health_check(store)
        m["orphan_count"] = h.get("lineage", {}).get("orphan_count", 0)
        m["quarantine_count"] = h.get("security", {}).get("unsafe_context_count", 0)
    except Exception:
        pass
    return m
