"""hexis_appraisal state store — the plugin's ENTIRE SQLite surface.

No other module in this plugin may import sqlite3 or touch state.db.

Contract (see .planning/phases/01-skeleton-state/01-CONTEXT.md, all locked):
- Location: $HERMES_HOME/hexis_appraisal/state.db, resolved via the host's
  get_hermes_home() with an env-var fallback. Never a literal path.
- PRAGMAs: journal_mode=WAL (persistent, set at creation), synchronous=NORMAL
  and busy_timeout on every write connection. Hot-path reads open read-only
  URI connections (file:<path>?mode=ro) and never create files.
- Write funnels: ALL state-table writes go through apply_deltas() in one
  transaction. Telemetry writes go through record_telemetry() — the only
  other write path, a single quick INSERT + cap eviction designed to be
  hot-path safe (OBS-01).
- Disposable-state doctrine: ANY structural problem (corruption, schema_version
  mismatch, missing table, failed open) -> quarantine + recreate fresh. No
  migration framework. v2 DBs quarantine-recreate at v3 on the first
  ensure_db after deploy (Phase-2 telemetry evidence is durable in
  02-VALIDATION.md) — expected, not a regression.
- No public function in this module may raise to a caller. Failure values:
  ensure_db -> False, read_snapshot -> None, apply_deltas -> False,
  get_meta -> None, read_turns_since -> [].
"""

import json
import logging
import os
import sqlite3
import statistics
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("hermes.plugins.hexis_appraisal.store")

SCHEMA_VERSION = 3

# Per-table row caps (seed values; tune later with telemetry).
CAPS = {
    "concerns": 20,
    "contradictions": 50,
    "turn_log": 500,
    "trust_scores": 64,
    "telemetry": 2000,
}

_DEFAULT_BUSY_TIMEOUT_MS = 5000

_TABLES = (
    "meta",
    "affect_summary",
    "concerns",
    "contradictions",
    "trust_scores",
    "turn_log",
    "telemetry",
)

# Caps/decay columns (expires_at, decayed_weight) are in the schema NOW even
# though only Phase 3 writes them — schema churn is the expensive part.
_SCHEMA_DDL = (
    """CREATE TABLE meta            (key TEXT PRIMARY KEY, value TEXT NOT NULL)""",
    """CREATE TABLE affect_summary  (id INTEGER PRIMARY KEY CHECK (id=1), summary TEXT,
                                     valence REAL, arousal REAL, intensity REAL, updated_at TEXT)""",
    """CREATE TABLE concerns        (id INTEGER PRIMARY KEY, text TEXT NOT NULL, weight REAL DEFAULT 1.0,
                                     decayed_weight REAL, status TEXT DEFAULT 'open', expires_at TEXT,
                                     created_at TEXT, updated_at TEXT)""",
    """CREATE TABLE contradictions  (id INTEGER PRIMARY KEY, kind TEXT CHECK (kind IN
                                     ('semantic','narrative','relational','emotional')),
                                     description TEXT NOT NULL, evidence TEXT,
                                     resolved INTEGER DEFAULT 0, decayed_weight REAL, expires_at TEXT,
                                     created_at TEXT)""",
    """CREATE TABLE trust_scores    (key TEXT PRIMARY KEY, value REAL NOT NULL, updated_at TEXT)""",
    # Schema v3 (REFL-01..03): turn_log gains assistant_excerpt (reflection
    # digests need the assistant side of each turn); the meta table carries
    # the reflection watermark keys (last_reflected_turn_log_id,
    # last_seen_session_id). Existing v2 DBs fail _verify_structure (version
    # mismatch) and are quarantine-recreated — disposable-state doctrine,
    # no migration code.
    """CREATE TABLE turn_log        (id INTEGER PRIMARY KEY, session_id TEXT, turn_id TEXT,
                                     user_excerpt TEXT, assistant_excerpt TEXT,
                                     appraisal_json TEXT, created_at TEXT)""",
    # Schema v2 (OBS-01): per-appraisal-call telemetry. Existing v1 DBs fail
    # _verify_structure (version mismatch + missing table) and are
    # quarantine-recreated — disposable-state doctrine, no migration code.
    """CREATE TABLE telemetry       (id INTEGER PRIMARY KEY, ts TEXT NOT NULL, session_id TEXT,
                                     wall_ms INTEGER, model TEXT, tokens_in INTEGER,
                                     tokens_out INTEGER, outcome TEXT NOT NULL, error TEXT)""",
)


def get_db_path() -> Path:
    """Resolve $HERMES_HOME/hexis_appraisal/state.db. Never raises."""
    home = None
    try:
        # Lazy host import — keeps this module importable outside the host.
        from hermes_constants import get_hermes_home

        home = Path(get_hermes_home())
    except Exception:
        home = None
    if home is None:
        home = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
    return home / "hexis_appraisal" / "state.db"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _quarantine(db_path) -> None:
    """Move a structurally-broken DB (and WAL/SHM sidecars) out of the way.

    Renames state.db -> state.db.quarantined-<UTC ts>; on rename failure falls
    back to os.remove. Never raises — on total failure it returns and lets
    ensure_db report False when recreate fails.
    """
    try:
        path = Path(db_path)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        target = path.with_name(path.name + ".quarantined-" + ts)
        moved = False
        try:
            os.replace(path, target)
            moved = True
        except OSError:
            try:
                os.remove(path)
            except OSError:
                logger.warning(
                    "hexis state quarantine failed — could not rename or remove %s", path
                )
                return
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(path) + suffix)
            if sidecar.exists():
                try:
                    os.replace(sidecar, Path(str(target) + suffix))
                except OSError:
                    try:
                        os.remove(sidecar)
                    except OSError:
                        pass
        logger.warning(
            "hexis state DB quarantined: %s -> %s",
            path,
            target if moved else "(removed; rename failed)",
        )
    except Exception as exc:
        logger.warning("hexis quarantine error (continuing): %s", exc)
        logger.debug("quarantine failure detail", exc_info=True)


def _verify_structure(path: Path) -> bool:
    """True iff the DB at path passes quick_check, has all six tables, and
    meta schema_version matches SCHEMA_VERSION. Never raises."""
    conn = None
    try:
        conn = sqlite3.connect(str(path))
        row = conn.execute("PRAGMA quick_check").fetchone()
        if row is None or row[0] != "ok":
            return False
        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if not set(_TABLES) <= tables:
            return False
        row = conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()
        if row is None or str(row[0]) != str(SCHEMA_VERSION):
            return False
        return True
    except Exception:
        return False
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _create_fresh(path: Path) -> bool:
    """Create a new DB with the locked schema. WAL is set here (persistent)."""
    conn = None
    try:
        conn = sqlite3.connect(str(path))
        conn.execute("PRAGMA busy_timeout=%d" % _DEFAULT_BUSY_TIMEOUT_MS)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        with conn:
            for ddl in _SCHEMA_DDL:
                conn.execute(ddl)
            conn.execute(
                "INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
        return True
    except Exception as exc:
        logger.warning("hexis state DB creation failed (degrading): %s", exc)
        logger.debug("creation failure detail", exc_info=True)
        return False
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def ensure_db(db_path=None) -> bool:
    """Create the DB + schema if absent; verify structure if present.

    ANY structural failure (corrupt file, bad quick_check, missing table,
    schema_version mismatch) -> quarantine then recreate fresh. Returns True
    if a usable DB exists at exit, False otherwise. Never raises.
    """
    try:
        path = Path(db_path) if db_path is not None else get_db_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if _verify_structure(path):
                return True
            _quarantine(path)
        return _create_fresh(path)
    except Exception as exc:
        logger.warning("hexis ensure_db failed (degrading): %s", exc)
        logger.debug("ensure_db failure detail", exc_info=True)
        return False


def _rows_as_dicts(conn, table: str) -> list:
    cur = conn.execute("SELECT * FROM %s ORDER BY id" % table)
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


# Lazy concern decay (03-CONTEXT, locked): heartbeat semantics without a
# scheduler. Computed at READ time only — reads never write; rows that fell
# below the prune threshold actually die in the reflection pass
# (reflection.apply_reflection composes concerns_prune).
_DECAY_HALF_LIFE_DAYS = 7.0
DECAY_PRUNE_THRESHOLD = 0.1


def _effective_weight(row, now):
    """weight * 0.5 ** (days_idle / 7), days_idle from updated_at
    (fallback created_at). Non-numeric weight or unparseable timestamp ->
    no decay (effective == weight). Never raises."""
    weight = row.get("weight")
    try:
        weight_value = float(weight)
    except (TypeError, ValueError):
        return weight
    ts_text = row.get("updated_at") or row.get("created_at")
    try:
        ts = datetime.fromisoformat(str(ts_text))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        days_idle = max(0.0, (now - ts).total_seconds() / 86400.0)
    except (TypeError, ValueError):
        return weight_value
    return weight_value * 0.5 ** (days_idle / _DECAY_HALF_LIFE_DAYS)


def read_snapshot(db_path=None, include_decayed=False):
    """Hot-path read over a read-only URI connection.

    Returns the snapshot dict, or None on ANY error (absent file, locked,
    corrupt, missing table). Never raises, never creates files.

    Each concerns row gains ``effective_weight`` (lazy decay, see
    _effective_weight); rows whose effective weight fell below
    DECAY_PRUNE_THRESHOLD are EXCLUDED from the snapshot unless
    ``include_decayed=True`` (the reflection pass's raw view — it needs the
    decayed rows to compose concerns_prune). Reads never write.
    """
    conn = None
    try:
        path = Path(db_path) if db_path is not None else get_db_path()
        conn = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
        row = conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()
        if row is None:
            return None
        schema_version = int(row[0])
        affect = None
        cur = conn.execute("SELECT * FROM affect_summary WHERE id=1")
        arow = cur.fetchone()
        if arow is not None:
            affect = dict(zip([c[0] for c in cur.description], arow))
        now = datetime.now(timezone.utc)
        concerns = []
        for concern in _rows_as_dicts(conn, "concerns"):
            effective = _effective_weight(concern, now)
            concern["effective_weight"] = effective
            if not include_decayed:
                try:
                    if float(effective) < DECAY_PRUNE_THRESHOLD:
                        continue
                except (TypeError, ValueError):
                    pass  # non-numeric: include undecayed (defensive)
            concerns.append(concern)
        snapshot = {
            "schema_version": schema_version,
            "affect_summary": affect,
            "concerns": concerns,
            "contradictions": _rows_as_dicts(conn, "contradictions"),
            "trust_scores": {
                key: value
                for key, value in conn.execute("SELECT key, value FROM trust_scores")
            },
            "turn_log_count": conn.execute(
                "SELECT COUNT(*) FROM turn_log"
            ).fetchone()[0],
        }
        return snapshot
    except Exception as exc:
        logger.warning("hexis read_snapshot failed (degrading): %s", exc)
        logger.debug("read_snapshot failure detail", exc_info=True)
        return None
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def get_meta(key, db_path=None):
    """Read one meta value over a read-only URI connection.

    Returns the stored string, or None on absent key / absent file / lock /
    any other error. Never raises, never creates files.
    """
    conn = None
    try:
        path = Path(db_path) if db_path is not None else get_db_path()
        conn = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
        row = conn.execute(
            "SELECT value FROM meta WHERE key=?", (str(key),)
        ).fetchone()
        return row[0] if row is not None else None
    except Exception as exc:
        logger.debug("hexis get_meta(%r) failed (degrading): %s", key, exc)
        return None
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def read_turns_since(after_id, db_path=None, limit=50):
    """turn_log rows with id > after_id, ordered by id ascending, capped.

    Read-only URI connection; returns [] on ANY error (absent file, locked,
    corrupt). Never raises, never creates files.
    """
    conn = None
    try:
        path = Path(db_path) if db_path is not None else get_db_path()
        conn = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
        cur = conn.execute(
            "SELECT * FROM turn_log WHERE id > ? ORDER BY id LIMIT ?",
            (int(after_id), int(limit)),
        )
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception as exc:
        logger.debug("hexis read_turns_since failed (degrading): %s", exc)
        return []
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def apply_deltas(deltas: dict, db_path=None, busy_timeout_ms=None) -> bool:
    """THE single write entry point — one transaction, caps enforced inside it.

    Recognized delta keys: affect_summary, concerns_add, concerns_update,
    concerns_resolve, concerns_prune, contradictions_add,
    contradictions_resolve, trust_scores, turn_log_add, meta_set. Unknown
    keys are ignored with a debug log. This function stays MECHANICAL —
    clamping/policy (delta bounds, decay math, baselines) lives in the
    callers (reflection.py). Returns True on commit, False on ANY error
    (the transaction is rolled back). Never raises.
    """
    conn = None
    try:
        path = Path(db_path) if db_path is not None else get_db_path()
        if busy_timeout_ms is None:
            busy_timeout_ms = _DEFAULT_BUSY_TIMEOUT_MS
        now = _utc_now_iso()
        conn = sqlite3.connect(str(path))
        conn.execute("PRAGMA busy_timeout=%d" % int(busy_timeout_ms))
        conn.execute("PRAGMA synchronous=NORMAL")
        with conn:  # one transaction: commits on success, rolls back on error
            for key, payload in (deltas or {}).items():
                if key == "affect_summary":
                    conn.execute(
                        "INSERT INTO affect_summary"
                        " (id, summary, valence, arousal, intensity, updated_at)"
                        " VALUES (1, ?, ?, ?, ?, ?)"
                        " ON CONFLICT(id) DO UPDATE SET"
                        " summary=excluded.summary, valence=excluded.valence,"
                        " arousal=excluded.arousal, intensity=excluded.intensity,"
                        " updated_at=excluded.updated_at",
                        (
                            payload.get("summary"),
                            payload.get("valence"),
                            payload.get("arousal"),
                            payload.get("intensity"),
                            now,
                        ),
                    )
                elif key == "concerns_add":
                    for item in payload:
                        conn.execute(
                            "INSERT INTO concerns"
                            " (text, weight, status, created_at, updated_at)"
                            " VALUES (?, ?, 'open', ?, ?)",
                            (item.get("text"), item.get("weight", 1.0), now, now),
                        )
                elif key == "concerns_update":
                    # Absolute weights, pre-clamped by the caller
                    # (reflection.py owns the policy).
                    for item in payload:
                        conn.execute(
                            "UPDATE concerns SET weight=?, updated_at=?"
                            " WHERE id=?",
                            (item.get("weight"), now, item.get("id")),
                        )
                elif key == "concerns_resolve":
                    for concern_id in payload:
                        conn.execute(
                            "UPDATE concerns SET status='resolved', updated_at=?"
                            " WHERE id=?",
                            (now, concern_id),
                        )
                elif key == "concerns_prune":
                    # Decay-prune (the only deletion besides cap-eviction).
                    for concern_id in payload:
                        conn.execute(
                            "DELETE FROM concerns WHERE id=?", (concern_id,)
                        )
                elif key == "contradictions_add":
                    for item in payload:
                        conn.execute(
                            "INSERT INTO contradictions"
                            " (kind, description, evidence, created_at)"
                            " VALUES (?, ?, ?, ?)",
                            (
                                item.get("kind"),
                                item.get("description"),
                                item.get("evidence"),
                                now,
                            ),
                        )
                elif key == "contradictions_resolve":
                    for contradiction_id in payload:
                        conn.execute(
                            "UPDATE contradictions SET resolved=1 WHERE id=?",
                            (contradiction_id,),
                        )
                elif key == "meta_set":
                    # Reflection bookkeeping (watermark + last-seen session);
                    # values stored as str.
                    for meta_key, meta_value in (payload or {}).items():
                        conn.execute(
                            "INSERT INTO meta (key, value) VALUES (?, ?)"
                            " ON CONFLICT(key) DO UPDATE SET"
                            " value=excluded.value",
                            (str(meta_key), str(meta_value)),
                        )
                elif key == "trust_scores":
                    for score_key, score_value in payload.items():
                        conn.execute(
                            "INSERT INTO trust_scores (key, value, updated_at)"
                            " VALUES (?, ?, ?)"
                            " ON CONFLICT(key) DO UPDATE SET"
                            " value=excluded.value, updated_at=excluded.updated_at",
                            (score_key, float(score_value), now),
                        )
                elif key == "turn_log_add":
                    for item in payload:
                        appraisal = item.get("appraisal_json")
                        if isinstance(appraisal, (dict, list)):
                            appraisal = json.dumps(appraisal, ensure_ascii=False)
                        conn.execute(
                            "INSERT INTO turn_log"
                            " (session_id, turn_id, user_excerpt,"
                            " assistant_excerpt, appraisal_json, created_at)"
                            " VALUES (?, ?, ?, ?, ?, ?)",
                            (
                                item.get("session_id"),
                                item.get("turn_id"),
                                item.get("user_excerpt"),
                                item.get("assistant_excerpt"),
                                appraisal,
                                now,
                            ),
                        )
                else:
                    logger.debug("apply_deltas: ignoring unknown delta key %r", key)
            # Enforce caps inside the SAME transaction: evict oldest rows
            # (lowest id) beyond each cap; trust_scores evicts oldest
            # updated_at beyond its cap.
            for table in ("concerns", "contradictions", "turn_log"):
                conn.execute(
                    "DELETE FROM {t} WHERE id NOT IN"
                    " (SELECT id FROM {t} ORDER BY id DESC LIMIT ?)".format(t=table),
                    (CAPS[table],),
                )
            conn.execute(
                "DELETE FROM trust_scores WHERE key NOT IN"
                " (SELECT key FROM trust_scores"
                "  ORDER BY updated_at DESC, rowid DESC LIMIT ?)",
                (CAPS["trust_scores"],),
            )
        return True
    except Exception as exc:
        logger.warning("hexis apply_deltas failed (degrading): %s", exc)
        logger.debug("apply_deltas failure detail", exc_info=True)
        return False
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def record_telemetry(outcome, *, wall_ms=None, model=None, tokens_in=None,
                     tokens_out=None, error=None, session_id=None,
                     db_path=None) -> bool:
    """Record one per-appraisal-call telemetry row (OBS-01).

    The only write path besides apply_deltas(): a single quick INSERT plus
    cap eviction in one transaction — hot-path safe by design. `outcome` is
    one of ok|timeout|parse_fail|llm_error|trust_fallback|skipped:<reason>
    (free-form after `skipped:`). `error` is truncated to 300 chars.
    Never raises; returns False on any failure (fail open, warning log only).
    """
    conn = None
    try:
        path = Path(db_path) if db_path is not None else get_db_path()
        if error is not None:
            error = str(error)[:300]
        conn = sqlite3.connect(str(path))
        conn.execute("PRAGMA busy_timeout=%d" % _DEFAULT_BUSY_TIMEOUT_MS)
        conn.execute("PRAGMA synchronous=NORMAL")
        with conn:  # one transaction: INSERT + cap eviction
            conn.execute(
                "INSERT INTO telemetry"
                " (ts, session_id, wall_ms, model, tokens_in, tokens_out,"
                " outcome, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    _utc_now_iso(),
                    session_id,
                    wall_ms,
                    model,
                    tokens_in,
                    tokens_out,
                    str(outcome),
                    error,
                ),
            )
            conn.execute(
                "DELETE FROM telemetry WHERE id NOT IN"
                " (SELECT id FROM telemetry ORDER BY id DESC LIMIT ?)",
                (CAPS["telemetry"],),
            )
        return True
    except Exception as exc:
        logger.warning("hexis record_telemetry failed (degrading): %s", exc)
        logger.debug("record_telemetry failure detail", exc_info=True)
        return False
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def telemetry_summary(db_path=None):
    """Derived OBS-01 view: failure counter + last error, by query.

    Read-only URI connection — never creates files. Returns
    {"total", "by_outcome", "failure_count", "last_error", "p50_wall_ms"}.
    Non-failures are exactly ok/trust_fallback/reflect_ok plus the
    skipped:* and reflect_skipped:* prefixes; failures are exactly
    timeout/llm_error/parse_fail/reflect_timeout/reflect_llm_error/
    reflect_parse_fail (exclusion-list shape on purpose — any future
    unknown outcome counts as a failure). last_error is the error of the
    newest failure row (same definition — skipped/trust_fallback/
    reflect_ok rows carry no error and are not failures), and p50_wall_ms
    is the median wall_ms over appraisal ok/trust_fallback rows only
    (reflect_* walls excluded). Returns None on any error. Never raises.
    """
    conn = None
    try:
        path = Path(db_path) if db_path is not None else get_db_path()
        conn = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
        by_outcome = {
            outcome: count
            for outcome, count in conn.execute(
                "SELECT outcome, COUNT(*) FROM telemetry GROUP BY outcome"
            )
        }
        total = sum(by_outcome.values())
        failure_count = sum(
            count
            for outcome, count in by_outcome.items()
            if outcome not in ("ok", "trust_fallback", "reflect_ok")
            and not outcome.startswith(("skipped:", "reflect_skipped:"))
        )
        row = conn.execute(
            "SELECT error FROM telemetry"
            " WHERE outcome NOT IN ('ok', 'trust_fallback', 'reflect_ok')"
            " AND outcome NOT LIKE 'skipped:%'"
            " AND outcome NOT LIKE 'reflect_skipped:%'"
            " ORDER BY id DESC LIMIT 1"
        ).fetchone()
        last_error = row[0] if row is not None else None
        wall_values = [
            r[0]
            for r in conn.execute(
                "SELECT wall_ms FROM telemetry"
                " WHERE outcome IN ('ok', 'trust_fallback')"
                " AND wall_ms IS NOT NULL"
            )
        ]
        p50 = int(statistics.median(wall_values)) if wall_values else None
        return {
            "total": total,
            "by_outcome": by_outcome,
            "failure_count": failure_count,
            "last_error": last_error,
            "p50_wall_ms": p50,
        }
    except Exception as exc:
        logger.warning("hexis telemetry_summary failed (degrading): %s", exc)
        logger.debug("telemetry_summary failure detail", exc_info=True)
        return None
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
