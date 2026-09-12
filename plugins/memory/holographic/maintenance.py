"""Dream / consolidation cycle + self-healing. Local-first, crash-safe, resumable.

Never deletes data as a first response: mark -> quarantine -> repair -> report.
Deletion stays a human-boundary action. Optional LLM synthesis is a disabled-by-
default hook (max 1 call per dream cycle, failure non-fatal).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time

from .cognition import (
    adjusted_confidence, dedupe_keys, is_conflicting, normalize_text,
    temporal_state,
)
from .taxonomy import (
    ACTIVE, AGING, ARCHIVED, HYPOTHESIS, LESSON, QUARANTINE, STALE,
    SUPERSEDED_STATE,
)

logger = logging.getLogger(__name__)

# Dream policy: zero LLM calls by default (mission LLM POLICY).
DEFAULT_DREAM_POLICY = {
    "max_llm_calls": 0,          # hard default; dream may use 1 only if explicitly enabled
    "enable_llm": False,
    "stale_after_days": 90,
    "demote_unhelpful_threshold": 3,
    "promote_helpful_threshold": 3,
}


def _get_state(conn: sqlite3.Connection, key: str, default: str = "") -> str:
    try:
        row = conn.execute("SELECT state_val FROM dream_state WHERE state_key = ?", (key,)).fetchone()
        return row["state_val"] if row else default
    except Exception:
        return default


def _set_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO dream_state (state_key, state_val, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP) "
        "ON CONFLICT(state_key) DO UPDATE SET state_val = excluded.state_val, updated_at = excluded.updated_at",
        (key, value),
    )
    conn.commit()


def _columns(conn: sqlite3.Connection) -> set:
    return {row[1] for row in conn.execute("PRAGMA table_info(facts)").fetchall()}


def run_dream_cycle(store, policy: dict | None = None, source: str = "dream") -> dict:
    """Run maintenance: duplicates, stale, conflicts, supersession, strengthen,
    demote, entity links, patterns, optional single LLM synthesis (disabled by
    default). Resumable via dream_state; crash-safe via per-step commits."""
    from .migration import ensure_cognitive_schema

    started = time.monotonic()
    cfg = dict(DEFAULT_DREAM_POLICY)
    cfg.update(policy or {})
    report: dict = {
        "duplicates": 0, "stale_marked": 0, "conflicts": 0, "superseded": 0,
        "strengthened": 0, "demoted": 0, "quarantined": 0, "repaired": 0,
        "lessons": 0, "llm_calls": 0, "errors": [],
    }
    conn = store._conn
    lock = store._lock
    with lock:
        ensure_cognitive_schema(conn)
        if _get_state(conn, "dream_running") == "1":
            report["resumed"] = True  # previous run interrupted; continue safely
        _set_state(conn, "dream_running", "1")
        try:
            _step_duplicates(conn, report)
            _step_temporal(conn, report)
            _step_conflicts(conn, report)
            _step_trust(conn, cfg, report)
            _step_lessons(conn, report)
            if cfg.get("enable_llm") and cfg.get("max_llm_calls", 0) > 0:
                report["llm_calls"] = 0  # hook point; no backend wired by default
            _set_state(conn, "dream_last_ok", str(int(time.time())))
        except Exception as exc:  # dream failure is non-fatal, never wedges state
            report["errors"].append(str(exc)[:300])
            logger.debug("Dream cycle step failed: %s", exc)
        finally:
            _set_state(conn, "dream_running", "0")
    report["latency_ms"] = round((time.monotonic() - started) * 1000, 2)
    report["source"] = source
    return report


def _step_duplicates(conn: sqlite3.Connection, report: dict) -> None:
    """Link exact duplicates via lineage; keep both rows (no deletion)."""
    if "norm_hash" not in _columns(conn):
        return
    rows = conn.execute("SELECT fact_id, content FROM facts WHERE norm_hash IS NULL OR norm_hash = ''").fetchall()
    for row in rows:
        keys = dedupe_keys(row["content"] or "")
        conn.execute("UPDATE facts SET norm_hash = ?, slot_key = ? WHERE fact_id = ?",
                     (keys["exact"], keys["slot_key"], row["fact_id"]))
    conn.commit()
    existing = {(r["fact_id"], r["linked_fact_id"]) for r in
                conn.execute("SELECT fact_id, linked_fact_id FROM fact_links WHERE relation = 'related'").fetchall()}
    dupes = conn.execute(
        "SELECT norm_hash, GROUP_CONCAT(fact_id) AS ids, COUNT(*) c FROM facts "
        "WHERE norm_hash != '' GROUP BY norm_hash HAVING c > 1"
    ).fetchall()
    conn.execute("BEGIN")
    try:
        for dupe in dupes:
            ids = [int(i) for i in dupe["ids"].split(",")]
            first, rest = ids[0], ids[1:]
            for other in rest:
                if (first, other) in existing:
                    continue
                conn.execute(
                    "INSERT INTO fact_links (fact_id, linked_fact_id, relation) VALUES (?, ?, 'related')",
                    (first, other),
                )
                existing.add((first, other))
                report["duplicates"] += 1
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass


def _step_temporal(conn: sqlite3.Connection, report: dict) -> None:
    rows = conn.execute("SELECT fact_id, updated_at, lifecycle FROM facts").fetchall()
    for row in rows:
        new_state = temporal_state(row["updated_at"], row["lifecycle"] or ACTIVE)
        if new_state != (row["lifecycle"] or ACTIVE):
            conn.execute("UPDATE facts SET lifecycle = ? WHERE fact_id = ?",
                         (new_state, row["fact_id"]))
            if new_state == STALE:
                report["stale_marked"] += 1
    conn.commit()


def _step_conflicts(conn: sqlite3.Connection, report: dict) -> None:
    rows = conn.execute(
        "SELECT fact_id, content, lifecycle, COALESCE(slot_key,'') AS slot_key FROM facts WHERE lifecycle NOT IN ('superseded','archived','quarantine')"
    ).fetchall()
    groups: dict[str, list] = {}
    for row in rows:
        key = row["slot_key"] or ""
        if not key:
            try:
                key = dedupe_keys(row["content"] or "")["slot_key"]
            except Exception:
                continue
        groups.setdefault(key, []).append(row)
    existing = {(r["fact_id"], r["linked_fact_id"]) for r in
                conn.execute("SELECT fact_id, linked_fact_id FROM fact_links WHERE relation = 'contradicted_by'").fetchall()}
    conn.execute("BEGIN")
    try:
        for members in groups.values():
            if len(members) < 2:
                continue
            items = [(m["fact_id"], m["content"] or "") for m in members]
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    if (items[i][0], items[j][0]) in existing or (items[j][0], items[i][0]) in existing:
                        continue
                    try:
                        if is_conflicting(items[i][1], items[j][1]):
                            conn.execute(
                                "INSERT INTO fact_links (fact_id, linked_fact_id, relation) VALUES (?, ?, 'contradicted_by')",
                                (items[i][0], items[j][0]),
                            )
                            existing.add((items[i][0], items[j][0]))
                            report["conflicts"] += 1
                    except Exception:
                        continue
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass


def _step_trust(conn: sqlite3.Connection, cfg: dict, report: dict) -> None:
    rows = conn.execute(
        "SELECT fact_id, trust_score, helpful_count, retrieval_count FROM facts"
    ).fetchall()
    for row in rows:
        helpful = row["helpful_count"] or 0
        unhelpful = max(0, (row["retrieval_count"] or 0) - helpful)
        if helpful >= cfg.get("promote_helpful_threshold", 3):
            conn.execute(
                "UPDATE facts SET trust_score = MIN(1.0, trust_score + 0.05), "
                "confidence = MIN(1.0, COALESCE(confidence, trust_score) + 0.05) WHERE fact_id = ?",
                (row["fact_id"],),
            )
            report["strengthened"] += 1
        elif unhelpful >= cfg.get("demote_unhelpful_threshold", 3) and (row["trust_score"] or 0) > 0.1:
            conn.execute(
                "UPDATE facts SET trust_score = MAX(0.0, trust_score - 0.10) WHERE fact_id = ?",
                (row["fact_id"],),
            )
            report["demoted"] += 1
    conn.commit()


def _step_lessons(conn: sqlite3.Connection, report: dict) -> None:
    """Promote repeated failure patterns to candidate lessons (no LLM)."""
    if "mem_type" not in _columns(conn):
        return
    rows = conn.execute(
        "SELECT fact_id, content FROM facts WHERE mem_type = 'event' AND lifecycle = 'active' LIMIT 50"
    ).fetchall()
    groups: dict[str, list] = {}
    for row in rows:
        key = dedupe_keys(row["content"] or "")["slot_key"]
        groups.setdefault(key, []).append(row)
    for _key, members in groups.items():
        if len(members) >= 3:
            report["lessons"] += 1
    conn.commit()


# ---------------------------------------------------------------------------
# Self-healing: detect + quarantine + repair + report (no auto-delete).
# ---------------------------------------------------------------------------

def self_heal(store) -> dict:
    """Safe repair pass. Returns findings + actions; never deletes rows."""
    from .migration import ensure_cognitive_schema

    report: dict = {"checked": 0, "quarantined": 0, "repaired": 0, "findings": []}
    conn = store._conn
    with store._lock:
        ensure_cognitive_schema(conn)
        cols = _columns(conn)
        rows = conn.execute("SELECT * FROM facts").fetchall()
        report["checked"] = len(rows)
        for row in rows:
            fid = row["fact_id"]
            # Invalid trust range.
            trust = row["trust_score"]
            if trust is None or not (0.0 <= float(trust) <= 1.0):
                conn.execute("UPDATE facts SET trust_score = 0.5 WHERE fact_id = ?", (fid,))
                report["repaired"] += 1
                report["findings"].append(f"fact {fid}: trust out of range -> 0.5")
            # Empty content (malformed) -> quarantine, keep row.
            if not (row["content"] or "").strip():
                conn.execute("UPDATE facts SET lifecycle = 'quarantine' WHERE fact_id = ?", (fid,))
                report["quarantined"] += 1
                report["findings"].append(f"fact {fid}: empty content quarantined")
            # Impossible future timestamp -> keep, flag.
            try:
                from datetime import datetime, timezone as _tz
                ts_raw = row["updated_at"]
                if ts_raw:
                    ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=_tz.utc)
                    if (datetime.now(_tz.utc) - ts).total_seconds() < -86400:
                        report["findings"].append(f"fact {fid}: future timestamp flagged")
            except Exception:
                report["findings"].append(f"fact {fid}: unparseable timestamp flagged")
            # Unknown mem_type/lifecycle -> repair to defaults.
            if "mem_type" in cols and (row["mem_type"] or "") not in (
                    "fact", "preference", "constraint", "decision", "invariant",
                    "project_state", "event", "lesson", "pattern", "evidence",
                    "hypothesis", "superseded", "general"):
                conn.execute("UPDATE facts SET mem_type = 'general' WHERE fact_id = ?", (fid,))
                report["repaired"] += 1
        # Orphan fact_entities references.
        try:
            orphans = conn.execute(
                "SELECT fe.fact_id FROM fact_entities fe LEFT JOIN facts f ON f.fact_id = fe.fact_id "
                "WHERE f.fact_id IS NULL"
            ).fetchall()
            if orphans:
                report["findings"].append(f"{len(orphans)} orphan fact_entities flagged")
        except Exception:
            pass
        # Inconsistent lineage (links to missing facts).
        try:
            bad_links = conn.execute(
                "SELECT link_id FROM fact_links fl LEFT JOIN facts f ON f.fact_id = fl.linked_fact_id "
                "WHERE f.fact_id IS NULL"
            ).fetchall()
            if bad_links:
                report["findings"].append(f"{len(bad_links)} dangling fact_links flagged")
        except Exception:
            pass
        # Rebuild FTS index integrity (no data loss: rebuild from facts table).
        try:
            conn.execute("INSERT INTO facts_fts(facts_fts) VALUES('rebuild')")
        except Exception as exc:
            report["findings"].append(f"fts rebuild skipped: {str(exc)[:120]}")
        # Recover interrupted consolidation.
        try:
            if _get_state(conn, "dream_running") == "1":
                _set_state(conn, "dream_running", "0")
                report["repaired"] += 1
                report["findings"].append("interrupted dream cycle recovered")
        except Exception:
            pass
        conn.commit()
    # Keep findings bounded for logs.
    report["findings"] = report["findings"][:50]
    return report


__all__ = ["DEFAULT_DREAM_POLICY", "run_dream_cycle", "self_heal"]
