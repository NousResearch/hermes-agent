#!/usr/bin/env python3
"""Profile review queue — persistent FIFO that drains a backlog of profiles,
producing a performance review AND a skill review for each, then recording
both as ``profile.performance.reviewed`` and ``profile.skill.reviewed``
events in the central activity ledger.

The queue survives restarts (SQLite table). One profile is processed per
worker tick so large backlogs do not starve other system activity.

Usage:
  # One-shot: drain the full backlog, profile by profile, 0.5s apart
  python3 scripts/profile-review-queue.py --drain

  # Worker: process up to N profiles then exit (for cron)
  python3 scripts/profile-review-queue.py --tick --max 5

  # Seed the queue with all profiles (idempotent — skips already-pending)
  python3 scripts/profile-review-queue.py --seed

  # Show queue state
  python3 scripts/profile-review-queue.py --status

The script reads:
  - ``~/.hermes/profiles/<name>/config.yaml`` (skill enable state)
  - ``~/.hermes/governance/profile-activity-ledger.sqlite`` (ledger events)
  - ``~/.hermes/sessions/*.jsonl`` (session summaries)

The script writes:
  - ``profile.performance.reviewed`` events with payload
    {profile, score 0-10, signals{...}, recommendations[], reviewed_by}
  - ``profile.skill.reviewed`` events with payload
    {profile, total_enabled, used, dormant, quarantined,
     recommendations[], reviewed_by}
  - Queue state in ``~/.hermes/governance/review-queue.sqlite``
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes_cli.profile_activity_ledger import append_event

# ── Paths ────────────────────────────────────────────────────────────────────

HERMES_HOME = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
PROFILES_DIR = HERMES_HOME / "profiles"
LEDGER_DB = HERMES_HOME / "governance" / "profile-activity-ledger.sqlite"
SESSIONS_DIR = HERMES_HOME / "sessions"
QUEUE_DB = HERMES_HOME / "governance" / "review-queue.sqlite"
LOGBOARD = HERMES_HOME / "governance" / "logboard"

# ── Queue schema ─────────────────────────────────────────────────────────────

_QUEUE_SCHEMA = """
CREATE TABLE IF NOT EXISTS review_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile TEXT NOT NULL,
    cycle TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    enqueued_at INTEGER NOT NULL,
    started_at INTEGER,
    completed_at INTEGER,
    error TEXT,
    performance_event_id TEXT,
    skill_event_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_rq_status ON review_queue(status, enqueued_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_rq_profile_cycle
    ON review_queue(profile, cycle, status);
"""


def _queue_conn() -> sqlite3.Connection:
    QUEUE_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(QUEUE_DB)
    con.executescript(_QUEUE_SCHEMA)
    return con


# ── Profile discovery ────────────────────────────────────────────────────────

def _all_profiles() -> list[str]:
    names: list[str] = []
    if (HERMES_HOME / "config.yaml").exists():
        names.append("default")
    if PROFILES_DIR.exists():
        for d in sorted(PROFILES_DIR.iterdir()):
            if d.is_dir() and (d / "config.yaml").exists():
                names.append(d.name)
    return names


# ── Data gathering for performance review ────────────────────────────────────

def _ledger_query(sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    if not LEDGER_DB.exists():
        return []
    try:
        con = sqlite3.connect(f"file:{LEDGER_DB}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        rows = list(con.execute(sql, params))
        con.close()
        return rows
    except Exception:
        return []


def _ledger_count(event_type: str, profile: str, days: int = 7) -> int:
    since = int(time.time()) - days * 86400
    rows = _ledger_query(
        "SELECT COUNT(*) AS c FROM activity_events "
        "WHERE event_type=? AND (actor_profile=? OR target_profile=?) "
        "AND occurred_at >= ?",
        (event_type, profile, profile, since),
    )
    return rows[0]["c"] if rows else 0


def _session_count(profile: str, days: int = 7) -> int:
    """Count session JSONL files whose first line is a meta record with this profile."""
    if not SESSIONS_DIR.exists():
        return 0
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
    count = 0
    # Only count files that look like session dumps and were modified recently
    for f in SESSIONS_DIR.glob("*.jsonl"):
        try:
            mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
        except OSError:
            continue
        if mtime < cutoff:
            continue
        # Try first non-empty line
        try:
            with f.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        break
                    meta = rec.get("metadata") or rec.get("meta") or {}
                    p = (
                        meta.get("profile")
                        or meta.get("actor_profile")
                        or rec.get("profile")
                    )
                    if p == profile:
                        count += 1
                    break
        except OSError:
            continue
    return count


def _profile_files_mtime(profile: str) -> dict[str, dict]:
    home = HERMES_HOME if profile == "default" else PROFILES_DIR / profile
    out = {}
    for fn in ("config.yaml", "SOUL.md", "USER.md"):
        p = home / fn
        if p.exists():
            try:
                st = p.stat()
                out[fn] = {
                    "mtime_epoch": int(st.st_mtime),
                    "age_days": (time.time() - st.st_mtime) / 86400,
                    "size": st.st_size,
                }
            except OSError:
                out[fn] = {"error": "stat failed"}
        else:
            out[fn] = {"exists": False}
    return out


def _enabled_skills(profile: str) -> list[str]:
    import yaml as _pyyaml

    cfg_path = HERMES_HOME / "config.yaml" if profile == "default" else PROFILES_DIR / profile / "config.yaml"
    if not cfg_path.exists():
        return []
    try:
        data = _pyyaml.safe_load(cfg_path.read_text()) or {}
    except Exception:
        return []
    skills = data.get("skills") or {}
    always = list(skills.get("always_skills") or [])
    enabled = list(skills.get("enabled_skills") or [])
    return sorted(set(always) | set(enabled))


def _skill_borrows_for_profile(profile: str, days: int = 30) -> dict[str, int]:
    since = int(time.time()) - days * 86400
    rows = _ledger_query(
        "SELECT object_id, COUNT(*) AS c FROM activity_events "
        "WHERE event_type='skill.borrowed' "
        "AND (actor_profile=? OR target_profile=?) "
        "AND occurred_at >= ? GROUP BY object_id",
        (profile, profile, since),
    )
    return {r["object_id"]: r["c"] for r in rows if r["object_id"]}


def _skill_loads_for_profile(profile: str, days: int = 30) -> dict[str, int]:
    since = int(time.time()) - days * 86400
    rows = _ledger_query(
        "SELECT object_id, COUNT(*) AS c FROM activity_events "
        "WHERE event_type='skill.loaded' "
        "AND (actor_profile=? OR target_profile=?) "
        "AND occurred_at >= ? GROUP BY object_id",
        (profile, profile, since),
    )
    return {r["object_id"]: r["c"] for r in rows if r["object_id"]}


# ── Reviewers ─────────────────────────────────────────────────────────────────

REVIEWER_PROFILE = "denji-reviewer"


def _build_performance_review(profile: str) -> dict[str, Any]:
    """Synthesise a performance review from session + ledger + file signals."""
    sessions_7d = _session_count(profile, days=7)
    sessions_30d = _session_count(profile, days=30)
    ledger_7d = _ledger_count("profile.review.weekly", profile, days=7)
    skill_borrows_7d = _ledger_count("skill.borrowed", profile, days=7)
    errors_7d = _ledger_count("error", profile, days=7)
    denials_7d = _ledger_count("skill.quarantined", profile, days=7)
    autos_7d = _ledger_count("skill.enabled_auto", profile, days=7)
    files = _profile_files_mtime(profile)

    # Score 0..10 — composite of activity, freshness, and error rate
    activity = min(10.0, (sessions_7d * 1.5 + skill_borrows_7d * 0.3))
    freshness = 0.0
    for fn, st in files.items():
        if isinstance(st, dict) and "age_days" in st:
            age = st["age_days"]
            if age < 7:
                freshness += 3.0
            elif age < 30:
                freshness += 2.0
            elif age < 90:
                freshness += 1.0
    freshness = min(freshness, 5.0)
    error_penalty = min(errors_7d * 0.5 + denials_7d * 0.3, 4.0)
    auto_bonus = min(autos_7d * 0.3, 1.0)
    score = round(max(0.0, min(10.0, activity + freshness + auto_bonus - error_penalty)), 1)

    signals: dict[str, Any] = {
        "sessions_7d": sessions_7d,
        "sessions_30d": sessions_30d,
        "ledger_events_7d": ledger_7d,
        "skill_borrows_7d": skill_borrows_7d,
        "errors_7d": errors_7d,
        "quarantine_denials_7d": denials_7d,
        "auto_promotions_7d": autos_7d,
    }
    file_ages: dict[str, float | None] = {
        fn: st.get("age_days") if isinstance(st, dict) else None
        for fn, st in files.items()
    }
    signals["file_ages_days"] = file_ages

    recommendations: list[str] = []
    if sessions_7d == 0:
        recommendations.append("No activity in last 7d — consider sunset or reactivation review")
    if errors_7d > 3:
        recommendations.append(f"{errors_7d} errors in 7d — investigate error rate")
    cfg_age = file_ages.get("config.yaml")
    if cfg_age is not None and cfg_age > 90:
        recommendations.append("config.yaml stale (>90d) — refresh to keep skills current")
    if autos_7d == 0 and skill_borrows_7d > 5:
        recommendations.append("Heavy borrow but no auto-promotions — review enablement threshold")
    if not recommendations:
        recommendations.append("Healthy — no action required")

    return {
        "profile": profile,
        "score": score,
        "signals": signals,
        "recommendations": recommendations,
        "reviewed_by": REVIEWER_PROFILE,
    }


def _build_skill_review(profile: str) -> dict[str, Any]:
    """Synthesise a skill review: enabled vs used vs dormant vs quarantined."""
    enabled = _enabled_skills(profile)
    borrows = _skill_borrows_for_profile(profile, days=30)
    loads = _skill_loads_for_profile(profile, days=30)
    used = set(borrows.keys()) | set(loads.keys())
    enabled_set = set(enabled)
    dormant = sorted(enabled_set - used)
    # Quarantined = in ledger as quarantined for this profile
    quarantine_rows = _ledger_query(
        "SELECT DISTINCT object_id FROM activity_events "
        "WHERE event_type='skill.quarantined' AND object_id IS NOT NULL"
    )
    quarantined_global = {r["object_id"] for r in quarantine_rows}
    quarantined_for_profile = sorted(quarantined_global & enabled_set)

    coverage = round((len(used & enabled_set) / max(1, len(enabled_set))) * 100, 1)

    recommendations: list[str] = []
    if dormant:
        recommendations.append(
            f"{len(dormant)} dormant skills enabled but unused in 30d: "
            + ", ".join(dormant[:5])
            + ("..." if len(dormant) > 5 else "")
        )
    if quarantined_for_profile:
        recommendations.append(
            f"{len(quarantined_for_profile)} quarantined skills still in enable list: "
            + ", ".join(quarantined_for_profile)
        )
    if not enabled:
        recommendations.append("No skills enabled — check skills.always_skills / enabled_skills")
    if coverage < 50 and enabled:
        recommendations.append(f"Low skill coverage ({coverage}%) — consider re-enabling used skills")
    if not recommendations:
        recommendations.append("Skill set is healthy and well-utilised")

    return {
        "profile": profile,
        "total_enabled": len(enabled),
        "used": len(used & enabled_set),
        "dormant": len(dormant),
        "dormant_list": dormant[:10],
        "quarantined": len(quarantined_for_profile),
        "quarantined_list": quarantined_for_profile,
        "coverage_pct": coverage,
        "recommendations": recommendations,
        "reviewed_by": REVIEWER_PROFILE,
    }


# ── Queue ops ─────────────────────────────────────────────────────────────────

def _seed_queue(cycle: str = "weekly") -> int:
    """Add all profiles to the queue if not already pending for this cycle. Returns count seeded."""
    con = _queue_conn()
    cur = con.cursor()
    seeded = 0
    for p in _all_profiles():
        try:
            cur.execute(
                "INSERT INTO review_queue (profile, cycle, status, enqueued_at) "
                "VALUES (?, ?, 'pending', ?)",
                (p, cycle, int(time.time())),
            )
            seeded += 1
        except sqlite3.IntegrityError:
            # Already pending for this cycle
            pass
    con.commit()
    con.close()
    return seeded


def _next_pending() -> sqlite3.Row | None:
    con = _queue_conn()
    con.row_factory = sqlite3.Row
    row = con.execute(
        "SELECT * FROM review_queue WHERE status='pending' ORDER BY enqueued_at LIMIT 1"
    ).fetchone()
    con.close()
    return row


def _mark_in_progress(queue_id: int) -> None:
    con = _queue_conn()
    con.execute(
        "UPDATE review_queue SET status='in_progress', started_at=? WHERE id=?",
        (int(time.time()), queue_id),
    )
    con.commit()
    con.close()


def _mark_done(queue_id: int, perf_eid: str, skill_eid: str) -> None:
    con = _queue_conn()
    con.execute(
        "UPDATE review_queue SET status='done', completed_at=?, "
        "performance_event_id=?, skill_event_id=? WHERE id=?",
        (int(time.time()), perf_eid, skill_eid, queue_id),
    )
    con.commit()
    con.close()


def _mark_failed(queue_id: int, err: str) -> None:
    con = _queue_conn()
    con.execute(
        "UPDATE review_queue SET status='failed', completed_at=?, error=? WHERE id=?",
        (int(time.time()), err[:500], queue_id),
    )
    con.commit()
    con.close()


def _queue_status() -> dict[str, int]:
    con = _queue_conn()
    counts = {"pending": 0, "in_progress": 0, "done": 0, "failed": 0, "total": 0}
    for row in con.execute(
        "SELECT status, COUNT(*) AS c FROM review_queue GROUP BY status"
    ):
        counts[row[0]] = row[1]
        counts["total"] += row[1]
    con.close()
    return counts


# ── Worker tick ───────────────────────────────────────────────────────────────

def _process_one() -> dict[str, Any] | None:
    row = _next_pending()
    if not row:
        return None
    profile = row["profile"]
    queue_id = row["id"]
    _mark_in_progress(queue_id)
    try:
        perf = _build_performance_review(profile)
        skill = _build_skill_review(profile)
        perf_eid = append_event(
            source="review-queue",
            event_type="profile.performance.reviewed",
            actor_profile=REVIEWER_PROFILE,
            target_profile=profile,
            object_type="profile",
            object_id=profile,
            summary=f"Performance review: score {perf['score']}/10",
            payload=perf,
        )
        skill_eid = append_event(
            source="review-queue",
            event_type="profile.skill.reviewed",
            actor_profile=REVIEWER_PROFILE,
            target_profile=profile,
            object_type="profile",
            object_id=profile,
            summary=(
                f"Skill review: {skill['used']}/{skill['total_enabled']} used, "
                f"{skill['dormant']} dormant, {skill['quarantined']} quarantined"
            ),
            payload=skill,
        )
        _mark_done(queue_id, perf_eid, skill_eid)
        return {"profile": profile, "perf_eid": perf_eid, "skill_eid": skill_eid, "score": perf["score"]}
    except Exception as e:
        _mark_failed(queue_id, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-300:]}")
        return {"profile": profile, "error": str(e)}


# ── CLI ───────────────────────────────────────────────────────────────────────

def _print_status() -> None:
    s = _queue_status()
    pending = s["pending"] + s["in_progress"]
    print(f"Queue state (in {QUEUE_DB}):")
    print(f"  pending:     {s['pending']}")
    print(f"  in_progress: {s['in_progress']}")
    print(f"  done:        {s['done']}")
    print(f"  failed:      {s['failed']}")
    print(f"  total:       {s['total']}")
    if pending:
        eta_min = pending * 0.5 / 60  # 0.5s per profile
        print(f"  ETA:         ~{eta_min:.1f} min ({pending} × 0.5s)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Profile review queue")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--seed", action="store_true", help="Seed the queue with all profiles")
    g.add_argument("--drain", action="store_true", help="Drain the queue, one profile per 0.5s")
    g.add_argument("--tick", action="store_true", help="Process up to --max profiles then exit")
    g.add_argument("--status", action="store_true", help="Show queue state")
    p.add_argument("--max", type=int, default=5, help="Max profiles to process in --tick mode")
    p.add_argument("--cycle", default="weekly", help="Cycle label (weekly/monthly/quarterly)")
    p.add_argument("--delay", type=float, default=0.5, help="Seconds between profiles in --drain")
    args = p.parse_args(argv)

    if args.status:
        _print_status()
        return 0

    if args.seed:
        n = _seed_queue(args.cycle)
        print(f"Seeded {n} new profiles for cycle={args.cycle}")
        _print_status()
        return 0

    if args.tick:
        processed = 0
        for _ in range(args.max):
            r = _process_one()
            if not r:
                break
            processed += 1
            if "error" in r:
                print(f"  ✗ {r['profile']}: {r['error']}")
            else:
                print(f"  ✓ {r['profile']}: score={r['score']}")
        print(f"Processed {processed} profiles")
        return 0

    if args.drain:
        s_before = _queue_status()
        print(f"Draining {s_before['pending']} pending profiles (delay {args.delay}s)...")
        processed = 0
        failed = 0
        start = time.time()
        while True:
            r = _process_one()
            if not r:
                break
            processed += 1
            if "error" in r:
                failed += 1
                print(f"  ✗ {r['profile']}: {r['error'][:80]}")
            else:
                print(f"  ✓ {r['profile']}: score={r['score']}")
            time.sleep(args.delay)
        elapsed = time.time() - start
        print(f"\nDone: {processed} processed ({failed} failed) in {elapsed:.1f}s")
        _print_status()
        return 0 if failed == 0 else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
