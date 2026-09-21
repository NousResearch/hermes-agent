#!/usr/bin/env python3
"""
Skill Reroute — Auto-reroutes blocked kanban tasks when a profile lacks
required forced skills.

Scans all boards for tasks blocked with forced_skill_rejected events,
finds a profile that has the missing skill, and reassigns.

Usage:
  python3 skill-reroute.py

Output mode:
  [SILENT] if no tasks needed rerouting
  Reports each reroute if action was taken
"""

import sqlite3
import glob
import json
import os
import re
import sys
import time

# Cross-process write lock — prevents WAL checkpoint races
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from kanban_write_lock import write_lock
except ImportError:
    write_lock = None

# ─── Config ───────────────────────────────────────────────────────────────

HERMES_HOME = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
KANBAN_HOME = os.path.join(HERMES_HOME, "kanban")
PROFILES_HOME = os.path.join(HERMES_HOME, "profiles")

# Preferred assignee per keyword match in missing skill names
DOMAIN_LEADS = {
    "content": "ceecee",
    "social": "ceecee",
    "postiz": "ceecee",
    "brand": "ceecee",
    "code": "octacon",
    "github": "octacon",
    "react": "octacon",
    "debug": "octacon",
    "test": "octacon",
    "research": "remii",
    "arxiv": "remii",
    "blog": "remii",
    "kanban": "wesker",
    "cron": "wesker",
    "infra": "wesker",
    "gateway": "wesker",
    "ops": "wesker",
    "security": "wesker",
    "health": "wesker",
}

DEFAULT_ESCALATION = "kensei-review"

# P13 isolation: when --dry-run is passed, every write path (task UPDATE,
# task_events INSERT) is suppressed. Read paths (board scan, skill map
# build, classification) run unchanged so the proposed reroutes are
# still computed and printed.
_DRY_RUN = False

# ─── Helpers ──────────────────────────────────────────────────────────────

def get_board_paths():
    """Return dict of board_label -> kanban.db path."""
    boards = {}
    default_db = os.path.join(HERMES_HOME, "kanban.db")
    if os.path.exists(default_db):
        boards["default"] = default_db
    board_dir = os.path.join(HERMES_HOME, "kanban/boards")
    if os.path.exists(board_dir):
        for p in sorted(glob.glob(os.path.join(board_dir, "*", "kanban.db"))):
            label = p.split("/")[-2]
            boards[label] = p
    return boards


def build_profile_skill_map():
    """
    Build a dict: skill_name -> [profile1, profile2, ...]
    Only checks profiles that have the skill directory present.
    """
    skill_map = {}
    if not os.path.exists(PROFILES_HOME):
        return skill_map

    for pd in sorted(glob.glob(os.path.join(PROFILES_HOME, "*/"))):
        profile = pd.rstrip("/").split("/")[-1]
        skills_path = os.path.join(pd, "skills")
        if not os.path.exists(skills_path):
            continue
        for cat_dir in glob.glob(os.path.join(skills_path, "*")):
            for skill_dir in glob.glob(os.path.join(cat_dir, "*")):
                name = skill_dir.rstrip("/").split("/")[-1]
                skill_map.setdefault(name, []).append(profile)

    return skill_map


def find_best_assignee(missing_skills, skill_map):
    """
    Find the best profile that has ALL missing skills.
    Prefers domain leads, falls back to kensei-review.
    """
    if not missing_skills:
        return None

    # Candidates: profiles that have ALL missing skills
    candidates = None
    for ms in missing_skills:
        profile_list = skill_map.get(ms, [])
        if candidates is None:
            candidates = set(profile_list)
        else:
            candidates &= set(profile_list)

    if not candidates:
        # Partial match: find profiles that have at least one
        partial = {}
        for ms in missing_skills:
            for p in skill_map.get(ms, []):
                partial[p] = partial.get(p, 0) + 1

        # Sort by most missing skills covered
        sorted_partial = sorted(partial.items(), key=lambda x: -x[1])
        if sorted_partial:
            candidates = {p for p, _ in sorted_partial}
        else:
            return DEFAULT_ESCALATION

    # Score candidates by domain lead preference
    def score(profile):
        s = 0
        for skill_name in missing_skills:
            for keyword, lead in DOMAIN_LEADS.items():
                if keyword in skill_name.lower() and lead == profile:
                    s += 10
        return s

    ranked = sorted(candidates, key=lambda p: (-score(p), p))
    return ranked[0] if ranked else DEFAULT_ESCALATION


def scan_and_reroute(db_path, board_label, skill_map):
    """
    Scan a single board for forced_skill_rejected tasks. Reroute any found.
    Returns (rerouted_count, skipped_count).
    """
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
    except Exception:
        return 0, 0

    now = int(time.time())
    rerouted = 0
    skipped = 0

    # Find tasks that are blocked AND have a forced_skill_rejected event
    # (the most recent per task)
    cur.execute("""
        SELECT t.id, t.title, t.assignee, t.skills,
               e.payload as reject_payload
        FROM tasks t
        JOIN task_events e ON e.task_id = t.id
        WHERE t.status = 'blocked'
          AND e.kind = 'forced_skill_rejected'
          AND e.id = (
              SELECT MAX(e2.id) FROM task_events e2
              WHERE e2.task_id = t.id AND e2.kind = 'forced_skill_rejected'
          )
    """)

    rows = cur.fetchall()

    for row in rows:
        tid = row["id"]
        current_assignee = row["assignee"]
        skills_json = row["skills"]

        try:
            payload = json.loads(row["reject_payload"])
        except (json.JSONDecodeError, TypeError):
            skipped += 1
            continue

        missing_skills = payload.get("missing_skills", [])
        if not missing_skills:
            skipped += 1
            continue

        # Find the new best assignee
        new_assignee = find_best_assignee(missing_skills, skill_map)

        if not new_assignee or new_assignee == current_assignee:
            skipped += 1
            continue

        # Reassign: update task + create events
        old_assignee = current_assignee or "(unassigned)"
        reason = f"Auto-rerouted from {old_assignee} to {new_assignee} — missing forced skill(s): {', '.join(missing_skills)}"

        if _DRY_RUN:
            rerouted += 1
            print(f"  ⊘ DRY-RUN REROUTE: {tid} '{row['title'][:50].strip()}'")
            print(f"    From: {old_assignee} → To: {new_assignee}")
            print(f"    Missing skills: {', '.join(missing_skills)}")
            continue

        if write_lock is not None:
            with write_lock(conn):
                cur.execute(
                    """UPDATE tasks
                       SET assignee = ?,
                           status = 'ready',
                           current_run_id = NULL,
                           claim_lock = NULL,
                           claim_expires = NULL,
                           started_at = NULL,
                           consecutive_failures = 0
                       WHERE id = ?""",
                    (new_assignee, tid),
                )
                # Event: assigned
                cur.execute(
                    "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?, 'assigned', ?, ?)",
                    (tid, json.dumps({"assignee": new_assignee, "reason": reason}), now),
                )
                # Event: unblocked
                cur.execute(
                    "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?, 'unblocked', ?, ?)",
                    (tid, json.dumps({"reason": reason}), now),
                )
                conn.commit()
        else:
            cur.execute(
                """UPDATE tasks
                   SET assignee = ?,
                       status = 'ready',
                       current_run_id = NULL,
                       claim_lock = NULL,
                       claim_expires = NULL,
                       started_at = NULL,
                       consecutive_failures = 0
                   WHERE id = ?""",
                (new_assignee, tid),
            )
            # Event: assigned
            cur.execute(
                "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?, 'assigned', ?, ?)",
                (tid, json.dumps({"assignee": new_assignee, "reason": reason}), now),
            )
            # Event: unblocked
            cur.execute(
                "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?, 'unblocked', ?, ?)",
                (tid, json.dumps({"reason": reason}), now),
            )
            conn.commit()

        rerouted += 1
        print(f"  ⊘ REROUTED: {tid} '{row['title'][:50].strip()}'")
        print(f"    From: {old_assignee} → To: {new_assignee}")
        print(f"    Missing skills: {', '.join(missing_skills)}")

    conn.close()
    return rerouted, skipped


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    global _DRY_RUN
    if "--dry-run" in sys.argv:
        _DRY_RUN = True
    boards = get_board_paths()
    if not boards:
        return  # silent — no kanban boards

    print(f"Scanning {len(boards)} board(s) for forced-skill blocks...", flush=True)

    skill_map = build_profile_skill_map()
    print(f"Built skill map: {sum(len(v) for v in skill_map.values())} profile-skill pairs", flush=True)

    total_rerouted = 0
    total_skipped = 0

    for label, db_path in sorted(boards.items()):
        rerouted, skipped = scan_and_reroute(db_path, label, skill_map)
        total_rerouted += rerouted
        total_skipped += skipped

    if total_rerouted == 0:
        # Check for stale forced_skill_rejected tasks that were manually moved
        # but still have unresolved reject events
        return  # silent — no forced-skill blocks
    else:
        print(f"\nDone: {total_rerouted} rerouted, {total_skipped} skipped")


if __name__ == "__main__":
    main()
