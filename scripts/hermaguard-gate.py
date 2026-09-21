#!/usr/bin/env python3
"""
Hermaguard Gate — monitors tasks that require Gate 0 adversarial review.

Runs every 2 hours (08:00-20:00) as a no_agent cron. Scans kanban boards
for backend/frontend/security/new-feature tasks that reached review status
without Hermaguard evidence. Reports findings to #governance.

Silent when nothing found (cron output contract).
"""

import json
import sqlite3
import sys
import os
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

HERMES_HOME = Path(os.environ.get("HERMES_HOME", "/home/kensei/.hermes"))
# W1-G (Batch 1): board DB identities resolved via _board_compat so retired
# slugs (default->core, ops->security-ops, content-lead->content) map to the
# current canonical DB path. Semantic board labels (keys) are preserved.
import _board_compat
BOARDS = _board_compat.build_board_db_map([
    "default", "apps", "content-lead", "ops", "research",
])

OUT_DIR = HERMES_HOME / "governance" / "logboard"
# OUT_DIR.mkdir() is deferred until immediately before the live log write
# (see main()). Importing the module must not create directories — tests
# import the module under a temp HERMES_HOME and assert no side effects.

TZ = timezone(timedelta(hours=1))
now = datetime.now(TZ)
WINDOW_HOURS = 3  # Look back 3 hours for review transitions


# --- Tasks that Gate 0 applies to ---
TIER_KEYWORDS = {
    "backend": ["backend", "api", "db", "database", "migration", "server", "model", "logic"],
    "frontend": ["frontend", "ui", "ux", "component", "style", "layout", "mobile", "react", "screen"],
    "security": ["auth", "credential", "token", "key", "password", "security", "encrypt", "jwt"],
    "new-feature": ["feature", "integration", "new", "add", "implement"],
}


def _kw_pattern(keywords) -> re.Pattern | None:
    """Compile keywords into one case-insensitive regex with word boundaries.

    Short tokens like 'ui' must NOT match inside ordinary words
    ('requirements', 'guidelines', 'repository') — that produced walls of
    false 'frontend' flags on legitimately completed tasks (2026-08-11).
    Multi-word phrases are matched as literal substrings.
    """
    parts = []
    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue
        if " " in kw:
            parts.append(re.escape(kw))
        else:
            parts.append(rf"\b{re.escape(kw)}\b")
    if not parts:
        return None
    return re.compile("|".join(parts), re.IGNORECASE)


SKIP_KEYWORDS = ["content", "post", "copy", "draft", "article", "config", "cron edit", "skill activation"]
SKIP_PATTERN = _kw_pattern(SKIP_KEYWORDS)
TIER_PATTERNS = {tier: _kw_pattern(kws) for tier, kws in TIER_KEYWORDS.items()}


def _should_require_gate0(title: str, body: str) -> str | None:
    """Return tier name if this task should have Gate 0, else None."""
    combined = (title or "") + " " + (body or "")

    # Skip content/config/infra outright
    if SKIP_PATTERN and SKIP_PATTERN.search(combined):
        return None

    for tier, pat in TIER_PATTERNS.items():
        if pat and pat.search(combined):
            return tier
    return None


def _scan_boards():
    tasks = []
    cutoff = int((now - timedelta(hours=WINDOW_HOURS)).timestamp())

    for slug, db_path in BOARDS.items():
        if not db_path.is_file():
            continue
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row

            rows = conn.execute(
                """
                SELECT id, title, body, assignee, status, tier, pipeline_stage,
                       created_at, completed_at,
                       COALESCE(completed_at, started_at, created_at) AS updated_at
                FROM tasks
                WHERE status IN ('review', 'running')
                  AND COALESCE(completed_at, started_at, created_at) >= ?
                ORDER BY COALESCE(completed_at, started_at, created_at) DESC
                """,
                (cutoff,),
            ).fetchall()

            for r in rows:
                d = dict(r)
                d["_board_slug"] = slug
                d["_board_db"] = db_path
                tasks.append(d)
            conn.close()
        except sqlite3.OperationalError as e:
            # Schema mismatch on a freshly auto-recreated empty board.
            # Skip, don't kill the whole scan.
            print(f"  [skip] {slug}: {e}", file=sys.stderr)

    return tasks


def _has_hermaguard_evidence(task_id: str, db_path: Path) -> bool:
    """Check if task_events or task_comments has hermaguard/adversarial review entry."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Check title/body for hermaguard mention in events payload
    rows = conn.execute(
        """
        SELECT kind, payload FROM task_events
        WHERE task_id = ?
          AND (
            LOWER(kind) LIKE '%hermaguard%'
            OR LOWER(kind) LIKE '%adversarial%'
            OR LOWER(kind) LIKE '%review%'
          )
        LIMIT 5
        """,
        (task_id,),
    ).fetchall()

    # Gate 0 evidence often lives in a review comment (2026-08-11: 4 tasks
    # had hermaguard/adversarial evidence in task_comments that the gate
    # missed — fixed by checking comments too). Guard the table: some
    # boards/schemas don't carry task_comments (e.g. test fixtures).
    comments = []
    has_comments_table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='task_comments'"
    ).fetchone()
    if has_comments_table:
        comments = conn.execute(
            """
            SELECT body FROM task_comments
            WHERE task_id = ?
              AND (
                LOWER(body) LIKE '%hermaguard%'
                OR LOWER(body) LIKE '%adversarial%'
                OR LOWER(body) LIKE '%gate 0%'
                OR LOWER(body) LIKE '%gate0%'
              )
            LIMIT 5
            """,
            (task_id,),
        ).fetchall()

    conn.close()

    # Also check if result field contains hermaguard mention
    conn = sqlite3.connect(str(db_path))
    result = conn.execute(
        "SELECT result FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    conn.close()

    if result and result[0]:
        text = str(result[0]).lower()
        if "hermaguard" in text or "adversarial" in text:
            return True

    for r in rows:
        payload = (r["payload"] or "").lower()
        if "hermaguard" in payload or "adversarial" in payload:
            return True

    for c in comments:
        text = (c["body"] or "").lower()
        if "hermaguard" in text or "adversarial" in text or "gate 0" in text or "gate0" in text:
            return True

    return False


def _parse_args(argv: list[str] | None = None):
    import argparse
    parser = argparse.ArgumentParser(description="Hermaguard Gate 0 monitor")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report flagged task IDs to stdout without writing the audit logboard.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    tasks = _scan_boards()
    flagged = []
    skipped_tier = []
    compliant = []

    for task in tasks:
        tier_tag = _should_require_gate0(task["title"], task["body"])
        if not tier_tag:
            skipped_tier.append(task)
            continue

        has_evidence = _has_hermaguard_evidence(task["id"], task["_board_db"])

        if has_evidence:
            compliant.append({"task": task, "tier_tag": tier_tag})
        else:
            flagged.append({"task": task, "tier_tag": tier_tag})

    # Live mode: always log to logboard for audit. Dry-run: never create files.
    if not args.dry_run:
        log_entry = {
            "gate_name": "hermaguard-gate",
            "timestamp": now.isoformat(),
            "window_hours": WINDOW_HOURS,
            "scanned": len(tasks),
            "compliant": len(compliant),
            "flagged": len(flagged),
            "skipped_tier": len(skipped_tier),
            "flagged_tasks": [
                {
                    "id": f["task"]["id"],
                    "board": f["task"]["_board_slug"],
                    "title": f["task"]["title"][:80],
                    "status": f["task"]["status"],
                    "tier_tag": f["tier_tag"],
                    "updated_at": datetime.fromtimestamp(
                        f["task"]["updated_at"], tz=timezone(timedelta(hours=1))
                    ).isoformat(),
                }
                for f in flagged
            ],
            "compliant_tasks": [
                {
                    "id": c["task"]["id"],
                    "board": c["task"]["_board_slug"],
                    "title": c["task"]["title"][:80],
                    "status": c["task"]["status"],
                    "tier_tag": c["tier_tag"],
                }
                for c in compliant[:10]  # Cap for brevity
            ],
        }
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        logfile = OUT_DIR / f"hermaguard-gate-{now.strftime('%Y%m%d-%H%M%S')}.json"
        logfile.write_text(json.dumps(log_entry, indent=2, default=str))

    # Silent when nothing flagged
    if not flagged:
        return 0

    # Deliver report. Only reached when tasks are missing Gate 0, so this is
    # always an action item (silent return above when all compliant).
    print(f"🔴 **Gate 0 (Adversarial Review) Compliance Report**")
    print(f"Window: last {WINDOW_HOURS}h  |  {now.strftime('%d/%m/%Y %H:%M:%S')}")
    print()
    print(f"Scanned: {len(tasks)} tasks  |  Compliant: {len(compliant)}  |  **Missing Gate 0: {len(flagged)}**")
    print()

    for f in flagged:
        t = f["task"]
        tag = f["tier_tag"]
        print(
            f"- `{t['id'][:12]}` ({t['_board_slug']}) [{tag}] `{t['status']}` — "
            f"{t['title'][:70]}"
        )

    print()
    print("**Action:** If in review, run `/hermaguard` before approving. "
          "If already done, log to self-eval and tighten task routing.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
