#!/usr/bin/env python3
"""Kanban v2 auto-archive cron.

Scans all board DBs for tasks in 'done' status where done_at is older
than 14 days. Archives them and logs an 'auto_archived' event.

Output: short text summary + HTML report as MEDIA attachment for Discord.

Usage:
    python3 kanban-auto-archive.py            # live run
    python3 kanban-auto-archive.py --dry-run  # preview only, no changes
"""

import argparse
import glob
import html as html_mod
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ARCHIVE_AFTER_DAYS = 14

# Output directory for HTML report
# P13 isolation: HERMES_HOME parameterises the kanban root.
_HERMES_HOME = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
REPORT_DIR = Path(os.path.join(_HERMES_HOME, "kanban/reports"))


def find_all_board_dbs() -> list[tuple[str, str]]:
    dbs = [("default", os.path.join(_HERMES_HOME, "kanban.db"))]
    boards_dir = os.path.join(_HERMES_HOME, "kanban/boards")
    if os.path.isdir(boards_dir):
        for child in sorted(os.listdir(boards_dir)):
            child_path = os.path.join(boards_dir, child, "kanban.db")
            if os.path.exists(child_path):
                dbs.append((child, child_path))
    return dbs


def archive_stale_tasks(board: str, db_path: str, dry_run: bool) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cols = {row["name"] for row in cur.execute("PRAGMA table_info(tasks)")}
    if "done_at" not in cols:
        conn.close()
        return []

    cutoff = int(time.time()) - (ARCHIVE_AFTER_DAYS * 86400)

    candidates = cur.execute(
        "SELECT id, title, assignee, done_at, status "
        "FROM tasks WHERE status = 'done' AND done_at IS NOT NULL AND done_at < ? "
        "ORDER BY done_at ASC",
        (cutoff,),
    ).fetchall()

    results = []
    now = int(time.time())

    for row in candidates:
        results.append({
            "board": board,
            "task_id": row["id"],
            "title": row["title"][:80] if row["title"] else "",
            "assignee": row["assignee"] or "unassigned",
            "done_at": row["done_at"],
            "age_days": (now - row["done_at"]) // 86400,
        })

    if not dry_run and results:
        for row in candidates:
            conn.execute(
                "UPDATE tasks SET status = 'archived', archived_at = ? WHERE id = ?",
                (now, row["id"]),
            )
            conn.execute(
                "INSERT INTO task_events (task_id, kind, payload, created_at) "
                "VALUES (?, 'auto_archived', ?, ?)",
                (row["id"], json.dumps({"reason": "auto-archive after 14 days in done"}), now),
            )
        conn.commit()

    conn.close()
    return results


def render_html(results: list[dict], dry_run: bool) -> str:
    uk_tz = timezone(timedelta(hours=1))
    now_str = datetime.now(uk_tz).strftime("%d/%m/%Y %H:%M")
    date_label = datetime.now(uk_tz).strftime("%A, %B %-d, %Y")
    mode = "DRY RUN" if dry_run else "ARCHIVED"

    by_board = {}
    for r in results:
        by_board.setdefault(r["board"], []).append(r)

    board_sections = ""
    for board in sorted(by_board.keys()):
        items = by_board[board]
        rows = ""
        for item in items:
            rows += (
                f'<tr><td class="mono">{html_mod.escape(item["task_id"])}</td>'
                f'<td>{html_mod.escape(item["title"])}</td>'
                f'<td class="muted">{html_mod.escape(item["assignee"])}</td>'
                f'<td class="muted">{item["age_days"]}d</td></tr>\n'
            )
        board_sections += f'<h2>{html_mod.escape(board)} ({len(items)} tasks)</h2>\n'
        board_sections += '<table class="data"><tr><th>ID</th><th>Title</th><th>Assignee</th><th>Age</th></tr>\n'
        board_sections += rows
        board_sections += '</table>\n'

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kanban Auto-Archive — {html_mod.escape(now_str)}</title>
<style>
  :root {{ color-scheme: dark; --bg: #11100f; --card: #1c1a18; --muted: #a8a29e; --text: #f5f5f4; --accent: #fbbf24; --line: #34302c; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 24px; background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-size: 14px; line-height: 1.6; }}
  .header {{ border-bottom: 1px solid var(--line); padding-bottom: 16px; margin-bottom: 24px; }}
  .header h1 {{ font-size: 20px; margin: 0 0 4px 0; }}
  .header .meta {{ color: var(--muted); font-size: 13px; }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 4px; font-size: 12px; font-weight: 600; }}
  .badge.dry {{ background: #422006; color: var(--accent); }}
  .badge.live {{ background: #052e16; color: #22c55e; }}
  h2 {{ font-size: 16px; margin: 24px 0 8px 0; color: var(--accent); }}
  table.data {{ width: 100%; border-collapse: collapse; margin-bottom: 16px; }}
  table.data th {{ text-align: left; padding: 6px 12px; border-bottom: 1px solid var(--line); color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
  table.data td {{ padding: 6px 12px; border-bottom: 1px solid var(--line); }}
  .mono {{ font-family: 'SF Mono', 'Fira Code', monospace; font-size: 12px; color: var(--accent); }}
  .muted {{ color: var(--muted); }}
  .summary {{ font-size: 16px; margin: 16px 0; }}
  .summary strong {{ color: var(--accent); }}
</style>
</head>
<body>
<div class="header">
  <h1>Kanban Auto-Archive <span class="badge {'dry' if dry_run else 'live'}">{mode}</span></h1>
  <div class="meta">{html_mod.escape(date_label)} · {html_mod.escape(now_str)}</div>
</div>
<div class="summary"><strong>{len(results)}</strong> tasks {'would be' if dry_run else ''} archived (done &gt; {ARCHIVE_AFTER_DAYS} days)</div>
{board_sections if results else '<p style="color:var(--muted)">No tasks eligible for archiving.</p>'}
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Kanban auto-archive cron")
    parser.add_argument("--dry-run", action="store_true", help="Preview without archiving")
    args = parser.parse_args()

    all_results = []
    for board, db_path in find_all_board_dbs():
        if not os.path.exists(db_path):
            continue
        try:
            results = archive_stale_tasks(board, db_path, args.dry_run)
            all_results.extend(results)
        except Exception as e:
            print(f"ERROR scanning {board}: {e}", file=sys.stderr)

    # Write HTML report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_type = "dryrun" if args.dry_run else "archive"
    date_str = datetime.now(timezone(timedelta(hours=1))).strftime("%Y%m%d-%H%M")
    html_path = REPORT_DIR / f"kanban-{report_type}-{date_str}.html"
    html_path.write_text(render_html(all_results, args.dry_run), encoding="utf-8")

    # Short text summary for Discord + MEDIA tag
    mode = "DRY RUN" if args.dry_run else "ARCHIVED"
    if not all_results:
        print(f"Kanban Auto-Archive [{mode}]: 0 tasks eligible (done > {ARCHIVE_AFTER_DAYS} days).")
        print(f"MEDIA:{html_path}")
        return

    by_board_count = {}
    for r in all_results:
        by_board_count[r["board"]] = by_board_count.get(r["board"], 0) + 1

    board_summary = " · ".join(f"{b}: {c}" for b, c in sorted(by_board_count.items()))
    print(f"Kanban Auto-Archive [{mode}]: {len(all_results)} tasks {('would be' if args.dry_run else '')} archived")
    print(f"Boards: {board_summary}")
    print(f"MEDIA:{html_path}")


if __name__ == "__main__":
    main()