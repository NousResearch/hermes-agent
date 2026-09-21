#!/usr/bin/env python3
"""Kanban v2 daily report.

Queries all boards for tasks updated in the last 24h. Produces a short
text summary for Discord + a full HTML report as MEDIA attachment.

Output: short text summary + HTML report
"""

import glob
import html as html_mod
import os
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

def _hermes_home() -> str:
    return os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))

REPORT_DIR = Path(_hermes_home()) / "kanban" / "reports"


def find_all_board_dbs() -> list[tuple[str, str]]:
    home = _hermes_home()
    dbs = [("default", os.path.join(home, "kanban.db"))]
    boards_dir = os.path.join(home, "kanban", "boards")
    if os.path.isdir(boards_dir):
        for child in sorted(os.listdir(boards_dir)):
            child_path = os.path.join(boards_dir, child, "kanban.db")
            if os.path.exists(child_path):
                dbs.append((child, child_path))
    return dbs


def get_epic_name(conn, epic_id):
    if not epic_id:
        return "ungrouped"
    try:
        row = conn.execute("SELECT title FROM epics WHERE id = ?", (epic_id,)).fetchone()
        return row[0] if row else epic_id
    except Exception:
        return epic_id


def collect_data():
    now = int(time.time())
    cutoff = now - 86400

    shipped = []
    in_progress = []
    blocked = []
    newly_created = []

    for board, db_path in find_all_board_dbs():
        if not os.path.exists(db_path):
            continue
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cols = {row["name"] for row in cur.execute("PRAGMA table_info(tasks)")}
            updated_col = "updated_at" if "updated_at" in cols else "completed_at"

            cur.execute(
                f"SELECT id, title, assignee, epic_id, done_at FROM tasks "
                f"WHERE status = 'done' AND done_at IS NOT NULL AND done_at > ? "
                f"ORDER BY done_at DESC",
                (cutoff,),
            )
            for row in cur.fetchall():
                epic_name = get_epic_name(conn, row["epic_id"])
                shipped.append({"board": board, "id": row["id"], "title": row["title"][:70], "assignee": row["assignee"] or "unassigned", "epic": epic_name})

            active_statuses = ("running", "ready", "todo", "review", "in_progress")
            placeholders = ",".join("?" for _ in active_statuses)
            cur.execute(
                f"SELECT id, title, assignee, epic_id, status FROM tasks "
                f"WHERE status IN ({placeholders}) AND {updated_col} > ? "
                f"ORDER BY priority DESC, {updated_col} DESC",
                (*active_statuses, cutoff),
            )
            for row in cur.fetchall():
                epic_name = get_epic_name(conn, row["epic_id"])
                in_progress.append({"board": board, "id": row["id"], "title": row["title"][:70], "assignee": row["assignee"] or "unassigned", "epic": epic_name, "status": row["status"]})

            cur.execute("SELECT id, title, assignee, epic_id, status_reason FROM tasks WHERE status = 'blocked' ORDER BY priority DESC")
            for row in cur.fetchall():
                epic_name = get_epic_name(conn, row["epic_id"])
                blocked.append({"board": board, "id": row["id"], "title": row["title"][:70], "assignee": row["assignee"] or "unassigned", "epic": epic_name, "reason": (row["status_reason"] or "")[:100]})

            cur.execute("SELECT id, title, assignee FROM tasks WHERE created_at > ? ORDER BY created_at DESC", (cutoff,))
            for row in cur.fetchall():
                newly_created.append({"board": board, "id": row["id"], "title": row["title"][:70], "assignee": row["assignee"] or "unassigned"})

            conn.close()
        except Exception as e:
            print(f"ERROR scanning {board}: {e}", file=sys.stderr)

    return shipped, in_progress, blocked, newly_created


def render_html(shipped, in_progress, blocked, newly_created) -> str:
    uk_tz = timezone(timedelta(hours=1))
    now_str = datetime.now(uk_tz).strftime("%d/%m/%Y %H:%M")
    date_label = datetime.now(uk_tz).strftime("%A, %B %-d, %Y")

    def task_rows(items, show_status=False, show_reason=False):
        rows = ""
        for item in items:
            epic_val = html_mod.escape(item.get("epic", "—"))
            status_cell = f'<td class="muted">{html_mod.escape(item.get("status", ""))}</td>' if show_status else ""
            reason_cell = f'<td class="muted">{html_mod.escape(item.get("reason", ""))}</td>' if show_reason else ""
            status_th = "<th>Status</th>" if show_status else ""
            reason_th = "<th>Reason</th>" if show_reason else ""
            if not rows:
                rows = f"<tr><th>ID</th><th>Title</th><th>Assignee</th><th>Epic</th>{status_th}{reason_th}</tr>\n"
            rows += (
                f'<tr><td class="mono">{html_mod.escape(item["id"])}</td>'
                f'<td>{html_mod.escape(item["title"])}</td>'
                f'<td class="muted">{html_mod.escape(item["assignee"])}</td>'
                f'<td class="muted">{epic_val}</td>'
                f'{status_cell}{reason_cell}</tr>\n'
            )
        return rows

    def section(title, items, show_status=False, show_reason=False):
        if not items:
            return ""
        rows = task_rows(items, show_status, show_reason)
        return f'<h2>{html_mod.escape(title)} ({len(items)})</h2>\n<table class="data">\n{rows}</table>\n'

    shipped_section = section("Shipped", shipped)
    progress_section = section("In Progress", in_progress, show_status=True)
    blocked_section = section("Blocked", blocked, show_reason=True)
    new_section = section("New Tasks", newly_created)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kanban Daily Report — {html_mod.escape(now_str)}</title>
<style>
  :root {{ color-scheme: dark; --bg: #11100f; --card: #1c1a18; --muted: #a8a29e; --text: #f5f5f4; --accent: #fbbf24; --line: #34302c; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 24px; background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-size: 14px; line-height: 1.6; }}
  .header {{ border-bottom: 1px solid var(--line); padding-bottom: 16px; margin-bottom: 24px; }}
  .header h1 {{ font-size: 20px; margin: 0 0 4px 0; }}
  .header .meta {{ color: var(--muted); font-size: 13px; }}
  .summary {{ display: flex; gap: 24px; margin: 16px 0 24px 0; }}
  .stat {{ text-align: center; }}
  .stat .num {{ font-size: 28px; font-weight: 700; color: var(--accent); }}
  .stat .label {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }}
  h2 {{ font-size: 16px; margin: 24px 0 8px 0; color: var(--accent); }}
  table.data {{ width: 100%; border-collapse: collapse; margin-bottom: 16px; }}
  table.data th {{ text-align: left; padding: 6px 12px; border-bottom: 1px solid var(--line); color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
  table.data td {{ padding: 6px 12px; border-bottom: 1px solid var(--line); }}
  .mono {{ font-family: 'SF Mono', 'Fira Code', monospace; font-size: 12px; color: var(--accent); }}
  .muted {{ color: var(--muted); }}
</style>
</head>
<body>
<div class="header">
  <h1>Kanban Daily Report</h1>
  <div class="meta">{html_mod.escape(date_label)} · {html_mod.escape(now_str)}</div>
</div>
<div class="summary">
  <div class="stat"><div class="num">{len(shipped)}</div><div class="label">Shipped</div></div>
  <div class="stat"><div class="num">{len(in_progress)}</div><div class="label">In Progress</div></div>
  <div class="stat"><div class="num">{len(blocked)}</div><div class="label">Blocked</div></div>
  <div class="stat"><div class="num">{len(newly_created)}</div><div class="label">New</div></div>
</div>
{shipped_section}
{progress_section}
{blocked_section}
{new_section if newly_created else '<p style="color:var(--muted)">No new tasks in the last 24 hours.</p>'}
</body>
</html>"""


def main():
    shipped, in_progress, blocked, newly_created = collect_data()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone(timedelta(hours=1))).strftime("%Y%m%d-%H%M")
    html_path = REPORT_DIR / f"kanban-daily-{date_str}.html"
    html_path.write_text(render_html(shipped, in_progress, blocked, newly_created), encoding="utf-8")

    # Short text summary
    print(f"Kanban Daily Report — {datetime.now(timezone(timedelta(hours=1))).strftime('%d/%m/%Y')}")
    print(f"{len(shipped)} shipped | {len(in_progress)} in progress | {len(blocked)} blocked | {len(newly_created)} new")
    if shipped:
        by_assignee = defaultdict(int)
        for s in shipped:
            by_assignee[s["assignee"]] += 1
        top = " · ".join(f"{a}: {c}" for a, c in sorted(by_assignee.items(), key=lambda x: -x[1])[:3])
        print(f"Top: {top}")
    if blocked:
        print(f"Blocked: {' · '.join(b['title'][:40] for b in blocked[:2])}")
    print(f"MEDIA:{html_path}")


if __name__ == "__main__":
    main()