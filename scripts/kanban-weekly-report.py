#!/usr/bin/env python3
"""Kanban v2 weekly report.

Queries all boards for tasks updated in the last 7 days. Produces a
short text summary for Discord + a full HTML report as MEDIA attachment.

Output: short text summary + HTML report with epic progress, shipped,
active by assignee, blocked items, top backlog.
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
    cutoff = now - (7 * 86400)

    shipped_by_epic = defaultdict(list)
    active_by_assignee = defaultdict(list)
    blocked_items = []
    backlog_items = []
    epic_summary = defaultdict(lambda: {"total": 0, "done": 0, "active": 0, "backlog": 0, "blocked": 0, "archived": 0})

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
                "SELECT id, title, assignee, epic_id, done_at FROM tasks "
                "WHERE status = 'done' AND done_at IS NOT NULL AND done_at > ? "
                "ORDER BY done_at DESC",
                (cutoff,),
            )
            for row in cur.fetchall():
                epic = get_epic_name(conn, row["epic_id"])
                shipped_by_epic[epic].append({"board": board, "id": row["id"], "title": row["title"][:70], "assignee": row["assignee"] or "unassigned"})

            active_statuses = ("running", "ready", "todo", "review", "in_progress")
            placeholders = ",".join("?" for _ in active_statuses)
            cur.execute(
                f"SELECT id, title, assignee, epic_id, status FROM tasks "
                f"WHERE status IN ({placeholders}) ORDER BY priority DESC",
                active_statuses,
            )
            for row in cur.fetchall():
                epic = get_epic_name(conn, row["epic_id"])
                active_by_assignee[row["assignee"] or "unassigned"].append(
                    {"board": board, "id": row["id"], "title": row["title"][:70], "epic": epic, "status": row["status"]}
                )

            cur.execute("SELECT id, title, assignee, epic_id, status_reason, priority FROM tasks WHERE status = 'blocked' ORDER BY priority DESC")
            for row in cur.fetchall():
                epic = get_epic_name(conn, row["epic_id"])
                blocked_items.append({"board": board, "id": row["id"], "title": row["title"][:70], "assignee": row["assignee"] or "unassigned", "epic": epic, "reason": (row["status_reason"] or "")[:100]})

            cur.execute("SELECT id, title, assignee, epic_id, priority FROM tasks WHERE status = 'backlog' ORDER BY priority DESC, created_at ASC LIMIT 10")
            for row in cur.fetchall():
                epic = get_epic_name(conn, row["epic_id"])
                backlog_items.append({"board": board, "id": row["id"], "title": row["title"][:70], "assignee": row["assignee"] or "unassigned", "epic": epic, "priority": row["priority"]})

            cur.execute("SELECT epic_id, status, COUNT(*) as cnt FROM tasks GROUP BY epic_id, status")
            for row in cur.fetchall():
                epic_id = row["epic_id"]
                if not epic_id:
                    continue
                epic_name = get_epic_name(conn, epic_id)
                status = row["status"]
                count = row["cnt"]
                epic_summary[epic_name]["total"] += count
                if status == "done":
                    epic_summary[epic_name]["done"] += count
                elif status == "archived":
                    epic_summary[epic_name]["archived"] += count
                elif status == "backlog":
                    epic_summary[epic_name]["backlog"] += count
                elif status == "blocked":
                    epic_summary[epic_name]["blocked"] += count
                else:
                    epic_summary[epic_name]["active"] += count

            conn.close()
        except Exception as e:
            print(f"ERROR scanning {board}: {e}", file=sys.stderr)

    return shipped_by_epic, active_by_assignee, blocked_items, backlog_items, epic_summary


def render_html(shipped_by_epic, active_by_assignee, blocked_items, backlog_items, epic_summary) -> str:
    uk_tz = timezone(timedelta(hours=1))
    now_str = datetime.now(uk_tz).strftime("%d/%m/%Y %H:%M")
    date_label = datetime.now(uk_tz).strftime("%A, %B %-d, %Y")

    total_shipped = sum(len(v) for v in shipped_by_epic.values())
    total_active = sum(len(v) for v in active_by_assignee.values())

    # Epic progress table
    epic_rows = ""
    for epic in sorted(epic_summary.keys()):
        s = epic_summary[epic]
        epic_rows += (
            f'<tr><td class="epic">{html_mod.escape(epic)}</td>'
            f'<td class="num">{s["total"]}</td><td class="num">{s["done"]}</td>'
            f'<td class="num">{s["active"]}</td><td class="num">{s["backlog"]}</td>'
            f'<td class="num">{s["blocked"]}</td><td class="num muted">{s["archived"]}</td></tr>\n'
        )

    # Shipped section
    shipped_html = ""
    if shipped_by_epic:
        for epic in sorted(shipped_by_epic.keys()):
            items = shipped_by_epic[epic]
            rows = ""
            for item in items:
                rows += (
                    f'<tr><td class="mono">{html_mod.escape(item["id"])}</td>'
                    f'<td>{html_mod.escape(item["title"])}</td>'
                    f'<td class="muted">{html_mod.escape(item["assignee"])}</td></tr>\n'
                )
            shipped_html += f'<h3>{html_mod.escape(epic)} ({len(items)})</h3>\n<table class="data"><tr><th>ID</th><th>Title</th><th>Assignee</th></tr>\n{rows}</table>\n'

    # Active by assignee
    active_html = ""
    if active_by_assignee:
        for assignee in sorted(active_by_assignee.keys()):
            items = active_by_assignee[assignee]
            rows = ""
            for item in items:
                rows += (
                    f'<tr><td class="mono">{html_mod.escape(item["id"])}</td>'
                    f'<td>{html_mod.escape(item["title"])}</td>'
                    f'<td class="muted">{html_mod.escape(item["epic"])}</td>'
                    f'<td class="muted">{html_mod.escape(item["status"])}</td></tr>\n'
                )
            active_html += f'<h3>{html_mod.escape(assignee)} ({len(items)})</h3>\n<table class="data"><tr><th>ID</th><th>Title</th><th>Epic</th><th>Status</th></tr>\n{rows}</table>\n'

    # Blocked
    blocked_html = ""
    if blocked_items:
        rows = ""
        for item in blocked_items:
            rows += (
                f'<tr><td class="mono">{html_mod.escape(item["id"])}</td>'
                f'<td>{html_mod.escape(item["title"])}</td>'
                f'<td class="muted">{html_mod.escape(item["assignee"])}</td>'
                f'<td class="muted">{html_mod.escape(item["reason"])}</td></tr>\n'
            )
        blocked_html = f'<h2>Blocked ({len(blocked_items)})</h2>\n<table class="data"><tr><th>ID</th><th>Title</th><th>Assignee</th><th>Reason</th></tr>\n{rows}</table>\n'

    # Backlog
    backlog_html = ""
    if backlog_items:
        rows = ""
        for item in backlog_items:
            rows += (
                f'<tr><td class="mono">P{item["priority"]}</td>'
                f'<td class="mono">{html_mod.escape(item["id"])}</td>'
                f'<td>{html_mod.escape(item["title"])}</td>'
                f'<td class="muted">{html_mod.escape(item["assignee"])}</td>'
                f'<td class="muted">{html_mod.escape(item["epic"])}</td></tr>\n'
            )
        backlog_html = f'<h2>Top Backlog ({len(backlog_items)})</h2>\n<table class="data"><tr><th>Pri</th><th>ID</th><th>Title</th><th>Assignee</th><th>Epic</th></tr>\n{rows}</table>\n'

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kanban Weekly Report — {html_mod.escape(now_str)}</title>
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
  h2 {{ font-size: 16px; margin: 28px 0 8px 0; color: var(--accent); border-bottom: 1px solid var(--line); padding-bottom: 4px; }}
  h3 {{ font-size: 14px; margin: 16px 0 6px 0; color: var(--text); }}
  table.data {{ width: 100%; border-collapse: collapse; margin-bottom: 16px; }}
  table.data th {{ text-align: left; padding: 6px 12px; border-bottom: 1px solid var(--line); color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
  table.data td {{ padding: 6px 12px; border-bottom: 1px solid var(--line); }}
  table.epic th, table.epic td {{ text-align: center; }}
  table.epic td.epic, table.epic th:first-child {{ text-align: left; }}
  .mono {{ font-family: 'SF Mono', 'Fira Code', monospace; font-size: 12px; color: var(--accent); }}
  .num {{ text-align: center; font-variant-numeric: tabular-nums; }}
  .muted {{ color: var(--muted); }}
</style>
</head>
<body>
<div class="header">
  <h1>Kanban Weekly Report</h1>
  <div class="meta">{html_mod.escape(date_label)} · {html_mod.escape(now_str)}</div>
</div>
<div class="summary">
  <div class="stat"><div class="num">{total_shipped}</div><div class="label">Shipped</div></div>
  <div class="stat"><div class="num">{total_active}</div><div class="label">Active</div></div>
  <div class="stat"><div class="num">{len(blocked_items)}</div><div class="label">Blocked</div></div>
  <div class="stat"><div class="num">{len(backlog_items)}</div><div class="label">Backlog</div></div>
</div>
<h2>Epic Progress</h2>
<table class="data epic">
<tr><th>Epic</th><th>Total</th><th>Done</th><th>Active</th><th>Backlog</th><th>Blocked</th><th>Archived</th></tr>
{epic_rows}
</table>
<h2>Shipped This Week ({total_shipped})</h2>
{shipped_html if shipped_by_epic else '<p style="color:var(--muted)">No tasks shipped this week.</p>'}
<h2>Active by Assignee ({total_active})</h2>
{active_html if active_by_assignee else '<p style="color:var(--muted)">No active tasks.</p>'}
{blocked_html}
{backlog_html}
</body>
</html>"""


def main():
    shipped_by_epic, active_by_assignee, blocked_items, backlog_items, epic_summary = collect_data()

    total_shipped = sum(len(v) for v in shipped_by_epic.values())
    total_active = sum(len(v) for v in active_by_assignee.values())

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone(timedelta(hours=1))).strftime("%Y%m%d")
    html_path = REPORT_DIR / f"kanban-weekly-{date_str}.html"
    html_path.write_text(render_html(shipped_by_epic, active_by_assignee, blocked_items, backlog_items, epic_summary), encoding="utf-8")

    # Short text summary
    print(f"Kanban Weekly Report — Week of {datetime.now(timezone(timedelta(hours=1))).strftime('%d/%m/%Y')}")
    print(f"{total_shipped} shipped | {total_active} active | {len(blocked_items)} blocked | {len(backlog_items)} backlog")
    if epic_summary:
        top_epics = sorted(epic_summary.items(), key=lambda x: x[1]["done"], reverse=True)[:3]
        epic_str = " · ".join(f"{name}: {s['done']} done" for name, s in top_epics)
        print(f"Epics: {epic_str}")
    if blocked_items:
        print(f"Blocked: {' · '.join(b['title'][:35] for b in blocked_items[:2])}")
    print(f"MEDIA:{html_path}")


if __name__ == "__main__":
    main()