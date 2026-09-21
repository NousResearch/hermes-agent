#!/usr/bin/env python3
"""KENSEI daily briefing: deterministic, Discord-safe, with an HTML attachment.

The single daily anchor message. Sections:
  NEEDS YOU      tasks with escalation_target set (the structured human-decision
                 signal), oldest first. This is the canonical "for Sahil" list.
  COMPLETED      done yesterday, grouped per agent (assignee).
  IN PROGRESS    currently running, per agent.
  BLOCKED        blocked / review / triage awaiting movement (awareness).
  BACKLOG        count, high-priority count, age of the oldest item.

Discord body stays short (title, counts, and the NEEDS-YOU items inline because
they are the actionable core); all detail lives in the HTML attachment. Stays
silent only when there is genuinely nothing in any section.
"""
from __future__ import annotations

import datetime as dt
import html
import os
import sqlite3
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

DB = Path(os.environ.get('HERMES_KANBAN_DB') or '/home/kensei/.hermes/kanban.db')
BASE = Path('/home/kensei/.hermes/runbooks/kanban-digest')
TZ = ZoneInfo('Europe/London')
HIGH_PRIORITY = 2  # tasks.priority >= this count as high

now = dt.datetime.now(TZ)
now_ts = int(now.timestamp())
yesterday = (now - dt.timedelta(days=1)).date()
start = int(dt.datetime.combine(yesterday, dt.time.min, TZ).timestamp())
end = int(dt.datetime.combine(yesterday + dt.timedelta(days=1), dt.time.min, TZ).timestamp())
out_dir = BASE / yesterday.strftime('%d-%m-%y')
out_dir.mkdir(parents=True, exist_ok=True)
html_path = out_dir / 'briefing.html'


def short(s: str | None, n: int = 86) -> str:
    s = ' '.join((s or '').split())
    return s if len(s) <= n else s[: n - 1].rstrip() + '…'


def age_days(ts) -> int:
    if not ts:
        return 0
    return max(0, (now_ts - int(ts)) // 86400)


conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row


def q(sql: str, params: tuple = ()) -> list[dict]:
    return [dict(r) for r in conn.execute(sql, params)]


needs_you = q(
    """
    SELECT id, title, status, assignee, escalation_target, status_reason,
           priority, created_at
    FROM tasks
    WHERE escalation_target IS NOT NULL AND status NOT IN ('done', 'archived')
    ORDER BY created_at ASC
    """
)
completed = q(
    """
    SELECT id, title, assignee, completed_at, result, status_reason
    FROM tasks
    WHERE status = 'done' AND completed_at >= ? AND completed_at < ?
    ORDER BY completed_at DESC
    """,
    (start, end),
)
in_progress = q(
    "SELECT id, title, assignee, started_at FROM tasks "
    "WHERE status = 'running' ORDER BY started_at ASC"
)
blocked = q(
    "SELECT id, title, assignee, status, status_reason, last_failure_error, "
    "created_at FROM tasks WHERE status IN ('blocked', 'review', 'triage') "
    "ORDER BY created_at ASC"
)
backlog = q(
    "SELECT id, title, priority, created_at FROM tasks "
    "WHERE status = 'backlog' ORDER BY created_at ASC"
)
conn.close()

backlog_high = [r for r in backlog if (r.get('priority') or 0) >= HIGH_PRIORITY]
backlog_oldest = age_days(backlog[0]['created_at']) if backlog else 0

# Nothing at all to say across every section: stay silent.
if not any((needs_you, completed, in_progress, blocked, backlog)):
    raise SystemExit(0)


def by_agent(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[(r.get('assignee') or 'unassigned')].append(r)
    return dict(sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])))


# ---- Discord body (short; NEEDS-YOU inline, rest in HTML) ----
counts = (
    f"{len(needs_you)} need you · {len(completed)} completed · "
    f"{len(in_progress)} in progress · {len(blocked)} blocked · "
    f"{len(backlog)} backlog"
)
if backlog:
    counts += f" (oldest {backlog_oldest}d)"

lines = [
    f"📋 KENSEI Daily Briefing · {now.strftime('%a %d %b')}",
    counts,
]
if needs_you:
    lines.append('')
    lines.append('🔴 NEEDS YOU')
    for r in needs_you[:8]:
        reason = short(r.get('status_reason'), 60)
        suffix = f' : {reason}' if reason else ''
        lines.append(f"• `{r['id']}` {short(r['title'], 70)}{suffix}")
    if len(needs_you) > 8:
        lines.append(f"• (+{len(needs_you) - 8} more in the attachment)")
lines.append('')
lines.append(f"MEDIA:{html_path}")


# ---- HTML attachment ----
def esc(s) -> str:
    return html.escape(str(s if s is not None else ''))


def agent_block(title: str, rows: list[dict], signal_key: str | None = None) -> str:
    if not rows:
        return ''
    out = [f"<h3>{esc(title)}</h3>"]
    for agent, items in by_agent(rows).items():
        out.append(f"<div class='agent'><span class='who'>{esc(agent)}</span>"
                   f"<span class='n'>{len(items)}</span></div>")
        out.append("<ul>")
        for r in items:
            sig = short(r.get(signal_key), 140) if signal_key else ''
            sig_html = f" <span class='sig'>{esc(sig)}</span>" if sig else ''
            out.append(f"<li><code>{esc(r['id'])}</code> {esc(short(r['title'], 150))}"
                       f"{sig_html}</li>")
        out.append("</ul>")
    return ''.join(out)


needs_rows = ''.join(
    "<tr>"
    f"<td><code>{esc(r['id'])}</code></td>"
    f"<td>{esc(short(r['title'], 160))}</td>"
    f"<td>{esc(r['status'])}</td>"
    f"<td>{esc(r.get('assignee') or '')}</td>"
    f"<td>{esc(age_days(r['created_at']))}d</td>"
    f"<td>{esc(short(r.get('status_reason'), 200))}</td>"
    "</tr>"
    for r in needs_you
) or "<tr><td colspan='6' class='muted'>Nothing awaiting your decision.</td></tr>"

blocked_rows = ''.join(
    "<tr>"
    f"<td><code>{esc(r['id'])}</code></td>"
    f"<td>{esc(short(r['title'], 160))}</td>"
    f"<td>{esc(r['status'])}</td>"
    f"<td>{esc(r.get('assignee') or '')}</td>"
    f"<td>{esc(age_days(r['created_at']))}d</td>"
    f"<td>{esc(short(r.get('status_reason') or r.get('last_failure_error'), 200))}</td>"
    "</tr>"
    for r in blocked
) or "<tr><td colspan='6' class='muted'>None.</td></tr>"

backlog_rows = ''.join(
    "<tr>"
    f"<td><code>{esc(r['id'])}</code></td>"
    f"<td>{esc(short(r['title'], 180))}</td>"
    f"<td>{esc(r.get('priority') or 0)}</td>"
    f"<td>{esc(age_days(r['created_at']))}d</td>"
    "</tr>"
    for r in backlog[:40]
) or "<tr><td colspan='4' class='muted'>Empty.</td></tr>"

html_doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>KENSEI Daily Briefing · {yesterday.strftime('%d/%m/%Y')}</title>
<style>
:root {{ color-scheme: dark; --bg:#0f1115; --card:#181b22; --line:#2a2f3a; --text:#e7e3d8;
  --muted:#8e96a6; --red:#e0584c; --amber:#e0982e; --green:#46b39a; --blue:#5b8def; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.55 system-ui,-apple-system,Segoe UI,sans-serif; }}
main {{ width:min(1100px, calc(100% - 32px)); margin:0 auto; padding:28px 0 56px; }}
h1 {{ margin:0 0 2px; font-size:26px; letter-spacing:-0.02em; }}
.muted {{ color:var(--muted); }}
.summary {{ display:flex; gap:10px; flex-wrap:wrap; margin:18px 0 24px; }}
.pill {{ background:var(--card); border:1px solid var(--line); border-radius:999px; padding:7px 13px; font-size:13px; }}
.pill b {{ font-variant-numeric:tabular-nums; }}
section {{ background:var(--card); border:1px solid var(--line); border-radius:14px; margin:16px 0; overflow:hidden; }}
section > h2 {{ margin:0; padding:13px 16px; border-bottom:1px solid var(--line); font-size:15px; letter-spacing:.02em; }}
.s-need > h2 {{ color:var(--red); }} .s-done > h2 {{ color:var(--green); }}
.s-prog > h2 {{ color:var(--amber); }} .s-block > h2 {{ color:#e07a3e; }} .s-back > h2 {{ color:var(--blue); }}
.body {{ padding:6px 16px 14px; }}
h3 {{ color:var(--text); font-size:13px; margin:14px 0 6px; }}
.agent {{ display:flex; align-items:center; gap:8px; margin:12px 0 2px; }}
.agent .who {{ font-weight:600; }} .agent .n {{ color:var(--muted); font-size:12px; }}
ul {{ margin:4px 0 0; padding-left:18px; }} li {{ margin:3px 0; }}
.sig {{ color:var(--muted); }}
table {{ width:100%; border-collapse:collapse; }}
th,td {{ padding:9px 12px; border-bottom:1px solid var(--line); vertical-align:top; text-align:left; }}
th {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.05em; }}
td:nth-child(5) {{ white-space:nowrap; font-variant-numeric:tabular-nums; }}
code {{ color:#fde68a; font-size:12.5px; }}
</style></head><body><main>
<h1>📋 KENSEI Daily Briefing</h1>
<p class="muted">{yesterday.strftime('%A %d %B %Y')} · generated {now.strftime('%H:%M')}</p>
<div class="summary">
<div class="pill" style="border-color:var(--red)">🔴 Need you <b>{len(needs_you)}</b></div>
<div class="pill" style="border-color:var(--green)">✅ Completed <b>{len(completed)}</b></div>
<div class="pill" style="border-color:var(--amber)">🔄 In progress <b>{len(in_progress)}</b></div>
<div class="pill" style="border-color:#e07a3e">⏸ Blocked <b>{len(blocked)}</b></div>
<div class="pill" style="border-color:var(--blue)">📥 Backlog <b>{len(backlog)}</b> ({len(backlog_high)} high)</div>
</div>

<section class="s-need"><h2>🔴 NEEDS YOU</h2>
<table><thead><tr><th>ID</th><th>Title</th><th>Status</th><th>Owner</th><th>Age</th><th>Reason</th></tr></thead>
<tbody>{needs_rows}</tbody></table></section>

<section class="s-done"><h2>✅ Completed yesterday ({len(completed)})</h2>
<div class="body">{agent_block('', completed, 'result') or "<p class='muted'>Nothing completed yesterday.</p>"}</div></section>

<section class="s-prog"><h2>🔄 In progress ({len(in_progress)})</h2>
<div class="body">{agent_block('', in_progress) or "<p class='muted'>Idle.</p>"}</div></section>

<section class="s-block"><h2>⏸ Blocked ({len(blocked)})</h2>
<table><thead><tr><th>ID</th><th>Title</th><th>Status</th><th>Owner</th><th>Age</th><th>Reason</th></tr></thead>
<tbody>{blocked_rows}</tbody></table></section>

<section class="s-back"><h2>📥 Backlog ({len(backlog)} · {len(backlog_high)} high · oldest {backlog_oldest}d)</h2>
<table><thead><tr><th>ID</th><th>Title</th><th>Priority</th><th>Age</th></tr></thead>
<tbody>{backlog_rows}</tbody></table></section>
</main></body></html>"""

html_path.write_text(html_doc)
if not html_path.exists():
    raise SystemExit('HTML attachment was not written')

# Optional spoken briefing via on-box Kokoro. Off by default; enable with
# BRIEFING_AUDIO=1 on the cron job. Runs under the repo .venv (Kokoro is not
# importable under system python3). Fail-safe: if synthesis fails the briefing
# still delivers as HTML + text.
if os.environ.get('BRIEFING_AUDIO', '').strip().lower() in ('1', 'true', 'yes', 'on'):
    import subprocess
    import sys
    spoken = '\n'.join(ln for ln in lines if not ln.startswith('MEDIA:')).strip()
    mp3_path = out_dir / 'briefing.mp3'
    txt_path = out_dir / 'briefing-audio.txt'
    try:
        txt_path.write_text(spoken, encoding='utf-8')
        subprocess.run(
            ['/home/kensei/repos/KenseiAgent/.venv/bin/python',
             '/home/kensei/.hermes/scripts/briefing_audio.py',
             str(txt_path), str(mp3_path),
             os.environ.get('BRIEFING_AUDIO_VOICE', 'kokoro:bf_emma')],
            check=True, timeout=240,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        if mp3_path.exists():
            lines.append(f'MEDIA:{mp3_path}')
    except Exception as e:
        # Surface the helper's stderr so a synth failure is diagnosable from
        # the cron log, not just the CalledProcessError repr.
        detail = getattr(e, 'stderr', b'') or b''
        if isinstance(detail, bytes):
            detail = detail.decode('utf-8', 'replace')
        print(f'briefing audio skipped: {e} {detail}'.strip(), file=sys.stderr)

print('\n'.join(lines).strip())
