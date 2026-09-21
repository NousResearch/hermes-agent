#!/usr/bin/env python3
"""
KENSEI Quarterly Deep Audit — generates system health report.
Runs as no_agent: true cron on first Monday of Mar/Jun/Sep/Dec.
"""
import json, sqlite3, datetime as dt, subprocess
from pathlib import Path

TZ = dt.timezone(dt.timedelta(hours=1))
now = dt.datetime.now(TZ)
out = []

# === Profile count ===
prof_dir = Path("/home/kensei/.hermes/profiles")
profiles = [d.name for d in prof_dir.iterdir() if d.is_dir() and (d / "config.yaml").exists()]
out.append(f"• Profiles: <code>{len(profiles)}</code>")

# === Kanban health ===
db = Path("/home/kensei/.hermes/kanban.db")
if db.exists():
    conn = sqlite3.connect(str(db))
    cur = conn.execute("SELECT status, COUNT(*) FROM tasks GROUP BY status")
    status_counts = dict(cur.fetchall())
    conn.close()
    for s in ["triage", "ready", "todo", "running", "review", "blocked", "done", "archived"]:
        c = status_counts.get(s, 0)
        if c > 0:
            out.append(f"• <b>{s}</b>: <code>{c}</code>")
else:
    out.append("• Kanban DB: not found")

# === Cron health ===
try:
    r = subprocess.run(["python3", "-c", """
import sqlite3
conn = sqlite3.connect('/home/kensei/.hermes/cron/cron.db')
cur = conn.execute("SELECT last_status, COUNT(*) FROM jobs GROUP BY last_status")
for row in cur.fetchall():
    print(f'{row[0]}:{row[1]}')
conn.close()
"""], capture_output=True, text=True, timeout=10)
    if r.stdout:
        for line in r.stdout.strip().split("\n"):
            if ":" in line:
                status, count = line.split(":")
                out.append(f"• Crons <b>{status}</b>: <code>{count}</code>")
except Exception as ex:
    out.append(f"• Cron DB: could not read ({ex})")

# === Worker Failure Analysis trend ===
log_dir = Path("/home/kensei/.hermes/governance/logboard")
trends = sorted(log_dir.glob("worker-failure-analysis-*.json"))
if trends:
    recent = trends[-1]
    data = json.loads(recent.read_text())
    out.append(f"• Latest WFA: <code>{data.get('total_blocked', '?')}</code> blocked tasks")
    cats = data.get("by_category", {})
    for cat in ["decomposition_gap", "needs_reasoning", "protocol_violation", "execution_error"]:
        c = cats.get(cat, 0)
        if c > 0:
            out.append(f"  <code>{cat}</code>: <code>{c}</code>")

# === Output ===
print(f"📊 <b>Quarterly System Audit</b> · {now.strftime('%d/%m/%y · %H:%M:%S')}")
print(f"Q{ (now.month - 1) // 3 + 1} {now.year}")
print()
for line in out:
    print(line)
print()
print("<b>Next steps</b>")
print("• Kensei-Strategic reviews full system state")
print("• Produces quarterly health report with recommendations")
print("• Changes logged to Profile Change Ledger")
