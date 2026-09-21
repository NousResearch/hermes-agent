#!/usr/bin/env python3
"""Morning Pulse — single consolidated fleet health report (Group 1 merge).

Replaces: system-health + kensei-system-audit-daily + wesker-ops-daily.
Deterministic no_agent script. Zero LLM cost.

stdout contract (no_agent cron):
  - ALWAYS posts (scheduled report, not a watchdog).
  - Line 1: "🌅 Morning Pulse — DD/MM/YYYY · <ALL GREEN | N WARNINGS | N CRITICAL>"
  - ≤10 grouped status lines with 🟢/🟡/🔴 per check family.
  - Last line: MEDIA:/home/kensei/.hermes/reports/morning-pulse-YYYYMMDD.html
  - HTML: dark mode, red/amber items first, full evidence values.
"""
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Europe/London")
NOW = datetime.now(TZ)
HERMES = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
REPORTS = HERMES / "reports"
GUILD = "1506021204363051249"

CHECKS = []          # (family, status, detail)  status in {"ok","warn","crit"}


def add(family, status, detail):
    CHECKS.append((family, status, detail))


# ── 1. Memory / swap ────────────────────────────────────────────────────────
def check_memory():
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1)
        info[k.strip()] = int(v.strip().split()[0])  # kB
    swap_total = info.get("SwapTotal", 0)
    swap_free = info.get("SwapFree", 0)
    mem_total = info.get("MemTotal", 1)
    mem_avail = info.get("MemAvailable", 0)
    mem_pct = 100 * (mem_total - mem_avail) // mem_total
    if swap_total == 0:
        add("memory", "warn", f"no swap configured · RAM {mem_pct}% used")
        return
    swap_pct = 100 * (swap_total - swap_free) // swap_total
    if swap_pct > 80:
        add("memory", "crit", f"swap {swap_pct}% used · RAM {mem_pct}%")
    elif swap_pct > 50:
        add("memory", "warn", f"swap {swap_pct}% used · RAM {mem_pct}%")
    else:
        add("memory", "ok", f"RAM {mem_pct}% · swap {swap_pct}%")


# ── 2. Disk ────────────────────────────────────────────────────────────────
def check_disk():
    worst = ""
    worst_free = 100
    for mount in ("/", "/home"):
        try:
            u = shutil.disk_usage(mount)
        except OSError:
            continue
        free_pct = 100 * u.free // u.total
        if free_pct < worst_free:
            worst_free = free_pct
            worst = mount
    if worst_free < 15:
        add("disk", "crit", f"{worst} only {worst_free}% free")
    elif worst_free < 25:
        add("disk", "warn", f"{worst} {worst_free}% free")
    else:
        add("disk", "ok", f"/ {shutil.disk_usage('/')[0] and ''}{100*shutil.disk_usage('/').free//shutil.disk_usage('/').total}% · /home {100*shutil.disk_usage('/home').free//shutil.disk_usage('/home').total}% free")


# ── 3. Gateway + orphans ───────────────────────────────────────────────────
def check_gateway():
    try:
        out = subprocess.run(["pgrep", "-fc", "hermes_cli.main gateway"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        n = int(out or 0)
    except Exception:
        n = 0
    if n == 0:
        add("gateway", "crit", "gateway process NOT running")
        return
    # orphan candidates: dead-parent children is complex; report count only
    add("gateway", "ok", f"up ({n} proc{'esses' if n != 1 else ''})")


# ── 4/5. Reuse the dedicated checkers (silent = healthy) ───────────────────
def run_checker(script, label, ok_detail):
    try:
        r = subprocess.run([sys.executable, str(HERMES / "scripts" / script)],
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout or "").strip()
        bad = r.returncode != 0 or bool(out)
        if bad:
            first = out.splitlines()[0][:110] if out else f"exit={r.returncode}"
            sev = "crit" if "ALERT" in out.upper() or "CRITICAL" in first else "warn"
            add(label, sev, first)
        else:
            add(label, "ok", ok_detail)
    except Exception as e:
        add(label, "warn", f"checker error: {str(e)[:80]}")


# ── 6. Backup freshness ────────────────────────────────────────────────────
def check_backup():
    d = Path.home() / "backups" / "daily"
    arcs = sorted(d.glob("kensei-*.tar.gz"), key=lambda p: p.stat().st_mtime) if d.is_dir() else []
    if not arcs:
        add("backup", "crit", "no archives in backups/daily")
        return
    newest = arcs[-1]
    age_h = (datetime.now().timestamp() - newest.stat().st_mtime) / 3600
    if age_h > 36:
        add("backup", "crit", f"newest archive {age_h:.0f}h old")
    else:
        add("backup", "ok", f"{age_h:.0f}h old ({newest.name[:34]})")


# ── 7. Cron failures last 24h ──────────────────────────────────────────────
def check_cron():
    errs = []
    try:
        store = json.loads((HERMES / "cron" / "jobs.json").read_text())["jobs"]
        errs = [(j.get("name", j["id"][:8]), j.get("last_error") or "")
                for j in store if j.get("enabled") and j.get("last_status") == "error"]
    except Exception as e:
        add("cron", "warn", f"jobs.json unreadable: {e}")
        return
    fails24 = 0
    try:
        con = sqlite3.connect(f"file:{HERMES}/cron/executions.db?mode=ro", uri=True)
        since = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        fails24 = con.execute(
            "SELECT COUNT(*) FROM executions WHERE started_at >= ? AND status='failed'",
            (since,)).fetchone()[0]
        con.close()
    except Exception:
        pass
    if len(errs) >= 3:
        add("cron", "crit", f"{len(errs)} jobs error-status · {fails24} failed runs/24h")
    elif errs:
        names = ", ".join(n for n, _ in errs[:3])
        add("cron", "warn", f"{len(errs)} job(s) errored: {names}")
    else:
        add("cron", "ok", "0 errors · scheduler current")


# ── 8. Config drift (~/.hermes git) ────────────────────────────────────────
def check_config_drift():
    try:
        r = subprocess.run(["git", "-C", str(HERMES), "status", "--porcelain"],
                           capture_output=True, text=True, timeout=15)
        lines = [l for l in r.stdout.splitlines() if l.strip()]
        if not lines:
            add("config", "ok", "~/.hermes clean")
        elif len(lines) > 12:
            add("config", "warn", f"{len(lines)} uncommitted paths (auto-commit pending)")
        else:
            add("config", "warn", f"{len(lines)} uncommitted: " + ", ".join(l[3:] for l in lines[:3]))
    except Exception as e:
        add("config", "warn", f"git check failed: {e}")


# ── Rendering ──────────────────────────────────────────────────────────────
ICON = {"ok": "🟢", "warn": "🟡", "crit": "🔴"}
ORDER = {"crit": 0, "warn": 1, "ok": 2}


def build_html(results):
    def esc(s):
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    crit = [r for r in results if r[1] == "crit"]
    warn = [r for r in results if r[1] == "warn"]

    def row(r):
        return ('<div class="card ' + r[1] + '"><span class="ico">' + ICON[r[1]]
                + '</span> <b>' + esc(r[0]) + "</b><div class=d>" + esc(r[2])
                + "</div></div>")
    attention = "\n".join(row(r) for r in sorted(crit + warn, key=lambda x: ORDER[x[1]]))
    body_rows = "\n".join(row(r) for r in sorted(results, key=lambda x: (ORDER[x[1]], x[0])))
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Morning Pulse — {NOW.strftime('%d/%m/%Y')}</title>
<style>
html {{ color-scheme: dark; }}
body {{ background:#11100f; color:#f5f5f4; font-family:ui-sans-serif,system-ui,sans-serif; margin:0; padding:2rem; }}
h1 {{ color:#fbbf24; font-size:1.35rem; margin:0 0 .2rem; }}
h2 {{ color:#fbbf24; font-size:1rem; margin:1.5rem 0 .5rem; border-bottom:1px solid #34302c; padding-bottom:.3rem; }}
.muted {{ color:#a8a29e; font-size:.85rem; }}
.card {{ background:#1c1a18; border:1px solid #34302c; border-left-width:4px; border-radius:8px; padding:.6rem .9rem; margin:.45rem 0; }}
.card.crit {{ border-left-color:#ef4444; }}
.card.warn {{ border-left-color:#f59e0b; }}
.card.ok {{ border-left-color:#22c55e; }}
.ico {{ margin-right:.25rem; }}
.d {{ color:#a8a29e; font-size:.85rem; margin-top:.2rem; }}
</style></head><body>
<h1>🌅 Morning Pulse</h1>
<div class="muted">{NOW.strftime('%d/%m/%Y %H:%M')} UK · generated by morning_pulse.py</div>
<h2>Needs attention</h2>
{attention or '<div class="card ok">Nothing — all checks green.</div>'}
<h2>All checks</h2>
{body_rows}
</body></html>"""


def main():
    check_memory()
    check_disk()
    check_gateway()
    run_checker("mcp-health-check.py", "mcp", "all MCP servers healthy")
    run_checker("discord-bot-health.py", "discord", "bots responding")
    check_backup()
    check_cron()
    check_config_drift()

    n_crit = sum(1 for c in CHECKS if c[1] == "crit")
    n_warn = sum(1 for c in CHECKS if c[1] == "warn")
    verdict = "ALL GREEN" if not (n_crit or n_warn) else (
        f"{n_crit} CRITICAL" if n_crit else f"{n_warn} WARNINGS")

    REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = NOW.strftime("%Y%m%d")
    html_path = REPORTS / f"morning-pulse-{stamp}.html"
    html_path.write_text(build_html(CHECKS), encoding="utf-8")
    if not html_path.is_file() or html_path.stat().st_size == 0:
        print("FAIL: report not written", file=sys.stderr)
        return 1

    lines = [f"🌅 Morning Pulse — {NOW.strftime('%d/%m/%Y')} · {verdict}"]
    if n_crit or n_warn:
        for fam, st, det in CHECKS:
            if st != "ok":
                lines.append(f"{ICON[st]} {fam}: {det}")
    else:
        grouped = " · ".join(f"{fam} {det}" for fam, st, det in CHECKS if st == "ok")
        # keep the green line readable; HTML has everything anyway
        lines.append("🟢 " + grouped[:230])
    lines.append("📊 Full report attached")
    lines.append(f"MEDIA:{html_path}")

    msg = "\n".join(lines)
    sys.stdout.write(msg + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
