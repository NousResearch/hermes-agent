#!/usr/bin/env python3
"""
System Health Daily — consolidated health check.

Replaces: heartbeat audit (hourly LLM), WFA daily, WFA delta, triage-processor.
          (blocked-task-escalator was paused 31/05 with same intent but
          re-enabled 02/06 — that cron's blocked-task scan is NOT
          subsumed here. See kensei-blocked-task-escalator for the
          dedicated 60-min stale-blocker check.)

Runs daily at 08:00. Checks system resources, cron health, kanban pipeline,
process health. Files kanban tasks for issues. Escalates persistent issues.
Silent if healthy.

Design: one script, one cron, one message per day (or zero).
"""
from __future__ import annotations

import datetime as dt
import html
import json
import os
import sqlite3
import subprocess
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from zoneinfo import ZoneInfo

HERMES = Path(os.environ.get("HERMES_HOME", "/home/kensei/.hermes"))
STATE_FILE = HERMES / "governance" / "system-health-state.json"
LOG_DIR = HERMES / "governance" / "logboard"
JOBS_FILE = HERMES / "cron" / "jobs.json"
CRON_OUTPUT = HERMES / "cron" / "output"

TZ = ZoneInfo("Europe/London")

# Thresholds
MEM_THRESHOLD_MB = 500
DISK_THRESHOLD_PCT = 80
SWAP_THRESHOLD_PCT = 90
ESCALATION_DAYS = 3
CRON_GRACE_MIN = 10
STALE_RUN_DAYS = 3

TERMINAL_STATUSES = {"done", "completed", "archived"}
SAFE_FAILURE_STATUSES = {"blocked", "failed", "triage"}
FAILED_ONLY_OUTCOMES = {"crashed", "reclaimed", "blocked", "timed_out", "spawn_failed", "gave_up"}
SUCCESSFUL_RUN_MARKERS = {"completed", "success", "succeeded", "ok"}

# P13 isolation: --dry-run skips every network/subprocess probe (sudo docker
# inspect, curl, systemctl, free, df) and suppresses every write path
# (create_task kanban filing, save_state, LOG_DIR json write). Read-only
# sqlite scans still run so drift is detected, but nothing is filed.
_DRY_RUN = "--dry-run" in sys.argv


def now() -> dt.datetime:
    return dt.datetime.now(TZ)


def run(cmd: list[str], timeout: int = 20) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except Exception as e:
        return 99, "", str(e)


def create_task(title: str, body: str, assignee: str, priority: str, key: str) -> str | None:
    if _DRY_RUN:
        return None  # dry-run: never file a kanban task
    # Priority: P1=1, P2=2, P3=3
    pnum = {"P1": "1", "P2": "2", "P3": "3"}.get(priority, "2")
    cmd = [
        "hermes", "kanban", "create", title,
        "--body", body,
        "--assignee", assignee,
        "--priority", pnum,
        "--triage",
        "--idempotency-key", key,
        "--created-by", "system-health",
        "--json",
    ]
    code, out, _ = run(cmd, timeout=30)
    if code != 0:
        return None
    try:
        data = json.loads(out)
        return data.get("task_id") or data.get("id")
    except Exception:
        return out.split()[0] if out else None


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"issues": {}, "last_run": None}


def save_state(state: dict) -> None:
    if _DRY_RUN:
        return  # dry-run: never write the state file
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def file_finding(title: str, body: str, priority: str, slug: str) -> tuple[str | None, str, str]:
    """File a kanban task with escalation for persistent issues."""
    key = f"system-health-{slug}"
    state = load_state()
    issue_key = slug

    if issue_key not in state.get("issues", {}):
        state.setdefault("issues", {})[issue_key] = {
            "first_seen": now().isoformat(),
            "last_seen": now().isoformat(),
            "escalated": False,
        }
    else:
        state["issues"][issue_key]["last_seen"] = now().isoformat()
        first = dt.datetime.fromisoformat(state["issues"][issue_key]["first_seen"])
        age_days = (now() - first).days
        if age_days >= ESCALATION_DAYS and not state["issues"][issue_key]["escalated"]:
            priority = "P1"
            body += f"\n\n**ESCALATED:** Issue persists for {age_days} days. Auto-escalated to P1."
            state["issues"][issue_key]["escalated"] = True
            title = f"[ESCALATED] {title}"

    save_state(state)
    task_id = create_task(title, body, "wesker", priority, key)
    return task_id, title, priority


# ────────────────────────── Checks ──────────────────────────


def check_memory() -> dict | None:
    if _DRY_RUN:
        return None  # dry-run: skip free probe
    code, out, _ = run(["free", "-m"], timeout=10)
    if code != 0:
        return {"title": "Memory check failed", "body": f"`free -m` exited {code}.", "priority": "P2", "slug": "memory-check-failed"}
    for line in out.splitlines():
        if line.startswith("Mem:"):
            parts = line.split()
            if len(parts) >= 7:
                avail = int(parts[6])
                total = int(parts[1])
                if avail < MEM_THRESHOLD_MB:
                    pct = int((total - avail) / total * 100)
                    return {
                        "title": f"Low memory: {avail}MB available",
                        "body": f"Available: {avail}MB (threshold {MEM_THRESHOLD_MB}MB). Total: {total}MB, usage: {pct}%.",
                        "priority": "P2",
                        "slug": "low-memory",
                    }
    return None


def check_disk() -> dict | None:
    if _DRY_RUN:
        return None  # dry-run: skip df probe
    code, out, _ = run(["df", "-h", "/"], timeout=10)
    if code != 0:
        return {"title": "Disk check failed", "body": f"`df -h /` exited {code}.", "priority": "P2", "slug": "disk-check-failed"}
    for line in out.splitlines():
        if line.startswith("/"):
            parts = line.split()
            if len(parts) >= 5:
                pct = int(parts[4].rstrip("%"))
                if pct > DISK_THRESHOLD_PCT:
                    return {
                        "title": f"Disk usage {pct}%",
                        "body": f"Root disk at {pct}% (threshold {DISK_THRESHOLD_PCT}%).",
                        "priority": "P2",
                        "slug": "disk-usage-high",
                    }
    return None


def check_swap() -> dict | None:
    if _DRY_RUN:
        return None  # dry-run: skip free probe
    code, out, _ = run(["free", "-m"], timeout=10)
    if code != 0:
        return None
    for line in out.splitlines():
        if line.startswith("Swap:"):
            parts = line.split()
            if len(parts) >= 3:
                total = int(parts[1])
                used = int(parts[2])
                if total > 0:
                    pct = int(used / total * 100)
                    if pct > SWAP_THRESHOLD_PCT:
                        return {
                            "title": f"Swap {pct}% full",
                            "body": f"Swap: {used}MB/{total}MB ({pct}%).",
                            "priority": "P2",
                            "slug": "swap-high",
                        }
    return None


def check_cron_gaps() -> dict | None:
    try:
        with open(JOBS_FILE, "r") as f:
            data = json.load(f)
    except Exception:
        return None
    jobs = data.get("jobs", [])
    gaps = []
    grace = dt.timedelta(minutes=CRON_GRACE_MIN)
    n = now()
    for j in jobs:
        if not j.get("enabled"):
            continue
        schedule = j.get("schedule", {})
        expr = schedule.get("expr", "") if isinstance(schedule, dict) else str(schedule)
        if not expr or "*/" in expr or "interval" in str(schedule):
            continue
        next_run = j.get("next_run_at")
        if not next_run:
            continue
        try:
            next_dt = dt.datetime.fromisoformat(next_run)
            if n > next_dt + grace:
                name = j.get("name", "?")
                overdue = int((n - next_dt).total_seconds() / 60)
                gaps.append(f"`{name}` — overdue by {overdue}m")
        except Exception:
            continue
    if gaps:
        return {
            "title": f"{len(gaps)} cron(s) overdue",
            "body": "Overdue crons:\n" + "\n".join(gaps[:6]),
            "priority": "P2",
            "slug": "cron-gaps",
        }
    return None


def check_cron_errors() -> dict | None:
    try:
        with open(JOBS_FILE, "r") as f:
            data = json.load(f)
    except Exception:
        return None
    jobs = data.get("jobs", [])
    failed = [j for j in jobs if j.get("enabled") and j.get("last_status") == "error"]
    if failed:
        # The title and body must describe the same authoritative set.  A prior
        # five-row display cap produced alerts titled "7 cron(s) failed" while
        # silently omitting two jobs from the body.
        lines = [f"`{j.get('name')}` — last_status error" for j in failed]
        return {
            "title": f"{len(failed)} cron(s) failed last run",
            "body": "Failed crons:\n" + "\n".join(lines),
            "priority": "P2",
            "slug": "cron-errors",
        }
    return None


def check_stale_crons() -> dict | None:
    """Crons whose last run is older than expected schedule window."""
    try:
        with open(JOBS_FILE, "r") as f:
            data = json.load(f)
    except Exception:
        return None
    jobs = data.get("jobs", [])
    stale = []
    n = now()
    for j in jobs:
        if not j.get("enabled"):
            continue
        last = j.get("last_run_at")
        if not last:
            # Never run — but skip jobs whose first fire time has not arrived yet.
            # Two flavours of "schedule hasn't fired yet":
            #   (a) one-shot with future run_at, or
            #   (b) recurring cron with next_run_at still in the future
            #       (e.g. a weekly job created mid-week after the weekly fire time).
            schedule = j.get("schedule", {})
            if isinstance(schedule, dict) and schedule.get("kind") == "once":
                run_at = schedule.get("run_at", "")
                if run_at:
                    try:
                        run_dt = dt.datetime.fromisoformat(run_at)
                        if run_dt > n:
                            continue  # Future one-shot — not stale yet
                    except (ValueError, TypeError):
                        pass
            elif isinstance(schedule, dict):
                # Recurring cron: check stored next_run_at. If it is in the
                # future, the schedule has not yet had a chance to fire — the
                # job is not stale, only the watchdog's clock-vs-created_at
                # metric is misleading.
                next_run = j.get("next_run_at", "")
                if next_run:
                    try:
                        next_dt = dt.datetime.fromisoformat(next_run)
                        if next_dt > n:
                            continue  # First scheduled fire still in the future
                    except (ValueError, TypeError):
                        pass
            # Never run but enabled > 24h
            created = j.get("created_at", "")
            if created:
                try:
                    created_dt = dt.datetime.fromisoformat(created)
                    if (n - created_dt).days > 1:
                        stale.append(f"`{j.get('name')}` — never run, created {(n - created_dt).days}d ago")
                except (ValueError, TypeError):
                    pass
            continue
        try:
            last_dt = dt.datetime.fromisoformat(last)
            age_days = (n - last_dt).days
            schedule = j.get("schedule", {})
            if isinstance(schedule, dict):
                kind = schedule.get("kind", "")
                # One-shot scheduled in future — not stale
                if kind == "once":
                    run_at = schedule.get("run_at", "")
                    if run_at:
                        run_dt = dt.datetime.fromisoformat(run_at)
                        if run_dt > n:
                            continue
                # Use actual schedule interval for threshold
                expr = schedule.get("expr", "")
                display = schedule.get("display", "")
                # Weekly: stale after 10 days, Monthly: after 35, Quarterly: after 100
                if "quarterly" in display.lower() or "quarterly" in j.get("name","").lower():
                    threshold = 100
                elif "monthly" in display.lower() or "monthly" in j.get("name","").lower():
                    threshold = 35
                elif "weekly" in display.lower() or "weekly" in j.get("name","").lower():
                    threshold = 10
                elif expr:
                    # Parse cron expr by cadence. A weekly job (0 9 * * 1) and a
                    # daily job (0 9 * * *) both have 5 fields, so field COUNT is
                    # not enough — inspect day-of-week and day-of-month.
                    parts = expr.split()
                    if len(parts) == 5:
                        _min, hour, dom, mon, dow = parts
                        if dow != "*":
                            threshold = 10  # weekly (fires on a specific weekday)
                        elif dom != "*":
                            threshold = 100 if mon != "*" else 35  # quarterly / monthly
                        elif any(c in hour for c in "*/,-"):
                            threshold = 1  # sub-daily / hourly
                        else:
                            threshold = 3  # daily
                    else:
                        threshold = 7
                else:
                    threshold = 7
            else:
                threshold = 7
            if age_days > threshold:
                stale.append(f"`{j.get('name')}` — last run {age_days}d ago")
        except Exception:
            continue
    if stale:
        return {
            "title": f"{len(stale)} cron(s) stale",
            "body": "Stale crons:\n" + "\n".join(stale[:6]),
            "priority": "P2",
            "slug": "cron-stale",
        }
    return None


def check_gateway() -> dict | None:
    if _DRY_RUN:
        return None  # dry-run: skip systemctl/pgrep probes
    code, out, _ = run(["systemctl", "is-active", "hermes-gateway"], timeout=10)
    if code != 0 or "active" not in out:
        return {
            "title": "Gateway not active",
            "body": f"`systemctl is-active hermes-gateway` returned: `{out}`",
            "priority": "P1",
            "slug": "gateway-inactive",
        }
    # Orphan count
    code, out, _ = run(["pgrep", "-f", "hermes_cli.main gateway"], timeout=10)
    if code == 0:
        pids = [p for p in out.splitlines() if p.strip()]
        if len(pids) > 14:  # 1 primary + ~12 profiles + margin
            return {
                "title": f"{len(pids)} gateway processes (orphans likely)",
                "body": f"Expected ~12-13, found {len(pids)}. Orphan watchdog may be failing.",
                "priority": "P2",
                "slug": "gateway-orphans",
            }
    return None


def check_kanban() -> dict | None:
    if _DRY_RUN:
        return None  # dry-run: skip _board_compat (hermes_cli import chain)
    # W1-G (Batch 1): board DB identities resolved via _board_compat.
    # Graceful degradation: in a bare no-agent bundle without the repo on
    # sys.path, hermes_cli is unavailable inside _board_compat — skip the
    # board checks rather than crash (ModuleNotFoundError is an ImportError).
    try:
        import _board_compat
        raw_boards = _board_compat.build_board_db_map([
            "ops", "research", "apps", "content-lead", "default",
        ])
    except ImportError:
        return None
    boards: dict[str, Path] = {}
    for slug, configured_path in raw_boards.items():
        if configured_path is None or (
            isinstance(configured_path, str) and not configured_path.strip()
        ):
            continue
        try:
            boards[slug] = Path(configured_path).expanduser()
        except (TypeError, ValueError, OSError, RuntimeError):
            continue

    triage_total = 0
    blocked_total = 0
    stale_total = 0
    stale_ts = int((now() - dt.timedelta(days=STALE_RUN_DAYS)).timestamp())

    for name, db_path in boards.items():
        if not db_path.exists():
            continue
        try:
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM tasks WHERE status = 'triage'")
            triage_total += cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM tasks WHERE status = 'blocked'")
            blocked_total += cur.fetchone()[0]
            # Derive staleness from task_events.created_at (exists on ALL boards).
            # Fall back to tasks.created_at for tasks with no events.
            cur.execute("""
                SELECT COUNT(*) FROM tasks t
                WHERE t.status NOT IN ('done','archived','completed')
                  AND COALESCE(
                    (SELECT MAX(created_at) FROM task_events WHERE task_id = t.id),
                    t.created_at
                  ) < ?
            """, (stale_ts,))
            stale_total += cur.fetchone()[0]
            conn.close()
        except Exception:
            continue

    issues = []
    if triage_total > 0:
        issues.append(f"{triage_total} triage task(s) waiting classification")
    if blocked_total > 5:
        issues.append(f"{blocked_total} blocked task(s)")
    if stale_total > 0:
        issues.append(f"{stale_total} stale task(s) (>3d no update)")

    if issues:
        return {
            "title": f"Kanban: {len(issues)} issue(s)",
            "body": "\n".join(issues),
            "priority": "P2",
            "slug": "kanban-health",
        }
    return None


def check_wfa_live() -> dict | None:
    """Worker Failure Analysis — live tasks only. Historical drift ignored."""
    if _DRY_RUN:
        return None  # dry-run: skip _board_compat (hermes_cli import chain)
    # W1-G (Batch 1): board DB identities resolved via _board_compat.
    # Graceful degradation: in a bare no-agent bundle without the repo on
    # sys.path, hermes_cli is unavailable inside _board_compat — skip the
    # WFA checks rather than crash (ModuleNotFoundError is an ImportError).
    try:
        import _board_compat
        _wfa_slugs = ["default", "ops", "research", "apps", "content-lead"]
        dbs = [(s, _board_compat.resolve_board_db(s)) for s in _wfa_slugs]
    except ImportError:
        return None

    live_findings = []
    for board, db_path in dbs:
        if not db_path.exists():
            continue
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT id, title, assignee, status, result, skills FROM tasks WHERE status NOT IN ('done','completed','archived')")
            for row in cur.fetchall():
                task_id = row["id"]
                status = (row["status"] or "").lower()
                # Check runs
                cur.execute("SELECT status, outcome FROM task_runs WHERE task_id = ? ORDER BY started_at", (task_id,))
                runs = cur.fetchall()
                run_outcomes = [r["outcome"] or r["status"] for r in runs]

                # Terminal without completed run
                if status in TERMINAL_STATUSES:
                    completed = [r for r in runs if (r["outcome"] or r["status"]) in SUCCESSFUL_RUN_MARKERS]
                    if not completed:
                        live_findings.append({
                            "task": task_id,
                            "board": board,
                            "kind": "terminal_without_completed_run",
                            "title": row["title"],
                        })

                # All runs crashed
                if runs and all((r["outcome"] == "crashed" or r["status"] == "crashed") for r in runs):
                    if status not in SAFE_FAILURE_STATUSES:
                        live_findings.append({
                            "task": task_id,
                            "board": board,
                            "kind": "all_runs_crashed",
                            "title": row["title"],
                        })

                # Result claims completion but only failed runs
                result = row["result"] or ""
                if result and any(w in result.lower() for w in ("complete", "done", "resolved", "fixed", "shipped", "success")):
                    if runs and all((r["outcome"] or r["status"]) in FAILED_ONLY_OUTCOMES for r in runs):
                        completed = [r for r in runs if (r["outcome"] or r["status"]) in SUCCESSFUL_RUN_MARKERS]
                        if not completed:
                            live_findings.append({
                                "task": task_id,
                                "board": board,
                                "kind": "result_claims_complete_but_runs_failed",
                                "title": row["title"],
                            })
            conn.close()
        except Exception:
            continue

    if live_findings:
        by_kind = Counter(f["kind"] for f in live_findings)
        lines = [f"{k}: {v}" for k, v in by_kind.most_common()]
        # Show top 5 examples
        examples = live_findings[:5]
        detail = "\n".join(f"- {f['board']}:{f['task'][:10]} {f['kind']} · {f['title'][:60]}" for f in examples)
        return {
            "title": f"WFA: {len(live_findings)} live task/run drift",
            "body": f"By type:\n" + "\n".join(lines) + f"\n\nTop examples:\n{detail}",
            "priority": "P1" if any(f["kind"] == "terminal_without_completed_run" for f in live_findings) else "P2",
            "slug": "wfa-live-drift",
        }
    return None


def check_discord_bots() -> dict | None:
    """Check all Discord gateway bot services are active."""
    if _DRY_RUN:
        return None  # dry-run: skip systemctl probes
    SERVICES = [
        ("kensei", "hermes-gateway.service"),
        ("ceecee", "hermes-gateway-ceecee.service"),
        ("denji", "hermes-gateway-denji.service"),
        ("dezzy", "hermes-gateway-dezzy.service"),
        ("gojo", "hermes-gateway-gojo.service"),
        ("light", "hermes-gateway-light.service"),
        ("misa-misa", "hermes-gateway-misa-misa.service"),
        ("mrhermagi", "hermes-gateway-mrhermagi.service"),
        ("octacon", "hermes-gateway-octacon.service"),
        ("remii", "hermes-gateway-remii.service"),
        ("wesker", "hermes-gateway-wesker.service"),
    ]
    down = []
    for name, service in SERVICES:
        code, out, _ = run(["systemctl", "is-active", service], timeout=10)
        if code != 0 or "active" not in out:
            down.append(name)
    if down:
        return {
            "title": f"Discord bots: {len(down)} down ({', '.join(down)})",
            "body": f"Down: {', '.join(down)}. Check `systemctl status hermes-gateway-<name>.service`.",
            "priority": "P2",
            "slug": "discord-bots-down",
        }
    return None


def _http_probe(url: str, timeout: int) -> tuple[int | None, str, str]:
    """Return HTTP status/body/error without shelling out or requiring root."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read(1_000_000).decode("utf-8", errors="replace")
            return response.status, body, ""
    except urllib.error.HTTPError as exc:
        body = exc.read(1_000_000).decode("utf-8", errors="replace")
        return exc.code, body, str(exc)
    except Exception as exc:
        return None, "", str(exc)


def check_web_backends() -> dict | None:
    """Check SearXNG + GroktoCrawl + DDGS health without Docker access."""
    if _DRY_RUN:
        return None  # dry-run: skip network/DDGS probes

    failures = []

    # SearXNG: probe the actual JSON search endpoint, bounded by urllib.
    status, body, error = _http_probe(
        "http://127.0.0.1:8082/search?q=health+check&format=json", timeout=5
    )
    if status != 200:
        detail = f"status {status}" if status is not None else (error or "not responding")
        failures.append(f"SearXNG API {detail}")
    else:
        try:
            data = json.loads(body)
            if len(data.get("results", [])) == 0:
                failures.append("SearXNG returning 0 results")
        except Exception:
            failures.append("SearXNG invalid JSON response")

    # GroktoCrawl: bounded health endpoint; no container inspection required.
    status, body, error = _http_probe("http://127.0.0.1:8090/health", timeout=10)
    if status != 200 or not body:
        detail = f"status {status}" if status is not None else (error or "not responding")
        failures.append(f"GroktoCrawl health {detail}")

    # DDGS is optional; retain the existing subprocess boundary but route it
    # through the common bounded runner so tests and failures cannot block the
    # system-health process.
    code, _, stderr = run(
        [
            "python3",
            "-c",
            "from ddgs import DDGS; ddgs=DDGS(); r=list(ddgs.text('test', max_results=1)); exit(0 if len(r)>0 else 1)",
        ],
        timeout=15,
    )
    if code != 0:
        failures.append(f"DDGS: {stderr[:60]}" if stderr else "DDGS import/search failed")

    if failures:
        priority = "P1" if len(failures) >= 2 else "P2"
        return {
            "title": f"Web backends: {len(failures)} failure(s)",
            "body": "\n".join(f"- {f}" for f in failures),
            "priority": priority,
            "slug": "web-backend-failures",
        }
    return None


# ────────────────────────── Main ──────────────────────────


def main() -> int:
    n = now()
    findings: list[dict] = []
    filed: list[tuple[str | None, str, str]] = []

    checks = [
        ("Memory", check_memory),
        ("Disk", check_disk),
        ("Swap", check_swap),
        ("Cron gaps", check_cron_gaps),
        ("Cron errors", check_cron_errors),
        ("Stale crons", check_stale_crons),
        ("Gateway", check_gateway),
        ("Kanban", check_kanban),
        ("WFA live", check_wfa_live),
        ("Discord bots", check_discord_bots),
        ("Web backends", check_web_backends),
    ]

    for name, check_fn in checks:
        result = check_fn()
        if result:
            findings.append(result)

    # File tasks
    for finding in findings:
        task_id, title, priority = file_finding(
            finding["title"], finding["body"], finding["priority"], finding["slug"]
        )
        filed.append((task_id, title, priority))

    # Update state
    state = load_state()
    state["last_run"] = n.isoformat()
    save_state(state)

    # Silent if healthy
    if not findings:
        return 0

    # Discord output
    stamp = n.strftime("%d/%m/%Y %H:%M:%S")
    print(f"System Health · {stamp}")
    p1_count = sum(1 for _, _, p in filed if p == "P1")
    p2_count = sum(1 for _, _, p in filed if p == "P2")
    print(f"Findings: {len(findings)} · P1: {p1_count} · P2: {p2_count}")
    for task_id, title, priority in filed:
        tid = task_id or "(failed to file)"
        print(f"• `{tid}` {title} · {priority}")

    # JSON log (suppressed under --dry-run to avoid touching the live
    # governance logboard in a disposable run)
    if not _DRY_RUN:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        logfile = LOG_DIR / f"system-health-{n.strftime('%Y%m%d-%H%M%S')}.json"
        payload = {
            "timestamp": n.isoformat(),
            "findings": findings,
            "filed": [{"task_id": t, "title": tt, "priority": p} for t, tt, p in filed],
        }
        logfile.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"JSON: {logfile}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        n = now()
        try:
            create_task(
                "[CRITICAL] System health script crashed",
                f"Script failed at {n.strftime('%d/%m/%Y %H:%M:%S')}: {type(e).__name__}: {e}",
                "wesker",
                "P1",
                f"system-health-crash-{n.strftime('%Y%m%d')}",
            )
        except Exception:
            pass
        print(f"System Health CRASH · {n.strftime('%d/%m/%Y %H:%M:%S')}: {type(e).__name__}: {e}")
        raise SystemExit(1)
