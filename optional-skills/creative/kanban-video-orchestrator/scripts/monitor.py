#!/usr/bin/env python3
"""
Monitor a running video-production kanban by polling the board for a tenant.

This is best-effort observability. It checks for missing heartbeats, overtime,
retries, and long-lived READY queues. It does not auto-restart tasks.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone


def hermes_available() -> bool:
    return shutil.which("hermes") is not None


def parse_ts(value) -> datetime | None:
    """Parse Hermes timestamps (epoch seconds or ISO strings) as UTC."""
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.isdigit():
            return datetime.fromtimestamp(float(text), tz=timezone.utc)
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


def run_json(cmd: list[str]):
    out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if out.returncode != 0:
        return None
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        return None


def kanban_list(tenant: str) -> list[dict]:
    rows = run_json(["hermes", "kanban", "list", "--tenant", tenant, "--json"])
    if isinstance(rows, list):
        return rows
    out = subprocess.run(["hermes", "kanban", "list", "--tenant", tenant], capture_output=True, text=True, check=False)
    parsed = []
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0].startswith("t_"):
            parsed.append({"id": parts[0], "status": parts[1], "assignee": parts[2], "title": " ".join(parts[3:])})
    return parsed


def kanban_show(task_id: str) -> dict | None:
    data = run_json(["hermes", "kanban", "show", task_id, "--json"])
    return data if isinstance(data, dict) else None


def enrich(tasks: list[dict]) -> list[dict]:
    enriched = []
    for t in tasks:
        task_id = t.get("id", "")
        detail = kanban_show(task_id) or {}
        task_detail = detail.get("task") if isinstance(detail.get("task"), dict) else {}
        merged = dict(task_detail or t)
        merged.update({k: v for k, v in t.items() if v is not None})
        runs = detail.get("runs") if isinstance(detail.get("runs"), list) else []
        if runs:
            latest = runs[-1]
            merged["latest_run_status"] = latest.get("status")
            merged["latest_run_outcome"] = latest.get("outcome")
            merged["latest_run_error"] = latest.get("error")
            merged["retries"] = max(0, len(runs) - 1)
            for src_key, dst_key in [
                ("last_heartbeat_at", "heartbeat_at"),
                ("started_at", "started_at"),
                ("max_runtime_seconds", "max_runtime_s"),
            ]:
                if latest.get(src_key) is not None:
                    merged[dst_key] = latest.get(src_key)
        else:
            if merged.get("max_runtime_seconds") is not None:
                merged["max_runtime_s"] = merged.get("max_runtime_seconds")
            if merged.get("last_heartbeat_at") is not None:
                merged["heartbeat_at"] = merged.get("last_heartbeat_at")
            merged.setdefault("retries", 0)
        enriched.append(merged)
    return enriched


def detect_issues(tasks: list[dict]) -> list[str]:
    now = datetime.now(timezone.utc)
    issues = []
    by_status = defaultdict(list)
    for t in tasks:
        by_status[str(t.get("status", "?")).lower()].append(t)

    for t in by_status.get("running", []):
        hb_dt = parse_ts(t.get("heartbeat_at") or t.get("last_heartbeat_at"))
        if hb_dt and now - hb_dt > timedelta(minutes=2):
            issues.append(f"STUCK: {t.get('id')} no heartbeat in {(now - hb_dt).total_seconds():.0f}s")
        started_dt = parse_ts(t.get("started_at"))
        max_rt = t.get("max_runtime_s") or t.get("max_runtime_seconds")
        if started_dt and max_rt:
            try:
                if (now - started_dt).total_seconds() > float(max_rt):
                    issues.append(f"OVERTIME: {t.get('id')} exceeded runtime cap {max_rt}s")
            except (TypeError, ValueError):
                pass

    ready = by_status.get("ready", [])
    if len(ready) >= 5 and not by_status.get("running", []):
        issues.append("QUEUE STALL: many READY tasks but nothing RUNNING; check workers, tenant, or dispatcher")

    for t in tasks:
        retries = t.get("retries", 0) or 0
        try:
            retries = int(retries)
        except Exception:
            retries = 0
        if retries >= 2:
            issues.append(f"FLAPPING: {t.get('id')} retried {retries}x")
    return issues


def print_snapshot(tasks: list[dict], issues: list[str]) -> None:
    counts = defaultdict(int)
    for t in tasks:
        counts[str(t.get("status", "?")).lower()] += 1
    print(f"[{datetime.now().strftime('%H:%M:%S')}] total={len(tasks)} " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    for t in tasks:
        print(f" - {t.get('id','?'):14} {t.get('status','?'):8} {t.get('assignee','?'):18} {str(t.get('title',''))[:60]}")
    if issues:
        print("\nIssues:", file=sys.stderr)
        for issue in issues:
            print(f" - {issue}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    if not hermes_available():
        print("ERROR: 'hermes' CLI not found in PATH", file=sys.stderr)
        sys.exit(1)

    def take_snapshot():
        tasks = enrich(kanban_list(args.tenant))
        issues = detect_issues(tasks)
        print_snapshot(tasks, issues)
        return 0 if not issues else 2

    if args.once:
        sys.exit(take_snapshot())

    try:
        while True:
            take_snapshot()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
