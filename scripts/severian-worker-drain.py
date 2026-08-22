#!/usr/bin/env python3
"""Drain the Severian derive-job queue so observations become searchable.

Gateway ingests enqueue derive jobs but nothing drains them; projections
(FTS + Qwen vectors) are only built when this worker runs. Run on a
short interval (no-agent cron) against the shared store. Silent on
success; emits a one-line summary only when it processed jobs or failed.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

STORE = Path(os.environ.get("SEVERIAN_STORAGE", "/home/kensei/.hermes/severian"))
PYTHON = Path(os.environ.get("SEVERIAN_WORKER_PYTHON", "/home/kensei/repos/Severian/.venv/bin/python"))
EMBEDDING = os.environ.get("SEVERIAN_EMBEDDING", "hash")  # must match store vector space (hash/deterministic)
SOCKET = os.environ.get("SEVERIAN_EMBED_SOCKET", "/run/severian/embed.sock")

cmd = [
    str(PYTHON),
    "-m", "severian.cli.app", "work",
    "--database", str(STORE / "severian.db"),
    "--fts", str(STORE / "severian.fts"),
    "--vectors", str(STORE / "severian.vec"),
    "--max-jobs", "200",
]
env = dict(os.environ)
env["SEVERIAN_EMBEDDING"] = EMBEDDING
env["SEVERIAN_EMBED_SOCKET"] = SOCKET
env["SEVERIAN_STORAGE"] = str(STORE)
env["PYTHONDONTWRITEBYTECODE"] = "1"

try:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240, env=env)
except subprocess.TimeoutExpired:
    print("SEVERIAN_WORKER timeout after 240s")
    sys.exit(1)

if proc.returncode != 0:
    print(f"SEVERIAN_WORKER failed rc={proc.returncode}: {proc.stderr.strip()[:500]}")
    sys.exit(1)

try:
    summary = json.loads(proc.stdout.strip().splitlines()[-1])
except Exception:
    # Unknown output shape — not a failure to report; the worker ran.
    sys.exit(0)

processed = summary.get("processed", 0)
failed = summary.get("failed", 0)
projection_failed = summary.get("projection_failed", 0)

# Post-pass health gate: the drain can be green while the store is quietly
# degrading (dead-lettered derive jobs, stranded outbox events, failed
# projections). Alarm on any of these so recovery is same-day, not weeks.
import sqlite3

alerts = []
con = sqlite3.connect(f"file:{STORE / 'severian.db'}?mode=ro", uri=True)
try:
    dl = con.execute(
        "SELECT COUNT(*) FROM records WHERE record_type='job' AND status='dead_letter'"
    ).fetchone()[0]
    pend = con.execute(
        "SELECT COUNT(*) FROM records WHERE record_type='outbox' AND status='pending'"
    ).fetchone()[0]
finally:
    con.close()
if dl > 0:
    alerts.append(f"dead_letter_jobs={dl}")
if pend > 50:
    alerts.append(f"pending_outbox={pend}")
if projection_failed > 0:
    alerts.append("projection_failed=1")

if processed > 0 or failed > 0 or alerts:
    parts = [f"SEVERIAN_WORKER processed={processed} failed={failed}"]
    if alerts:
        parts.append("ALERT: " + ", ".join(alerts))
    print("; ".join(parts))
    sys.exit(1 if alerts else 0)
