#!/usr/bin/env python3
"""Mailbox Cleaner MCP bridge.

Exposes read-only operational controls for the mailbox-cleaner skill so external
MCP clients can inspect health and request dry-run invocations without touching
mail directly. Real mailbox mutations remain owned by the scheduled Hermes cron
jobs and their prompts/approval gates.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

HOME = Path.home()
HERMES_HOME = Path(os.environ.get("HERMES_HOME", HOME / ".hermes"))
SKILL_DIR = HERMES_HOME / "skills" / "mailbox-cleaner"
JOBS_FILE = HERMES_HOME / "cron" / "jobs.json"
CONFIG_FILE = HERMES_HOME / "config.yaml"
REQUIRED_SKILL_FILES = [
    "SKILL.md",
    "main-prompt.md",
    "jobhunt-prompt.md",
    "urgent-detection-prompt.md",
    "reply-parser.md",
    "triage-tests.md",
    "urgent-detection-tests.md",
]
EXPECTED_JOBS = {
    "main": "mailbox-cleaner-main",
    "jobhunt": "mailbox-cleaner-jobhunt",
    "urgent": "mailbox-cleaner-urgent-detector",
}

mcp = FastMCP("mailbox_cleaner")


def _load_jobs() -> list[dict[str, Any]]:
    if not JOBS_FILE.exists():
        return []
    try:
        data = json.loads(JOBS_FILE.read_text())
    except Exception:
        return []
    jobs = data.get("jobs", data if isinstance(data, list) else [])
    return jobs if isinstance(jobs, list) else []


def _job_summary(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": job.get("id"),
        "name": job.get("name"),
        "enabled": job.get("enabled"),
        "schedule": job.get("schedule"),
        "deliver": job.get("deliver"),
        "last_status": job.get("last_status"),
        "last_run_at": job.get("last_run_at"),
        "next_run_at": job.get("next_run_at"),
        "provider": job.get("provider") or (job.get("model") or {}).get("provider"),
        "model": job.get("model") if isinstance(job.get("model"), str) else (job.get("model") or {}).get("model"),
        "enabled_toolsets": job.get("enabled_toolsets"),
    }


def _find_job(flow: str) -> Optional[dict[str, Any]]:
    name = EXPECTED_JOBS.get(flow, flow)
    for job in _load_jobs():
        if job.get("name") == name:
            return job
    return None


@mcp.tool()
def mailbox_cleaner_health() -> Dict[str, Any]:
    """Return mailbox-cleaner operational health: files, crons, MCP config, healthcheck env."""
    missing_files = [name for name in REQUIRED_SKILL_FILES if not (SKILL_DIR / name).exists()]
    jobs = {flow: _job_summary(job) if job else None for flow, job in ((flow, _find_job(flow)) for flow in EXPECTED_JOBS)}
    config_text = CONFIG_FILE.read_text(errors="ignore") if CONFIG_FILE.exists() else ""
    mcp_state = {
        "google_workspace_configured": "google_workspace:" in config_text and "enabled: true" in config_text,
        "outlook_configured": "outlook:" in config_text and "ms-365-mcp-server" in config_text,
        "mailbox_cleaner_bridge_configured": "mailbox_cleaner:" in config_text and "mailbox_cleaner_mcp.py" in config_text,
    }
    env_state = {
        "healthchecks_ping_url_present": bool(os.environ.get("HEALTHCHECKS_PING_URL")),
    }
    healthy = not missing_files and all(jobs.values()) and all(mcp_state.values())
    return {
        "healthy": healthy,
        "skill_dir": str(SKILL_DIR),
        "missing_skill_files": missing_files,
        "jobs": jobs,
        "mcp": mcp_state,
        "env": env_state,
        "note": "This bridge is read-only except dry-run cron triggers; mailbox writes remain gated in cron prompts.",
    }


@mcp.tool()
def mailbox_cleaner_jobs() -> Dict[str, Any]:
    """List mailbox-cleaner cron jobs and schedules."""
    jobs = [_job_summary(job) for job in _load_jobs() if str(job.get("name", "")).startswith("mailbox-cleaner-")]
    return {"count": len(jobs), "jobs": jobs}


@mcp.tool()
def mailbox_cleaner_trigger_dry_run(flow: str) -> Dict[str, Any]:
    """Queue a mailbox-cleaner dry-run cron by flow: main, jobhunt, or urgent.

    The standing cron prompts are configured in dry-run/read-only mode, so this
    queues the same safe workflow that scheduled runs use.
    """
    if flow not in EXPECTED_JOBS:
        return {"ok": False, "error": f"unknown flow {flow!r}; expected one of {sorted(EXPECTED_JOBS)}"}
    job = _find_job(flow)
    if not job:
        return {"ok": False, "error": f"cron job for flow {flow!r} not found"}
    job_id = job.get("id")
    cmd = ["hermes", "cron", "run", str(job_id)]
    try:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
    except Exception as exc:
        return {"ok": False, "job_id": job_id, "error": repr(exc)}
    return {
        "ok": proc.returncode == 0,
        "job_id": job_id,
        "flow": flow,
        "stdout": proc.stdout[-2000:],
        "stderr": proc.stderr[-2000:],
        "returncode": proc.returncode,
    }


if __name__ == "__main__":
    mcp.run()
