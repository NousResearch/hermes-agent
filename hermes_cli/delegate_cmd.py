#!/usr/bin/env python3
"""Operator CLI for async delegation jobs."""

from __future__ import annotations

import json
import time
from typing import Any, Dict


def _fmt_time(ts: Any) -> str:
    try:
        val = float(ts)
    except (TypeError, ValueError):
        return "-"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(val))


def _print_json(data: Dict[str, Any]) -> None:
    print(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False))


def delegate_command(args) -> None:
    from tools import delegation_jobs

    action = getattr(args, "delegate_action", None) or "list"
    if action in ("list", "ls"):
        jobs = delegation_jobs.list_jobs(
            status=getattr(args, "status", None),
            profile=getattr(args, "profile", None),
            parent_session_id=getattr(args, "parent_session_id", None),
            limit=getattr(args, "limit", 50),
        )
        if getattr(args, "json", False):
            _print_json({"jobs": jobs})
            return
        if not jobs:
            print("No delegation jobs found.")
            return
        print(f"{'JOB ID':<28} {'STATUS':<11} {'PROFILE':<12} {'STARTED':<19} GOAL")
        for job in jobs:
            goal = (job.get("goal") or "").replace("\n", " ")
            if len(goal) > 60:
                goal = goal[:57] + "..."
            print(
                f"{job.get('job_id',''):<28} {job.get('status',''):<11} "
                f"{job.get('profile',''):<12} {_fmt_time(job.get('started_at')):<19} {goal}"
            )
        return

    if action == "status":
        job = delegation_jobs.get_job(args.job_id)
        if not job:
            print(f"Job not found: {args.job_id}")
            return
        _print_json(job)
        return

    if action == "wait":
        result = delegation_jobs.wait_jobs([args.job_id], timeout=getattr(args, "timeout", 0))
        _print_json(result)
        return

    if action == "cancel":
        _print_json(delegation_jobs.cancel_jobs([args.job_id]))
        return

    if action == "logs":
        result = delegation_jobs.get_result(args.job_id, tail_chars=getattr(args, "tail_chars", 12000))
        if getattr(args, "json", False):
            _print_json(result)
            return
        stdout = result.get("stdout_tail") or ""
        stderr = result.get("stderr_tail") or ""
        if stdout:
            print("== stdout ==")
            print(stdout.rstrip())
        if stderr:
            print("== stderr ==")
            print(stderr.rstrip())
        if not stdout and not stderr:
            print("No logs yet.")
        return

    raise SystemExit(f"Unknown delegate action: {action}")

