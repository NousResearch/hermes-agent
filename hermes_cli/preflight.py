"""Preflight checks for long-running dev and Hermes maintenance tasks."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from hermes_cli.runtime_safety import build_runtime_safety_report


@dataclass
class PreflightCheck:
    name: str
    ok: bool
    severity: str
    detail: str


@dataclass
class PreflightReport:
    ok: bool
    mode: str
    classification: str
    checks: list[PreflightCheck]

    def to_json(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "mode": self.mode,
            "classification": self.classification,
            "checks": [asdict(check) for check in self.checks],
        }


def _run(args: list[str], timeout: float = 3.0) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def classify_workspace(cwd: str | Path | None = None) -> str:
    path = Path(cwd or os.getcwd()).resolve()
    try:
        repo = Path(__file__).resolve().parents[1]
        if path == repo or repo in path.parents:
            return "hermes-maintenance"
    except Exception:
        pass
    return "app-dev"


def check_gateway() -> PreflightCheck:
    if shutil.which("systemctl") is None:
        return PreflightCheck("gateway", True, "info", "systemctl unavailable; skipped")
    try:
        result = _run([
            "systemctl", "--user", "show", "hermes-gateway",
            "-p", "ActiveState", "-p", "SubState", "-p", "TimeoutStopUSec", "--no-pager",
        ])
    except Exception as exc:
        return PreflightCheck("gateway", False, "blocker", f"could not query gateway: {exc}")
    out = result.stdout or ""
    ok = result.returncode == 0 and "ActiveState=active" in out and "SubState=running" in out
    detail = "active/running" if ok else "gateway not active/running"
    if "TimeoutStopUSec=" in out:
        timeout_line = next((line for line in out.splitlines() if line.startswith("TimeoutStopUSec=")), "")
        if timeout_line:
            detail += f" {timeout_line}"
    return PreflightCheck("gateway", ok, "blocker" if not ok else "ok", detail)


def check_runtime_safety() -> list[PreflightCheck]:
    report = build_runtime_safety_report()
    counts = report["counts"]
    checks = [
        PreflightCheck(
            "unsafe_due_cron",
            counts["unsafe_gateway_control_due"] == 0,
            "blocker" if counts["unsafe_gateway_control_due"] else "ok",
            f"due unsafe gateway-control cron jobs: {counts['unsafe_gateway_control_due']}",
        ),
        PreflightCheck(
            "stale_resume_pending",
            counts["stale_resume_pending"] == 0,
            "blocker" if counts["stale_resume_pending"] else "ok",
            f"stale resume_pending flags: {counts['stale_resume_pending']}",
        ),
    ]
    scheduled = counts["unsafe_gateway_control_scheduled"] + counts["gateway_control_adjacent"]
    checks.append(
        PreflightCheck(
            "scheduled_gateway_control",
            True,
            "warn" if scheduled else "ok",
            f"scheduled/adjacent gateway-control cron jobs: {scheduled}",
        )
    )
    return checks


def check_kanban() -> PreflightCheck:
    home = get_hermes_home()
    candidates = [home / "kanban.db"]
    board = os.environ.get("HERMES_KANBAN_BOARD")
    if board:
        candidates.insert(0, Path(board))
    existing = [p for p in candidates if p.exists()]
    if not existing:
        return PreflightCheck("kanban", True, "info", "no kanban db found; skipped")
    bad = []
    for path in existing:
        try:
            conn = sqlite3.connect(str(path))
            value = conn.execute("PRAGMA integrity_check").fetchone()[0]
            conn.close()
            if value != "ok":
                bad.append(f"{path}: {value}")
        except Exception as exc:
            bad.append(f"{path}: {exc}")
    return PreflightCheck("kanban", not bad, "blocker" if bad else "ok", "; ".join(bad) if bad else "integrity ok")


def check_disk_memory() -> list[PreflightCheck]:
    usage = shutil.disk_usage(str(get_hermes_home()))
    disk_ok = usage.free >= 5 * 1024 * 1024 * 1024
    checks = [PreflightCheck("disk", disk_ok, "blocker" if not disk_ok else "ok", f"free_gb={usage.free / (1024**3):.1f}")]
    mem_available_kb = None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                mem_available_kb = int(line.split()[1])
                break
    except Exception:
        pass
    if mem_available_kb is None:
        checks.append(PreflightCheck("memory", True, "info", "MemAvailable unavailable; skipped"))
    else:
        mem_ok = mem_available_kb >= 1_000_000
        checks.append(PreflightCheck("memory", mem_ok, "blocker" if not mem_ok else "ok", f"available_mb={mem_available_kb // 1024}"))
    return checks


def build_preflight_report(mode: str = "app-dev") -> PreflightReport:
    classification = classify_workspace()
    checks: list[PreflightCheck] = []
    checks.append(check_gateway())
    checks.extend(check_runtime_safety())
    checks.append(check_kanban())
    checks.extend(check_disk_memory())
    if mode == "hermes-maintenance" and classification != "hermes-maintenance":
        checks.append(PreflightCheck("workspace_mode", False, "blocker", f"cwd classified as {classification}, expected hermes-maintenance"))
    else:
        checks.append(PreflightCheck("workspace_mode", True, "ok", f"cwd classified as {classification}"))
    ok = all(check.ok or check.severity in {"warn", "info"} for check in checks)
    return PreflightReport(ok=ok, mode=mode, classification=classification, checks=checks)


def run_preflight(args: argparse.Namespace) -> int:
    mode = getattr(args, "mode", "app-dev") or "app-dev"
    report = build_preflight_report(mode=mode)
    if getattr(args, "json", False):
        print(json.dumps(report.to_json(), indent=2, sort_keys=True))
    else:
        print(f"Hermes preflight: {'PASS' if report.ok else 'FAIL'} mode={report.mode} classification={report.classification}")
        for check in report.checks:
            marker = "✓" if check.ok else "✗"
            print(f"  {marker} {check.name}: {check.detail}")
    return 0 if report.ok else 1
