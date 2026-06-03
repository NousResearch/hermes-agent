#!/usr/bin/env python3
"""Read-only Hermes reliability diagnostics.

This script intentionally reports counts, paths, and configuration markers only.
It does not restart services, send messages, inspect private message content, or
call cloud storage APIs.
"""

from __future__ import annotations

import argparse
import configparser
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_EXPECTED_CHECKOUT = "/home/jenny/.hermes/hermes-context-routing-e1d-integration"
DEFAULT_HERMES_HOME = "/home/jenny/.hermes"
DEFAULT_AI_OPS_BRAIN = "/home/jenny/ai-ops-brain"
DEFAULT_GATEWAY_VENV_PYTHON = "/home/jenny/.hermes/hermes-agent-backup/venv/bin/python"
DEFAULT_SERVICE = "hermes-gateway.service"
STALE_ACTIVE_FOREGROUND_SECONDS = 6 * 60 * 60


def _run_command(args: list[str], timeout: float = 3.0, cwd: str | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
    except FileNotFoundError:
        return {"ok": False, "returncode": None, "stdout": "", "stderr": "not found"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "returncode": None, "stdout": "", "stderr": "timeout"}
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "").strip(),
        "stderr": (proc.stderr or "").strip(),
    }


def _same_path(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    return str(Path(left).expanduser().resolve(strict=False)) == str(
        Path(right).expanduser().resolve(strict=False)
    )


def parse_systemctl_show(output: str) -> dict[str, str | None]:
    values: dict[str, str | None] = {
        "WorkingDirectory": None,
        "ExecStart": None,
        "MainPID": None,
    }
    for line in (output or "").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in values:
            values[key] = value.strip() or None
    return values


def get_service_state(service: str = DEFAULT_SERVICE) -> dict[str, Any]:
    result = _run_command(
        [
            "systemctl",
            "--user",
            "show",
            service,
            "-p",
            "WorkingDirectory",
            "-p",
            "ExecStart",
            "-p",
            "MainPID",
            "--no-pager",
        ]
    )
    parsed = parse_systemctl_show(result["stdout"])
    return {
        "service": service,
        "working_directory": parsed.get("WorkingDirectory"),
        "exec_start": parsed.get("ExecStart"),
        "main_pid": parsed.get("MainPID"),
        "command_ok": result["ok"],
        "error": result["stderr"] or None,
    }


def get_proc_cwd(pid: str | int | None) -> str | None:
    pid_text = str(pid or "").strip()
    if not pid_text or pid_text == "0":
        return None
    try:
        return str(Path(f"/proc/{pid_text}/cwd").resolve(strict=True))
    except OSError:
        return None


def resolve_module_paths(
    python_executable: str = DEFAULT_GATEWAY_VENV_PYTHON,
    modules: tuple[str, ...] = ("hermes_cli.main", "gateway.run"),
    cwd: str | None = None,
) -> dict[str, str | None]:
    code = (
        "import importlib.util, json; "
        f"mods={modules!r}; "
        "print(json.dumps({m: ((importlib.util.find_spec(m).origin) "
        "if importlib.util.find_spec(m) else None) for m in mods}, sort_keys=True))"
    )
    result = _run_command([python_executable, "-c", code], timeout=5.0, cwd=cwd)
    if not result["ok"]:
        return {module: None for module in modules}
    try:
        parsed = json.loads(result["stdout"])
    except json.JSONDecodeError:
        return {module: None for module in modules}
    return {module: parsed.get(module) for module in modules}


def get_cli_project(hermes_bin: str | None = None) -> dict[str, str | None]:
    hermes_path = hermes_bin or shutil.which("hermes")
    if not hermes_path:
        return {"binary": None, "project": None, "version_output": None}
    result = _run_command([hermes_path, "--version"], timeout=8.0)
    project = None
    for line in result["stdout"].splitlines():
        if line.startswith("Project:"):
            project = line.split(":", 1)[1].strip() or None
            break
    return {
        "binary": hermes_path,
        "project": project,
        "version_output": result["stdout"] if result["ok"] else None,
    }


def evaluate_runtime_topology(
    *,
    expected_runtime_checkout: str,
    service_working_directory: str | None,
    proc_cwd: str | None,
    module_paths: dict[str, str | None],
    cli_project: str | None,
) -> dict[str, Any]:
    expected = str(Path(expected_runtime_checkout).expanduser().resolve(strict=False))
    module_matches = {
        name: bool(path and _same_path(Path(path).parents[1], expected))
        for name, path in module_paths.items()
    }
    service_matches = _same_path(service_working_directory, expected)
    process_matches = _same_path(proc_cwd, expected)
    modules_match = bool(module_matches) and all(module_matches.values())
    cli_matches = _same_path(cli_project, expected) if cli_project else None
    split_brain = any(
        value is False
        for value in (service_matches, process_matches, modules_match, cli_matches)
        if value is not None
    )
    return {
        "expected_runtime_checkout": expected,
        "service_matches_expected": service_matches,
        "process_matches_expected": process_matches,
        "module_matches": module_matches,
        "modules_match_expected": modules_match,
        "cli_project_matches_expected": cli_matches,
        "split_brain_risk": split_brain,
    }


def inspect_active_task_store(path: str | Path) -> dict[str, Any]:
    store_path = Path(path)
    result: dict[str, Any] = {
        "exists": store_path.exists(),
        "parsed": False,
        "record_count": 0,
        "foreground_count": 0,
        "status_counts": {},
        "mode_counts": {},
        "final_report_counts": {},
        "foreground_missing_task_count": 0,
        "stale_active_foreground_count": 0,
        "task_contract_count": 0,
        "task_contract_missing_summary_count": 0,
        "task_contract_status_counts": {},
        "stale_or_superseded_task_contract_count": 0,
        "task_contracts_with_intended_repo_count": 0,
        "task_contracts_with_restart_policy_count": 0,
        "updated_at_age_buckets": {},
    }
    if not store_path.exists():
        return result
    try:
        data = json.loads(store_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return result
    if not isinstance(data, dict):
        return result
    records = [record for record in data.values() if isinstance(record, dict)]
    result["parsed"] = True
    result["record_count"] = len(records)
    result["foreground_count"] = sum(
        1 for record in records if record.get("mode") == "foreground_session"
    )
    result["status_counts"] = dict(
        sorted(Counter(str(record.get("status") or "unknown") for record in records).items())
    )
    result["mode_counts"] = dict(
        sorted(Counter(str(record.get("mode") or "unknown") for record in records).items())
    )
    final_statuses = [
        str(record.get("final_report_status"))
        for record in records
        if record.get("final_report_status")
    ]
    result["final_report_counts"] = dict(sorted(Counter(final_statuses).items()))
    contract_records = [
        record
        for record in records
        if isinstance(record.get("task_contract"), dict)
    ]
    contract_statuses: Counter[str] = Counter()
    for record in contract_records:
        contract = record.get("task_contract") or {}
        status = str(contract.get("status") or "active")
        contract_statuses[status] += 1
        if not (contract.get("task_summary_safe") or record.get("task_summary_safe")):
            result["task_contract_missing_summary_count"] += 1
        if status in {"completed", "superseded"}:
            result["stale_or_superseded_task_contract_count"] += 1
        if contract.get("intended_repo"):
            result["task_contracts_with_intended_repo_count"] += 1
        if contract.get("restart_policy"):
            result["task_contracts_with_restart_policy_count"] += 1
    result["task_contract_count"] = len(contract_records)
    result["task_contract_status_counts"] = dict(sorted(contract_statuses.items()))
    now = datetime.now(timezone.utc)
    age_buckets: Counter[str] = Counter()
    for record in records:
        is_foreground = record.get("mode") == "foreground_session"
        is_active = str(record.get("status") or "") == "active"
        if is_foreground and not (record.get("task_summary") or record.get("command")):
            result["foreground_missing_task_count"] += 1
        raw_updated_at = str(record.get("updated_at") or "")
        try:
            updated_at = datetime.fromisoformat(raw_updated_at)
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=timezone.utc)
            age_seconds = (now - updated_at).total_seconds()
        except ValueError:
            age_buckets["bad_or_missing"] += 1
            if is_foreground and is_active:
                result["stale_active_foreground_count"] += 1
            continue
        if age_seconds > STALE_ACTIVE_FOREGROUND_SECONDS:
            age_buckets["stale"] += 1
            if is_foreground and is_active:
                result["stale_active_foreground_count"] += 1
        else:
            age_buckets["fresh"] += 1
    result["updated_at_age_buckets"] = dict(sorted(age_buckets.items()))
    return result


def inspect_goal_store(state_db_path: str | Path) -> dict[str, Any]:
    db_path = Path(state_db_path)
    result = {
        "exists": db_path.exists(),
        "readable": False,
        "goal_count": 0,
        "active_count": 0,
        "done_count": 0,
        "paused_count": 0,
        "status_counts": {},
        "field_presence": {},
    }
    if not db_path.exists():
        return result
    uri = f"file:{db_path}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
        rows = conn.execute(
            "SELECT value FROM state_meta WHERE key LIKE 'goal:%'"
        ).fetchall()
    except sqlite3.Error:
        return result
    finally:
        try:
            conn.close()  # type: ignore[name-defined]
        except Exception:
            pass
    result["readable"] = True
    result["goal_count"] = len(rows)
    statuses: Counter[str] = Counter()
    fields: Counter[str] = Counter()
    for (raw_value,) in rows:
        try:
            payload = json.loads(raw_value or "{}")
        except json.JSONDecodeError:
            payload = {}
        raw_status = payload.get("status")
        if raw_status:
            status = str(raw_status)
        elif payload.get("done"):
            status = "done"
        elif payload.get("active"):
            status = "active"
        else:
            status = "paused"
        statuses[status] += 1
        for field in (
            "goal",
            "status",
            "created_at",
            "updated_at",
            "last_turn_at",
            "turns_used",
            "max_turns",
            "subgoals",
            "task_summary",
            "intended_repo",
            "intended_branch",
            "expected_path",
            "active_task",
            "final_report",
        ):
            if field in payload and payload.get(field) not in (None, "", []):
                fields[field] += 1
        if status == "active":
            result["active_count"] += 1
        if status == "done":
            result["done_count"] += 1
        if status == "paused":
            result["paused_count"] += 1
    result["status_counts"] = dict(sorted(statuses.items()))
    result["field_presence"] = dict(sorted(fields.items()))
    return result


def parse_rclone_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    result: dict[str, Any] = {"exists": config_path.exists(), "remote_count": 0, "remotes": []}
    if not config_path.exists():
        return result
    parser = configparser.ConfigParser()
    try:
        parser.read(config_path, encoding="utf-8")
    except configparser.Error:
        return result
    remotes = []
    for section in parser.sections():
        remotes.append(
            {
                "name": section,
                "type": parser.get(section, "type", fallback=None),
                "drive_type": parser.get(section, "drive_type", fallback=None),
            }
        )
    result["remote_count"] = len(remotes)
    result["remotes"] = remotes
    return result


def _file_marker_status(path: Path, markers: list[str]) -> dict[str, Any]:
    status = {
        "path": str(path),
        "exists": path.exists(),
        "markers": {marker: False for marker in markers},
    }
    if not path.exists():
        return status
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return status
    folded = text.lower().replace("`", "")
    status["markers"] = {marker: marker.lower() in folded for marker in markers}
    return status


def check_storage_policy_presence(
    hermes_home: str | Path = DEFAULT_HERMES_HOME,
    ai_ops_brain: str | Path = DEFAULT_AI_OPS_BRAIN,
) -> dict[str, Any]:
    home = Path(hermes_home)
    brain = Path(ai_ops_brain)
    files = {
        "USER.md": _file_marker_status(
            home / "memories" / "USER.md",
            ["stay lean", "onedrive:", "familyhub-onedrive:", "outside OneDrive AI"],
        ),
        "MEMORY.md": _file_marker_status(
            home / "memories" / "MEMORY.md",
            [
                "onedrive:AI/",
                "familyhub-onedrive:Family Hub/App Storage/",
                "littleton-google-drive:",
                "gemini-drive:",
                "command-based",
            ],
        ),
        "storage_runbook": _file_marker_status(
            brain / "ai-ops/storage-hygiene/global-storage-drive-rules-2026-06-02.md",
            ["14 days", "30 days", "48 hours", "keeper", "review-package", "dry-run", "proof"],
        ),
    }
    all_present = all(
        file_status["exists"] and all(file_status["markers"].values())
        for file_status in files.values()
    )
    return {"all_required_markers_present": all_present, "files": files}


def check_quality_policy_presence(
    repo_root: str | Path,
    hermes_home: str | Path = DEFAULT_HERMES_HOME,
) -> dict[str, Any]:
    root = Path(repo_root)
    session_py = root / "gateway" / "session.py"
    quality_lanes_py = root / "gateway" / "quality_lanes.py"
    delegate_evidence_py = root / "gateway" / "delegate_evidence.py"
    delegate_tool_py = root / "tools" / "delegate_tool.py"
    toolsets_py = root / "toolsets.py"
    source = ""
    try:
        source = session_py.read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    quality_source = ""
    try:
        quality_source = quality_lanes_py.read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    delegate_source = ""
    try:
        delegate_source = delegate_tool_py.read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    delegate_evidence_source = ""
    try:
        delegate_evidence_source = delegate_evidence_py.read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    toolsets_source = ""
    try:
        toolsets_source = toolsets_py.read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    source_markers = {
        "quality_lane_markers_constant": "QUALITY_LANE_POLICY_MARKERS" in source,
        "prompt_renderer": "render_quality_lane_policy_for_prompt" in source,
        "prompt_builder_call": "render_quality_lane_policy_for_prompt()" in source,
    }
    gate_markers = {
        "quality_lane_required_fields": "QUALITY_LANE_REQUIRED_FIELDS" in quality_source,
        "section_renderer": "def require_quality_lane_section" in quality_source,
        "section_validator": "def validate_quality_lane_section" in quality_source,
        "delegate_capability_detector": "def detect_delegate_task_capability" in quality_source,
        "final_report_wrapper": "def ensure_quality_lane_section" in quality_source,
    }
    fallback_available = (
        "Subagent unavailable/not invoked; checklist fallback used." in quality_source
        and "checklist fallback" in quality_source.lower()
    )
    delegate_capability = "unknown"
    if delegate_source or toolsets_source:
        delegate_capability = (
            "yes"
            if "def delegate_task" in delegate_source and "delegate_task" in toolsets_source
            else "no"
        )
    delegate_tracking_available = (
        "def record_delegate_evidence" in delegate_evidence_source
        and "def get_recent_delegate_evidence" in delegate_evidence_source
    )
    delegate_records: dict[str, Any] = {
        "count": 0,
        "status_counts": {},
    }
    if delegate_tracking_available:
        try:
            root_text = str(root.resolve())
            if root_text not in sys.path:
                sys.path.insert(0, root_text)
            from gateway.delegate_evidence import get_recent_delegate_evidence

            recent = get_recent_delegate_evidence()
            status_counts: dict[str, int] = {}
            for item in recent:
                status = str(item.get("status") or "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1
            delegate_records = {"count": len(recent), "status_counts": status_counts}
        except Exception:
            delegate_records = {"count": 0, "status_counts": {}, "read_error": True}
    memory_text = ""
    home = Path(hermes_home)
    for path in (home / "memories" / "MEMORY.md", home / "memories" / "USER.md"):
        try:
            memory_text += "\n" + path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
    memory_folded = memory_text.lower()
    memory_markers = {
        "WAHA": "waha" in memory_folded,
        "Signal Room": "signal room" in memory_folded,
        "video production": "video production" in memory_folded,
        "Hermes reliability": "hermes reliability" in memory_folded,
    }
    return {
        "injection_path_enabled": all(source_markers.values()),
        "source_markers": source_markers,
        "quality_lane_gate_available": all(gate_markers.values()),
        "quality_gate_markers": gate_markers,
        "delegate_capability_detected": delegate_capability,
        "real_subagent_capability_detected": delegate_capability,
        "delegate_evidence_tracking_available": delegate_tracking_available,
        "recent_delegate_records": delegate_records,
        "checklist_fallback_available": fallback_available,
        "high_risk_final_report_template_available": (
            gate_markers["quality_lane_required_fields"]
            and gate_markers["section_renderer"]
            and gate_markers["final_report_wrapper"]
        ),
        "memory_markers": memory_markers,
    }


def inspect_delegate_evidence_store(path: str | Path) -> dict[str, Any]:
    store_path = Path(path)
    result: dict[str, Any] = {
        "exists": store_path.exists(),
        "parsed": False,
        "record_count": 0,
        "recent_record_count": 0,
        "lane_status_counts": {},
        "status_counts": {},
        "checklist_fallback_count": 0,
        "unresolved_or_failed_count": 0,
    }
    if not store_path.exists():
        return result
    try:
        data = json.loads(store_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return result
    records = data.get("records") if isinstance(data, dict) else None
    if not isinstance(records, list):
        return result

    safe_records = [item for item in records if isinstance(item, dict)]
    status_counts: Counter[str] = Counter()
    lane_counts: dict[str, Counter[str]] = {}
    fallback_count = 0
    unresolved = 0
    for item in safe_records:
        lane = str(item.get("lane") or "unknown")
        status = str(item.get("status") or "unknown")
        status_counts[status] += 1
        lane_counts.setdefault(lane, Counter())[status] += 1
        if item.get("evidence_source") == "checklist_fallback":
            fallback_count += 1
        if status in {"failed", "skipped", "pending", "unknown"}:
            unresolved += 1

    result.update(
        {
            "parsed": True,
            "record_count": len(safe_records),
            "recent_record_count": len(safe_records[-20:]),
            "lane_status_counts": {
                lane: dict(counts) for lane, counts in sorted(lane_counts.items())
            },
            "status_counts": dict(status_counts),
            "checklist_fallback_count": fallback_count,
            "unresolved_or_failed_count": unresolved,
        }
    )
    return result


def inspect_mount_names() -> dict[str, Any]:
    result = _run_command(["mount"], timeout=3.0)
    names = []
    if result["ok"]:
        for line in result["stdout"].splitlines():
            lowered = line.lower()
            if any(term in lowered for term in ("onedrive", "google", "gdrive", "rclone", "fuse")):
                names.append(line.split(" ", 3)[0])
    return {"command_ok": result["ok"], "cloud_like_mount_count": len(names), "cloud_like_mount_sources": names}


def collect_report(args: argparse.Namespace) -> dict[str, Any]:
    hermes_home = Path(args.hermes_home).expanduser()
    service_state = get_service_state(args.service)
    proc_cwd = get_proc_cwd(service_state.get("main_pid"))
    module_paths = resolve_module_paths(
        args.gateway_venv_python,
        cwd=service_state.get("working_directory") or args.expected_runtime_checkout,
    )
    cli = get_cli_project(args.hermes_bin)
    topology = evaluate_runtime_topology(
        expected_runtime_checkout=args.expected_runtime_checkout,
        service_working_directory=service_state.get("working_directory"),
        proc_cwd=proc_cwd,
        module_paths=module_paths,
        cli_project=cli.get("project"),
    )
    return {
        "runtime": {
            "service": service_state,
            "proc_cwd": proc_cwd,
            "module_paths": module_paths,
            "cli": {"binary": cli.get("binary"), "project": cli.get("project")},
            "topology": topology,
        },
        "stores": {
            "active_tasks": inspect_active_task_store(hermes_home / "session_active_tasks.json"),
            "goals": inspect_goal_store(hermes_home / "state.db"),
            "delegate_evidence": inspect_delegate_evidence_store(
                hermes_home / "delegate_evidence.json"
            ),
        },
        "quality_policy": check_quality_policy_presence(
            args.expected_runtime_checkout,
            hermes_home,
        ),
        "storage": {
            "policy": check_storage_policy_presence(hermes_home, args.ai_ops_brain),
            "rclone_config": parse_rclone_config(Path.home() / ".config/rclone/rclone.conf"),
            "mounts": inspect_mount_names(),
        },
    }


def format_report(report: dict[str, Any]) -> str:
    runtime = report["runtime"]
    stores = report["stores"]
    quality_policy = report["quality_policy"]
    storage = report["storage"]
    delegate_evidence = stores.get("delegate_evidence") or {}
    lines = [
        "Hermes reliability doctor (read-only)",
        "",
        "Runtime:",
        f"  service WorkingDirectory: {runtime['service'].get('working_directory')}",
        f"  service MainPID: {runtime['service'].get('main_pid')}",
        f"  /proc/MainPID/cwd: {runtime.get('proc_cwd')}",
        f"  CLI binary: {runtime['cli'].get('binary')}",
        f"  CLI project: {runtime['cli'].get('project')}",
        f"  split-brain risk: {runtime['topology'].get('split_brain_risk')}",
        "",
        "Module paths:",
    ]
    for name, path in sorted(runtime["module_paths"].items()):
        lines.append(f"  {name}: {path}")
    lines.extend(
        [
            "",
            "Stores:",
            f"  active task records: {stores['active_tasks'].get('record_count')}",
            f"  foreground records: {stores['active_tasks'].get('foreground_count')}",
            f"  foreground records missing task body: {stores['active_tasks'].get('foreground_missing_task_count')}",
            f"  stale active foreground records: {stores['active_tasks'].get('stale_active_foreground_count')}",
            f"  active task contracts: {stores['active_tasks'].get('task_contract_count')}",
            f"  task contracts missing summary: {stores['active_tasks'].get('task_contract_missing_summary_count')}",
            f"  task contract statuses: {stores['active_tasks'].get('task_contract_status_counts')}",
            f"  stale/superseded task contracts: {stores['active_tasks'].get('stale_or_superseded_task_contract_count')}",
            f"  task contracts with intended_repo: {stores['active_tasks'].get('task_contracts_with_intended_repo_count')}",
            f"  task contracts with restart_policy: {stores['active_tasks'].get('task_contracts_with_restart_policy_count')}",
            f"  final report statuses: {stores['active_tasks'].get('final_report_counts')}",
            f"  goal rows: {stores['goals'].get('goal_count')}",
            f"  active goals: {stores['goals'].get('active_count')}",
            f"  goal statuses: {stores['goals'].get('status_counts')}",
            f"  goal field presence: {stores['goals'].get('field_presence')}",
            f"  delegate evidence records: {delegate_evidence.get('record_count')}",
            f"  recent delegate evidence records: {delegate_evidence.get('recent_record_count')}",
            f"  delegate lane status counts: {delegate_evidence.get('lane_status_counts')}",
            f"  delegate checklist fallback records: {delegate_evidence.get('checklist_fallback_count')}",
            f"  unresolved/failed delegate lane records: {delegate_evidence.get('unresolved_or_failed_count')}",
            "",
            "Quality policy:",
            f"  injection enabled: {quality_policy.get('injection_path_enabled')}",
            f"  quality lane gate available: {quality_policy.get('quality_lane_gate_available')}",
            f"  delegate capability detected: {quality_policy.get('delegate_capability_detected')}",
            f"  delegate evidence tracking available: {quality_policy.get('delegate_evidence_tracking_available')}",
            f"  recent delegate records: {quality_policy.get('recent_delegate_records')}",
            f"  real subagent capability detected: {quality_policy.get('real_subagent_capability_detected')}",
            f"  checklist fallback available: {quality_policy.get('checklist_fallback_available')}",
            f"  high-risk final report template available: {quality_policy.get('high_risk_final_report_template_available')}",
            f"  memory markers: {quality_policy.get('memory_markers')}",
            "",
            "Storage:",
            f"  policy markers complete: {storage['policy'].get('all_required_markers_present')}",
            f"  rclone config exists: {storage['rclone_config'].get('exists')}",
            f"  rclone remotes: {[remote['name'] for remote in storage['rclone_config'].get('remotes', [])]}",
            f"  cloud-like mounts: {storage['mounts'].get('cloud_like_mount_count')}",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only Hermes reliability diagnostics.")
    parser.add_argument("--expected-runtime-checkout", default=DEFAULT_EXPECTED_CHECKOUT)
    parser.add_argument("--hermes-home", default=DEFAULT_HERMES_HOME)
    parser.add_argument("--ai-ops-brain", default=DEFAULT_AI_OPS_BRAIN)
    parser.add_argument("--gateway-venv-python", default=DEFAULT_GATEWAY_VENV_PYTHON)
    parser.add_argument("--service", default=DEFAULT_SERVICE)
    parser.add_argument("--hermes-bin", default=None)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = collect_report(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
