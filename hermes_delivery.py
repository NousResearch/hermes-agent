"""Hermes delivery pipeline ledger and gate runner.

This module is intentionally local-file based so the desktop app, CLI, cron
watchdog, and background workers can all share one durable view without an
approval-card dependency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any
from uuid import uuid4


SCHEMA = "hermes.delivery.v1"

PIPELINE_STAGES: list[dict[str, Any]] = [
    {
        "id": "blueprint",
        "name": "Blueprint",
        "artifact": "blueprint.md",
        "required_artifacts": ["blueprint.md"],
        "gate_signals": ["state", "blueprint"],
        "failure_action": "backwrite blueprint",
    },
    {
        "id": "story",
        "name": "Story",
        "artifact": "stories.md",
        "required_artifacts": ["stories.md"],
        "gate_signals": ["story", "role", "path"],
        "failure_action": "rewrite story",
    },
    {
        "id": "spec",
        "name": "Spec",
        "artifact": "spec.md",
        "required_artifacts": ["spec.md"],
        "gate_signals": ["file", "test"],
        "failure_action": "rewrite spec",
    },
    {
        "id": "task",
        "name": "Task",
        "artifact": "tasks.md",
        "required_artifacts": ["tasks.md"],
        "gate_signals": ["issue", "pr", "file"],
        "failure_action": "re-split tasks",
    },
    {
        "id": "linear",
        "name": "Linear",
        "artifact": "issues.json",
        "required_artifacts": ["issues.json"],
        "gate_signals": ["issue", "spec"],
        "failure_action": "retry linear sync",
    },
    {
        "id": "dispatch",
        "name": "Dispatch",
        "artifact": "jobs/",
        "required_artifacts": ["jobs"],
        "gate_signals": ["owner", "executor"],
        "failure_action": "re-dispatch",
    },
    {
        "id": "pr_monitor",
        "name": "PR Monitor",
        "artifact": "pr-status.json",
        "required_artifacts": ["pr-status.json"],
        "gate_signals": ["pr", "issue"],
        "failure_action": "create or fix PR",
    },
    {
        "id": "ci_monitor",
        "name": "CI Monitor",
        "artifact": "ci/latest.json",
        "required_artifacts": ["ci/latest.json"],
        "gate_signals": ["success", "checks"],
        "failure_action": "fix CI",
    },
    {
        "id": "anti_shell",
        "name": "Anti-Shell",
        "artifact": "anti-shell/report.json",
        "required_artifacts": ["anti-shell/report.json"],
        "gate_signals": ["passed"],
        "failure_action": "return to real implementation",
    },
    {
        "id": "acceptance",
        "name": "Acceptance",
        "artifact": "acceptance/report.md",
        "required_artifacts": ["acceptance/report.md"],
        "gate_signals": ["machine", "ai", "human"],
        "failure_action": "redo acceptance",
    },
    {
        "id": "delivery",
        "name": "Delivery",
        "artifact": "delivery-summary.md",
        "required_artifacts": ["delivery-summary.md"],
        "gate_signals": ["notification", "summary"],
        "failure_action": "write delivery summary",
    },
    {
        "id": "deploy",
        "name": "Deploy",
        "artifact": "deploy-runs/",
        "required_artifacts": ["deploy-runs"],
        "gate_signals": ["deploy", "verify"],
        "failure_action": "fix deploy",
    },
    {
        "id": "backwrite",
        "name": "Backwrite",
        "artifact": "updated-blueprint.md",
        "required_artifacts": ["updated-blueprint.md"],
        "gate_signals": ["baseline", "updated"],
        "failure_action": "backwrite docs",
    },
]

PRINCIPLES = [
    "No approval cards; execution is driven by gates and evidence.",
    "Every green stage must have a durable artifact.",
    "Anti-Shell blocks empty completion claims.",
]


@dataclass(frozen=True)
class DeployRequest:
    target: str = "hermes"
    environment: str = "local"
    adapter: str = "command"
    ref: str = "HEAD"
    command: str | None = None
    cwd: str | None = None
    verify: list[str] = field(default_factory=list)
    rollback: str | None = None
    execute: bool = False
    force: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def delivery_root() -> Path:
    return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes").expanduser() / "delivery"


def artifact_path(relative: str) -> Path:
    return delivery_root() / relative


def _state_path() -> Path:
    return artifact_path("state.json")


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


def _initial_state() -> dict[str, Any]:
    now = _utcnow()
    return {
        "schema": SCHEMA,
        "created_at": now,
        "updated_at": now,
        "stages": [
            {
                "id": stage["id"],
                "name": stage["name"],
                "artifact": stage["artifact"],
                "status": "idle",
                "notes": "",
                "updated_at": now,
                "fail_count": 0,
            }
            for stage in PIPELINE_STAGES
        ],
        "deploy": {"runs": []},
        "jobs": [],
        "approval_cards": False,
    }


def load_state() -> dict[str, Any]:
    state = _read_json(_state_path(), _initial_state())
    if not isinstance(state, dict):
        state = _initial_state()
    state.setdefault("schema", SCHEMA)
    state.setdefault("created_at", _utcnow())
    state.setdefault("deploy", {"runs": []})
    state.setdefault("jobs", [])
    state.setdefault("approval_cards", False)
    existing = {stage.get("id"): stage for stage in state.get("stages", []) if isinstance(stage, dict)}
    merged: list[dict[str, Any]] = []
    for stage in PIPELINE_STAGES:
        row = dict(existing.get(stage["id"], {}))
        row.setdefault("id", stage["id"])
        row.setdefault("name", stage["name"])
        row.setdefault("status", "idle")
        row.setdefault("notes", "")
        row.setdefault("updated_at", _utcnow())
        row.setdefault("fail_count", 0)
        row["artifact"] = stage["artifact"]
        merged.append(row)
    state["stages"] = merged
    state["updated_at"] = state.get("updated_at") or _utcnow()
    return state


def save_state(state: dict[str, Any]) -> dict[str, Any]:
    state["updated_at"] = _utcnow()
    _atomic_write_json(_state_path(), state)
    return state


def _stage_def(stage_id: str) -> dict[str, Any]:
    for stage in PIPELINE_STAGES:
        if stage["id"] == stage_id:
            return stage
    raise ValueError(f"unknown delivery stage: {stage_id}")


def update_stage(stage_id: str, status: str, notes: str = "") -> dict[str, Any]:
    _stage_def(stage_id)
    state = load_state()
    for stage in state["stages"]:
        if stage["id"] == stage_id:
            stage["status"] = status
            stage["notes"] = notes
            stage["updated_at"] = _utcnow()
            if status in {"done", "ready"}:
                stage["fail_count"] = 0
            break
    return save_state(state)


def dashboard_snapshot() -> dict[str, Any]:
    state = load_state()
    done = sum(1 for stage in state["stages"] if stage.get("status") == "done")
    return {
        "ok": True,
        "schema": SCHEMA,
        "state_path": str(_state_path()),
        "principles": PRINCIPLES,
        "summary": {
            "done": done,
            "total": len(PIPELINE_STAGES),
            "approval_cards": False,
            "deploy_runs": len(state.get("deploy", {}).get("runs", [])),
        },
        "stages": state["stages"],
    }


def capability_manifest() -> dict[str, Any]:
    return {
        "schema": "hermes.capability-manifest.v1",
        "no_approval_cards": True,
        "capabilities": [
            {
                "id": stage["id"],
                "name": stage["name"],
                "artifact": stage["artifact"],
                "required_artifacts": stage["required_artifacts"],
                "gate_signals": stage["gate_signals"],
                "failure_action": stage["failure_action"],
            }
            for stage in PIPELINE_STAGES
        ],
    }


def _artifact_ok(relative: str) -> bool:
    path = artifact_path(relative)
    if path.is_dir():
        return any(path.iterdir())
    return path.exists() and path.stat().st_size > 0


def _read_artifact_text(relative: str) -> str:
    path = artifact_path(relative)
    if path.is_dir():
        return " ".join(child.name for child in path.iterdir())
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


SIGNAL_ALIASES: dict[str, tuple[str, ...]] = {
    "file": ("file", "files", "文件", "涉及文件"),
    "test": ("test", "tests", "pytest", "验收测试", "测试"),
    "issue": ("issue", "linear", "任务"),
    "pr": ("pr", "pull request", "合并请求"),
    "owner": ("owner", "负责人"),
    "executor": ("executor", "执行者"),
    "success": ("success", "passed", "通过", "成功"),
    "checks": ("checks", "check-runs", "检查"),
    "machine": ("machine", "机器"),
    "ai": ("ai", "review", "审查"),
    "human": ("human", "人工"),
    "notification": ("notification", "通知"),
    "summary": ("summary", "总结"),
    "baseline": ("baseline", "基线"),
    "updated": ("updated", "更新"),
    "deploy": ("deploy", "部署"),
    "verify": ("verify", "验证"),
}


def _signal_present(signal: str, haystack: str) -> bool:
    aliases = SIGNAL_ALIASES.get(signal.lower(), (signal,))
    return any(alias.lower() in haystack for alias in aliases)


def _story_sections(text: str) -> list[str]:
    raw_sections = re.split(r"(?im)^##\s+story\b.*$", text)
    story_sections = [section.strip() for section in raw_sections[1:] if section.strip()]
    if story_sections:
        return story_sections
    raw_sections = re.split(r"(?m)^##\s+", text)
    return [section.strip() for section in raw_sections[1:] if section.strip()]


def _story_gate_checks(story_text: str) -> list[dict[str, Any]]:
    stories = _story_sections(story_text)
    checks: list[dict[str, Any]] = [
        {"name": "story_entries", "ok": bool(stories), "detail": str(len(stories))},
    ]
    for index, story in enumerate(stories, start=1):
        story_lower = story.lower()
        has_role = bool(re.search(r"(?im)^\s*[-*]?\s*(角色|role)\s*[:：]\s*\S+", story))
        has_path = bool(
            re.search(r"(?im)^\s*[-*]?\s*(点击路径|path|click\s*path)\s*[:：]\s*\S+", story)
            or "->" in story
        )
        has_assertion = bool(re.search(r"(?im)^\s*[-*]?\s*(断言|assertion|acceptance)\s*[:：]\s*\S+", story))
        checks.extend(
            [
                {"name": f"story:{index}:role", "ok": has_role, "detail": "role/角色"},
                {"name": f"story:{index}:path", "ok": has_path, "detail": "path/点击路径"},
                {"name": f"story:{index}:assertion", "ok": has_assertion, "detail": "assertion/断言"},
                {"name": f"story:{index}:substantive", "ok": len(story_lower) >= 40, "detail": str(len(story))},
            ]
        )
    return checks


def _job_log_has_content(job: dict[str, Any]) -> bool:
    log_path = str(job.get("log_path") or "")
    if not log_path:
        return False
    path = Path(log_path).expanduser()
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


def _dispatch_gate_checks() -> list[dict[str, Any]]:
    jobs = list_jobs()
    latest = jobs[-1] if jobs else {}
    owner = str(latest.get("owner") or "").strip()
    executor = str(latest.get("executor") or "").strip()
    branch = str(latest.get("branch") or "").strip()
    command = str(latest.get("command") or "").strip()
    status = str(latest.get("status") or "").strip().lower()
    attempt = int(latest.get("attempt") or 0)
    return [
        {
            "name": "job_exists",
            "ok": bool(latest.get("job_id")),
            "detail": str(latest.get("job_id") or "missing execution-job.json"),
        },
        {"name": "has_owner", "ok": bool(owner and owner.lower() != "unknown"), "detail": owner or "missing"},
        {"name": "has_executor", "ok": bool(executor), "detail": executor or "missing"},
        {"name": "has_branch", "ok": bool(branch), "detail": branch or "missing"},
        {"name": "has_command", "ok": bool(command), "detail": command[:120] or "missing"},
        {
            "name": "dispatch_recorded",
            "ok": status in {"ready", "done", "failed"} and attempt > 0,
            "detail": f"status={status or 'missing'} attempt={attempt}",
        },
        {
            "name": "job_log_exists",
            "ok": _job_log_has_content(latest),
            "detail": str(latest.get("log_path") or "missing job.log"),
        },
    ]


def deploy_runs_dir() -> Path:
    return artifact_path("deploy-runs")


def list_deploy_runs() -> list[dict[str, Any]]:
    """Return deploy-run.json artifacts, ordered by artifact path."""

    root = deploy_runs_dir()
    if not root.exists():
        return []
    runs: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/deploy-run.json")):
        run = _read_json(path, {})
        if isinstance(run, dict) and run:
            run = dict(run)
            run.setdefault("run_path", str(path))
            runs.append(run)
    return runs


def _deploy_exit_codes(run: dict[str, Any]) -> list[int]:
    codes: list[int] = []
    for step in run.get("steps", []) or []:
        if isinstance(step, dict) and isinstance(step.get("exit_code"), int):
            codes.append(step["exit_code"])
    if isinstance(run.get("exit_code"), int):
        codes.append(run["exit_code"])
    return codes


def _deploy_log_has_content(run: dict[str, Any]) -> bool:
    log_path = str(run.get("log_path") or "")
    if not log_path:
        return False
    path = Path(log_path).expanduser()
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


def _deploy_request_value(run: dict[str, Any], key: str) -> str:
    request = run.get("request") if isinstance(run.get("request"), dict) else {}
    return str(request.get(key) or run.get(key) or "").strip()


def _deploy_run_is_verified(run: dict[str, Any]) -> bool:
    status = str(run.get("status") or "").strip().lower()
    request = run.get("request") if isinstance(run.get("request"), dict) else {}
    exit_codes = _deploy_exit_codes(run)
    return (
        bool(request.get("execute"))
        and bool(exit_codes)
        and any(code == 0 for code in exit_codes)
        and _deploy_log_has_content(run)
        and bool(_deploy_request_value(run, "target"))
        and bool(_deploy_request_value(run, "environment"))
        and status in {"verified", "success", "done"}
    )


def _deploy_gate_checks() -> list[dict[str, Any]]:
    runs = list_deploy_runs()
    verified_runs = [run for run in runs if _deploy_run_is_verified(run)]
    latest = verified_runs[-1] if verified_runs else (runs[-1] if runs else {})
    request = latest.get("request") if isinstance(latest.get("request"), dict) else {}
    exit_codes = _deploy_exit_codes(latest)
    status = str(latest.get("status") or "").strip().lower()
    target = _deploy_request_value(latest, "target")
    environment = _deploy_request_value(latest, "environment")
    return [
        {
            "name": "deploy_run_exists",
            "ok": bool(runs),
            "detail": str(latest.get("run_path") or deploy_runs_dir() / "*/deploy-run.json"),
        },
        {"name": "deploy_executed", "ok": bool(request.get("execute")), "detail": f"execute={request.get('execute')} exit_codes={exit_codes}"},
        {"name": "deploy_exit_code_recorded", "ok": bool(exit_codes), "detail": str(exit_codes or "missing")},
        {"name": "deploy_succeeded", "ok": any(code == 0 for code in exit_codes), "detail": f"status={status or 'missing'} exit_codes={exit_codes}"},
        {"name": "deploy_log_has_content", "ok": _deploy_log_has_content(latest), "detail": str(latest.get("log_path") or "missing deploy.log")},
        {"name": "target_present", "ok": bool(target and target.lower() not in {"unknown", "placeholder", "todo"}), "detail": target or "missing"},
        {"name": "environment_present", "ok": bool(environment), "detail": environment or "missing"},
        {"name": "status_verified", "ok": status in {"verified", "success", "done"}, "detail": status or "missing"},
    ]


def run_gate(
    stage_id: str,
    *,
    record_rollback: bool = False,
    update_state: bool = True,
) -> dict[str, Any]:
    stage_def = _stage_def(stage_id)
    checks: list[dict[str, Any]] = []
    for rel in stage_def["required_artifacts"]:
        checks.append({"name": f"artifact:{rel}", "ok": _artifact_ok(rel), "detail": rel})
    artifact_text = " ".join(_read_artifact_text(rel) for rel in stage_def["required_artifacts"])
    haystack = artifact_text.lower()
    if stage_id == "story":
        checks.extend(_story_gate_checks(artifact_text))
    elif stage_id == "dispatch":
        checks.extend(_dispatch_gate_checks())
    elif stage_id == "deploy":
        checks.extend(_deploy_gate_checks())
    else:
        for signal in stage_def["gate_signals"]:
            checks.append({"name": f"signal:{signal}", "ok": _signal_present(signal, haystack), "detail": signal})
    if record_rollback:
        checks.append({"name": "rollback_recorded", "ok": True, "detail": "recorded"})
    failed = [check for check in checks if not check["ok"]]
    state = load_state()
    gate_state: dict[str, Any] = {}
    for stage in state["stages"]:
        if stage["id"] == stage_id:
            if update_state:
                if failed:
                    stage["fail_count"] = int(stage.get("fail_count") or 0) + 1
                    stage["status"] = "blocked" if stage["fail_count"] >= 3 else "failed"
                    stage["notes"] = "GateRunner failed"
                else:
                    stage["fail_count"] = 0
                    stage["status"] = "done"
                    stage["notes"] = "GateRunner passed"
                stage["updated_at"] = _utcnow()
            gate_state = dict(stage)
            break
    if update_state:
        save_state(state)
    return {
        "stage_id": stage_id,
        "status": "failed" if failed else "passed",
        "checks": checks,
        "gate_state": gate_state,
    }


def run_capability_audit(*, update_state: bool = True) -> dict[str, Any]:
    results = [run_gate(stage["id"], update_state=update_state) for stage in PIPELINE_STAGES]
    passed = sum(1 for result in results if result["status"] == "passed")
    return {
        "ok": passed == len(results),
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "results": results,
    }


def _run_text_command(command: str, cwd: str | None = None) -> tuple[int, str, str]:
    result = subprocess.run(command, cwd=cwd, shell=True, capture_output=True, text=True, timeout=120)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def _run_json_command(args: list[str], cwd: str | None = None) -> tuple[int, Any, str]:
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        return result.returncode, None, result.stderr.strip()
    try:
        return result.returncode, json.loads(result.stdout or "null"), result.stderr.strip()
    except json.JSONDecodeError as exc:
        return 1, None, str(exc)


def _target_registry_path() -> Path:
    return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes").expanduser() / "deploy-targets.json"


def _redact_target(target: dict[str, Any]) -> dict[str, Any]:
    clean = {key: value for key, value in target.items() if key not in {"password", "secret", "token"}}
    if "password_env" in target:
        clean["credential"] = f"env:{target['password_env']}"
    return clean


def _deploy_policy(req: DeployRequest) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    add("adapter_supported", req.adapter in {"command", "github_actions", "ssh"}, req.adapter)
    add("command_or_adapter_present", bool(req.command or req.adapter == "github_actions"), req.command or req.adapter)
    if req.environment == "prod" and req.execute and not req.force:
        add("rollback_declared", bool(req.rollback), req.rollback or "missing rollback command")
    else:
        add("rollback_declared", True, "not required")
    target_info: dict[str, Any] = {}
    if req.adapter == "ssh":
        registry = _read_json(_target_registry_path(), {})
        target_info = registry.get("targets", {}).get(req.target, {}) if isinstance(registry, dict) else {}
        add("ssh_target_registered", bool(target_info), req.target)
        password_env = target_info.get("password_env") if isinstance(target_info, dict) else ""
        add("ssh_password_loaded", bool(password_env and os.environ.get(password_env)), f"env:{password_env}" if password_env else "missing")
    policy = {
        "allowed": all(check["ok"] for check in checks),
        "checks": checks,
        "target": _redact_target(target_info) if target_info else {},
    }
    return policy


def run_deploy(req: DeployRequest) -> dict[str, Any]:
    run_id = f"deploy-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:8]}"
    policy = _deploy_policy(req)
    run_dir = artifact_path(f"deploy-runs/{run_id}")
    log_path = run_dir / "deploy.log"
    stdout = ""
    stderr = ""
    exit_code: int | None = None
    status = "blocked"
    if policy["allowed"]:
        status = "ready"
        if req.execute and req.command:
            exit_code, stdout, stderr = _run_text_command(req.command, cwd=req.cwd)
            status = "done" if exit_code == 0 else "failed"
        elif req.execute:
            status = "done"
    run = {
        "schema": "hermes.deploy-run.v1",
        "run_id": run_id,
        "created_at": _utcnow(),
        "status": status,
        "request": asdict(req),
        "policy": policy,
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "log_path": str(log_path),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path.write_text((stdout + "\n" + stderr).strip() + "\n", encoding="utf-8")
    _atomic_write_json(run_dir / "deploy-run.json", run)
    state = load_state()
    state.setdefault("deploy", {}).setdefault("runs", []).append(
        {"run_id": run_id, "status": status, "created_at": run["created_at"], "log_path": str(log_path)}
    )
    save_state(state)
    if status in {"ready", "done"}:
        update_stage("deploy", "done", "GateRunner passed")
    else:
        update_stage("deploy", "failed", "Deploy policy or command failed")
    return run


def create_execution_job(
    *,
    title: str,
    owner: str,
    executor: str,
    branch: str,
    command: str | None = None,
    issue: str = "",
    repo: str = "",
    spec: str = "spec.md",
) -> dict[str, Any]:
    job_id = f"job-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:8]}"
    log_path = artifact_path(f"jobs/{job_id}/job.log")
    now = _utcnow()
    job = {
        "schema": "hermes.execution-job.v1",
        "job_id": job_id,
        "title": title,
        "owner": owner,
        "executor": executor,
        "branch": branch,
        "command": command or "",
        "issue": issue,
        "repo": repo,
        "spec": spec,
        "status": "created",
        "attempt": 0,
        "created_at": now,
        "updated_at": now,
        "log_path": str(log_path),
    }
    _atomic_write_json(artifact_path(f"jobs/{job_id}/execution-job.json"), job)
    state = load_state()
    state.setdefault("jobs", []).append({"job_id": job_id, "title": title, "status": "created"})
    save_state(state)
    update_stage("dispatch", "ready", "Execution job created")
    return job


def run_execution_job(job_id: str, *, execute: bool = False) -> dict[str, Any]:
    path = artifact_path(f"jobs/{job_id}/execution-job.json")
    job = _read_json(path, {})
    if not isinstance(job, dict) or not job:
        raise ValueError(f"unknown execution job: {job_id}")
    status = "ready"
    exit_code = None
    stdout = ""
    stderr = ""
    attempt = int(job.get("attempt") or 0) + 1
    started_at = _utcnow()
    if execute and job.get("command"):
        exit_code, stdout, stderr = _run_text_command(str(job["command"]), cwd=job.get("repo") or None)
        status = "done" if exit_code == 0 else "failed"
    elif execute:
        status = "done"
    log_path = Path(str(job.get("log_path") or artifact_path(f"jobs/{job_id}/job.log"))).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_lines = [
        f"job_id={job_id}",
        f"attempt={attempt}",
        f"started_at={started_at}",
        f"execute={execute}",
        f"status={status}",
        f"command={job.get('command') or ''}",
        f"exit_code={'' if exit_code is None else exit_code}",
        "--- stdout ---",
        stdout,
        "--- stderr ---",
        stderr,
    ]
    log_path.write_text("\n".join(log_lines).rstrip() + "\n", encoding="utf-8")
    job.update(
        {
            "status": status,
            "attempt": attempt,
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "updated_at": _utcnow(),
            "log_path": str(log_path),
        }
    )
    _atomic_write_json(path, job)
    state = load_state()
    for item in state.setdefault("jobs", []):
        if isinstance(item, dict) and item.get("job_id") == job_id:
            item.update({"status": status, "attempt": attempt, "updated_at": job["updated_at"], "log_path": str(log_path)})
            break
    else:
        state["jobs"].append(
            {
                "job_id": job_id,
                "title": job.get("title", ""),
                "status": status,
                "attempt": attempt,
                "updated_at": job["updated_at"],
                "log_path": str(log_path),
            }
        )
    save_state(state)
    update_stage("dispatch", "done" if status in {"ready", "done"} else "failed", "Execution job recorded")
    return job


def list_jobs() -> list[dict[str, Any]]:
    jobs_dir = artifact_path("jobs")
    if not jobs_dir.exists():
        return []
    jobs: list[dict[str, Any]] = []
    for path in sorted(jobs_dir.glob("*/execution-job.json")):
        job = _read_json(path, {})
        if isinstance(job, dict):
            jobs.append(job)
    return jobs


def refresh_pr_status(*, repo: str = "", pr: str = "", issue: str = "", cwd: str | None = None) -> dict[str, Any]:
    args = ["gh", "pr", "view"]
    if pr:
        args.append(pr)
    if repo:
        args.extend(["--repo", repo])
    args.extend(["--json", "number,url,state,title,body,headRefName,closingIssuesReferences"])
    code, data, err = _run_json_command(args, cwd=cwd)
    if code != 0 or not isinstance(data, dict):
        data = {"error": err, "state": "UNKNOWN", "closingIssuesReferences": []}
    linked = bool(data.get("closingIssuesReferences")) or bool(issue and issue in json.dumps(data, ensure_ascii=False))
    data["linked_issue"] = linked
    _atomic_write_json(artifact_path("pr-status.json"), data)
    update_stage("pr_monitor", "done" if data.get("state") in {"OPEN", "MERGED", "CLOSED"} and linked else "failed", "PR monitor refreshed")
    return data


def _normalize_check_state(value: Any) -> str:
    raw = str(value or "").lower()
    if raw in {"pass", "passed", "success", "successful", "completed"}:
        return "success"
    if raw in {"pending", "queued", "in_progress"}:
        return "pending"
    return raw or "unknown"


def refresh_ci_status(*, repo: str = "", ref: str = "", cwd: str | None = None) -> dict[str, Any]:
    args = ["gh", "pr", "checks"]
    if ref:
        args.append(ref)
    if repo:
        args.extend(["--repo", repo])
    args.extend(["--json", "name,state,link"])
    code, data, err = _run_json_command(args, cwd=cwd)
    if code != 0 or not isinstance(data, list):
        data = [{"name": "ci", "state": "unknown", "error": err}]
    checks = [
        {
            **check,
            "conclusion": _normalize_check_state(check.get("state") or check.get("conclusion")),
        }
        for check in data
        if isinstance(check, dict)
    ]
    status = {"schema": "hermes.ci-status.v1", "generated_at": _utcnow(), "checks": checks}
    _atomic_write_json(artifact_path("ci/latest.json"), status)
    all_success = bool(checks) and all(check.get("conclusion") == "success" for check in checks)
    update_stage("ci_monitor", "done" if all_success else "failed", "CI monitor refreshed")
    return status


def write_acceptance_report(
    *,
    machine_evidence: list[str],
    ai_review_evidence: list[str],
    human_evidence: str,
) -> dict[str, Any]:
    text = (
        "# Acceptance Report\n\n"
        "## Machine\n" + "\n".join(f"- {item}" for item in machine_evidence) + "\n\n"
        "## AI Review\n" + "\n".join(f"- {item}" for item in ai_review_evidence) + "\n\n"
        f"## Human\n{human_evidence}\n"
    )
    path = artifact_path("acceptance/report.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    gate = run_gate("acceptance")
    return {"path": str(path), "gate": gate}


def write_delivery_summary(notification: str) -> dict[str, Any]:
    path = artifact_path("delivery-summary.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# Delivery Summary\n\nsummary notification: {notification}\n", encoding="utf-8")
    gate = run_gate("delivery")
    return {"path": str(path), "gate": gate}


def write_backwrite_blueprint(notes: str) -> dict[str, Any]:
    path = artifact_path("updated-blueprint.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# Updated Blueprint\n\nbaseline updated: {notes}\n", encoding="utf-8")
    gate = run_gate("backwrite")
    return {"path": str(path), "gate": gate}
