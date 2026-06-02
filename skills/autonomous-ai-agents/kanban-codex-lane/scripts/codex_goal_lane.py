#!/usr/bin/env python3
"""Operate a Hermes-owned Codex builder lane with Claude review."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FINAL_STATUSES = {
    "accepted",
    "human_review",
    "rework_required",
    "rejected",
    "stopped",
    "timed_out",
}
VERDICT_RE = re.compile(r'"verdict"\s*:\s*"(pass|fail)"', re.IGNORECASE)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in value)
    safe = "-".join(part for part in safe.split("-") if part)
    return safe[:48] or "manual"


def hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home())
    except Exception:
        return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))


def default_state_root() -> Path:
    return hermes_home() / "codex-goal-lanes"


def run_command(
    argv: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 30,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=str(cwd) if cwd else None,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def run_shell_command(
    command: str,
    *,
    cwd: Path,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        shell=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def check(
    checks: list[dict[str, Any]],
    name: str,
    ok: bool,
    detail: str,
    *,
    required: bool = True,
) -> None:
    checks.append(
        {
            "name": name,
            "status": "ok" if ok else ("fail" if required else "warn"),
            "required": required,
            "detail": detail,
        }
    )


def check_tool(tool: str, checks: list[dict[str, Any]], *, required: bool) -> Path | None:
    path = shutil.which(tool)
    check(
        checks,
        f"{tool}_cli",
        path is not None,
        str(path) if path else f"{tool} not found on PATH",
        required=required,
    )
    return Path(path) if path else None


def codex_auth_available() -> tuple[bool, str]:
    if os.environ.get("OPENAI_API_KEY"):
        return True, "OPENAI_API_KEY is present"
    codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    auth_file = codex_home / "auth.json"
    if auth_file.exists():
        return True, f"{auth_file} exists"
    return False, "OPENAI_API_KEY is absent and Codex OAuth auth.json was not found"


def claude_auth_available() -> tuple[bool, str]:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True, "ANTHROPIC_API_KEY is present"
    candidates = [
        Path.home() / ".claude.json",
        Path.home() / ".claude" / ".credentials.json",
        Path.home() / ".config" / "claude-code" / "auth.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return True, f"{candidate} exists"
    return False, "ANTHROPIC_API_KEY is absent and no Claude Code auth file was found"


def repo_root(repo: Path) -> Path:
    result = run_command(["git", "-C", str(repo), "rev-parse", "--show-toplevel"])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"{repo} is not a git repository")
    return Path(result.stdout.strip()).resolve()


def git_common_dir(repo: Path) -> Path:
    result = run_command(["git", "-C", str(repo), "rev-parse", "--git-common-dir"])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"{repo} has no git common dir")
    common = Path(result.stdout.strip())
    if not common.is_absolute():
        common = repo / common
    return common.resolve()


def current_ref(repo: Path) -> str:
    result = run_command(["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"])
    if result.returncode == 0 and result.stdout.strip() and result.stdout.strip() != "HEAD":
        return result.stdout.strip()
    result = run_command(["git", "-C", str(repo), "rev-parse", "HEAD"])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "could not resolve git HEAD")
    return result.stdout.strip()


def preflight_checks(
    repo: Path,
    *,
    require_codex: bool,
    require_claude: bool,
    simulate: bool,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    repo = repo.resolve()

    git_path = check_tool("git", checks, required=True)
    if git_path:
        result = run_command(["git", "-C", str(repo), "rev-parse", "--show-toplevel"])
        check(
            checks,
            "git_repo",
            result.returncode == 0,
            result.stdout.strip() if result.returncode == 0 else result.stderr.strip(),
            required=True,
        )
        worktree = run_command(["git", "-C", str(repo), "worktree", "list"])
        check(
            checks,
            "git_worktree",
            worktree.returncode == 0,
            "git worktree list succeeded"
            if worktree.returncode == 0
            else worktree.stderr.strip(),
            required=True,
        )

    codex_required = require_codex and not simulate
    if check_tool("codex", checks, required=codex_required):
        version = run_command(["codex", "--version"], timeout=15)
        check(
            checks,
            "codex_version",
            version.returncode == 0,
            version.stdout.strip() if version.returncode == 0 else version.stderr.strip(),
            required=codex_required,
        )
        goals = run_command(["codex", "features", "list"], timeout=15)
        detail = "goals feature listed" if "goals" in goals.stdout.lower() else "goals feature not listed"
        check(checks, "codex_goals_feature", goals.returncode == 0, detail, required=False)
    codex_auth_ok, codex_auth_detail = codex_auth_available()
    check(checks, "codex_auth", codex_auth_ok, codex_auth_detail, required=codex_required)

    claude_required = require_claude and not simulate
    if check_tool("claude", checks, required=claude_required):
        version = run_command(["claude", "--version"], timeout=15)
        check(
            checks,
            "claude_version",
            version.returncode == 0,
            version.stdout.strip() if version.returncode == 0 else version.stderr.strip(),
            required=claude_required,
        )
    claude_auth_ok, claude_auth_detail = claude_auth_available()
    check(checks, "claude_auth", claude_auth_ok, claude_auth_detail, required=claude_required)

    failed = [item for item in checks if item["required"] and item["status"] == "fail"]
    return {"success": not failed, "checks": checks, "failed": failed}


def state_dir(state_root: Path, run_id: str) -> Path:
    return state_root / run_id


def state_path(state_root: Path, run_id: str) -> Path:
    return state_dir(state_root, run_id) / "state.json"


def save_state(state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    path = Path(state["state_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_state(state_root: Path, run_id: str) -> dict[str, Any]:
    path = state_path(state_root, run_id)
    if not path.exists():
        raise RuntimeError(f"run state not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def record_phase(state: dict[str, Any], name: str, status: str, evidence: dict[str, Any]) -> None:
    state.setdefault("phases", []).append(
        {
            "name": name,
            "status": status,
            "evidence": evidence,
            "recorded_at": utc_now(),
        }
    )


def output(data: dict[str, Any]) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def ensure_worktree(repo: Path, worktree: Path, branch: str, base: str) -> None:
    repo = repo_root(repo)
    if worktree.exists():
        existing_root = repo_root(worktree)
        if existing_root == repo:
            raise RuntimeError("existing --worktree must be isolated; refusing to use the repo root")
        if git_common_dir(existing_root) != git_common_dir(repo):
            raise RuntimeError("existing --worktree is not attached to the target repo")
        existing_branch = current_ref(existing_root)
        if existing_branch != branch:
            raise RuntimeError(
                f"existing --worktree is on {existing_branch!r}, expected isolated branch {branch!r}"
            )
        status = run_command(["git", "-C", str(existing_root), "status", "--short"])
        if status.returncode != 0 or status.stdout.strip():
            raise RuntimeError("existing --worktree must be clean before Codex starts")
        return
    worktree.parent.mkdir(parents=True, exist_ok=True)
    result = run_command(
        ["git", "-C", str(repo), "worktree", "add", "-b", branch, str(worktree), base],
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())


def codex_command(worktree: Path, autonomy: str) -> list[str]:
    command = ["codex", "exec", "--cd", str(worktree)]
    if autonomy == "yolo":
        command.append("--dangerously-bypass-approvals-and-sandbox")
    else:
        command.extend(["-c", 'approval_policy="never"', "--sandbox", "workspace-write"])
    command.append("-")
    return command


def read_goal(args: argparse.Namespace) -> str:
    if args.goal_file:
        return Path(args.goal_file).read_text(encoding="utf-8")
    if args.goal:
        return args.goal
    raise RuntimeError("provide --goal or --goal-file")


def start_live_codex(state: dict[str, Any], prompt: str, timeout: int | None, wait: bool) -> None:
    worktree = Path(state["worktree"])
    log_path = Path(state["logs"]["builder"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = codex_command(worktree, state["autonomy"])
    state["command"] = shlex.join(command)
    state["status"] = "codex_starting"
    save_state(state)

    flags = {}
    if platform.system() == "Windows":
        flags["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    else:
        flags["start_new_session"] = True

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[{utc_now()}] starting: {state['command']}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=str(worktree),
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            **flags,
        )
        if process.stdin:
            process.stdin.write(prompt)
            process.stdin.close()

    state["pid"] = process.pid
    state["process_group"] = process.pid
    state["status"] = "codex_running"
    record_phase(
        state,
        "codex_builder",
        "running",
        {"pid": process.pid, "command": state["command"]},
    )
    save_state(state)

    if not wait:
        return

    try:
        exit_code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        stop_pid(process.pid)
        state["pid"] = None
        state["status"] = "timed_out"
        state["result"] = "timed_out"
        record_phase(state, "codex_builder", "timed_out", {"timeout_seconds": timeout})
        save_state(state)
        return

    state["pid"] = None
    state["codex_exit_code"] = exit_code
    if exit_code == 0:
        state["status"] = "codex_completed"
        state["result"] = "pending_verification"
        record_phase(state, "codex_builder", "completed", {"exit_code": exit_code})
    else:
        state["status"] = "human_review"
        state["result"] = "human_review"
        record_phase(state, "codex_builder", "failed", {"exit_code": exit_code})
    save_state(state)


def simulate_codex(state: dict[str, Any], goal: str, result: str) -> None:
    log_path = Path(state["logs"]["builder"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if result == "timeout":
        state["status"] = "timed_out"
        state["result"] = "timed_out"
        log_path.write_text("SIMULATED Codex timeout.\n", encoding="utf-8")
        record_phase(state, "codex_builder", "timed_out", {"simulated": True})
        save_state(state)
        return
    if result == "failure":
        state["status"] = "human_review"
        state["result"] = "human_review"
        log_path.write_text("SIMULATED Codex failure.\n", encoding="utf-8")
        record_phase(state, "codex_builder", "failed", {"simulated": True})
        save_state(state)
        return

    artifact = Path(state["worktree"]) / "codex_lane_simulated_output.txt"
    artifact.write_text(
        "\n".join(
            [
                "Simulated Codex builder output.",
                f"run_id: {state['run_id']}",
                f"task_id: {state['task_id']}",
                f"goal: {goal.strip()[:240]}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    log_path.write_text(
        f"SIMULATED Codex builder completed and wrote {artifact}.\n",
        encoding="utf-8",
    )
    state["status"] = "codex_completed"
    state["result"] = "pending_verification"
    state["artifacts"].append(str(artifact))
    state["command"] = "simulated codex builder"
    record_phase(
        state,
        "codex_builder",
        "completed",
        {"simulated": True, "artifact": str(artifact)},
    )
    save_state(state)


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def stop_pid(pid: int) -> str:
    if not process_alive(pid):
        return "not_running"
    if platform.system() == "Windows":
        result = subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            text=True,
            capture_output=True,
            check=False,
        )
        return "stopped" if result.returncode == 0 else "stop_failed"
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return "not_running"
    except OSError:
        os.kill(pid, signal.SIGTERM)
    time.sleep(0.5)
    if process_alive(pid):
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            return "stopped"
        except OSError:
            os.kill(pid, signal.SIGKILL)
    return "stopped"


def update_running_status(state: dict[str, Any]) -> None:
    pid = state.get("pid")
    if not pid or state.get("status") in FINAL_STATUSES:
        return
    if not process_alive(int(pid)):
        state["pid"] = None
        state["status"] = "human_review"
        state["result"] = "human_review"
        record_phase(state, "process", "exited_without_reconciliation", {"pid": pid})
        save_state(state)


def latest_phase_status(state: dict[str, Any], phase_name: str) -> str | None:
    for phase in reversed(state.get("phases", [])):
        if phase.get("name") == phase_name:
            return phase.get("status")
    return None


def claude_review_passed(state: dict[str, Any]) -> bool:
    return latest_phase_status(state, "claude_review") in {"review_passed", "completed"}


def parse_claude_verdict(stdout: str) -> tuple[str, str]:
    """Return (verdict, reason) from Claude Code JSON output."""
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        payload = None

    texts: list[str] = []
    if isinstance(payload, dict):
        direct = payload.get("verdict")
        if isinstance(direct, str) and direct.lower() in {"pass", "fail"}:
            return direct.lower(), "top-level verdict"
        for key in ("result", "content", "text", "message"):
            value = payload.get(key)
            if isinstance(value, str):
                texts.append(value)
        texts.append(json.dumps(payload, sort_keys=True))
    else:
        texts.append(stdout)

    combined = "\n".join(texts)
    try:
        embedded = json.loads(combined.strip())
    except json.JSONDecodeError:
        embedded = None
    if isinstance(embedded, dict):
        verdict = embedded.get("verdict")
        if isinstance(verdict, str) and verdict.lower() in {"pass", "fail"}:
            return verdict.lower(), "embedded verdict"

    match = VERDICT_RE.search(combined)
    if match:
        return match.group(1).lower(), "text verdict"
    return "unknown", "Claude output did not contain an explicit pass/fail verdict"


def command_preflight(args: argparse.Namespace) -> int:
    result = preflight_checks(
        Path(args.repo),
        require_codex=args.require_codex,
        require_claude=args.require_claude,
        simulate=args.simulate,
    )
    output(result)
    return 0 if result["success"] else 2


def command_run(args: argparse.Namespace) -> int:
    repo = repo_root(Path(args.repo))
    goal = read_goal(args)
    preflight = preflight_checks(
        repo,
        require_codex=True,
        require_claude=True,
        simulate=args.simulate,
    )
    if not preflight["success"]:
        output({"success": False, "preflight": preflight})
        return 2

    task_slug = slugify(args.task_id)
    run_id = args.run_id or f"{task_slug}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    state_root = Path(args.state_root).resolve()
    lane_dir = state_dir(state_root, run_id)
    base = args.base or current_ref(repo)
    branch = args.branch or f"codex/{task_slug}/{run_id.split('-')[-1]}"
    worktree = (
        Path(args.worktree).resolve()
        if args.worktree
        else Path(tempfile.gettempdir()) / f"hermes-codex-lane-{run_id}"
    )
    ensure_worktree(repo, worktree, branch, base)

    lane_dir.mkdir(parents=True, exist_ok=True)
    goal_path = lane_dir / "goal.txt"
    goal_path.write_text(goal, encoding="utf-8")
    state = {
        "run_id": run_id,
        "task_id": args.task_id,
        "repo": str(repo),
        "worktree": str(worktree),
        "branch": branch,
        "base": base,
        "mode": args.mode,
        "autonomy": args.autonomy,
        "status": "initialized",
        "result": "pending",
        "command": None,
        "pid": None,
        "process_group": None,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "state_path": str(lane_dir / "state.json"),
        "goal_path": str(goal_path),
        "logs": {
            "builder": str(lane_dir / "builder.log"),
            "review": str(lane_dir / "review.log"),
            "verify": str(lane_dir / "verify.log"),
        },
        "phases": [],
        "artifacts": [],
        "preflight": preflight,
        "codex_lane": {
            "used": True,
            "mode": args.mode,
            "run_id": run_id,
            "worktree": str(worktree),
            "branch": branch,
            "command": None,
            "reviewer": "claude-code",
            "hermes_verification": "required",
            "result": "pending",
            "accepted_commits": [],
            "rejected_reason": "",
            "tests_run": [],
            "artifacts": [],
        },
    }
    record_phase(state, "worktree", "ready", {"worktree": str(worktree), "branch": branch})
    save_state(state)

    run_blocked = False
    if args.simulate:
        simulate_codex(state, goal, args.simulate_result)
    else:
        if args.mode == "goal":
            run_blocked = True
            state["status"] = "human_review"
            state["result"] = "human_review"
            record_phase(
                state,
                "codex_builder",
                "blocked",
                {
                    "reason": "live Codex /goal requires an interactive PTY; use --mode exec or --simulate",
                },
            )
            save_state(state)
        else:
            start_live_codex(state, goal, args.timeout, args.wait)

    state = load_state(state_root, run_id)
    state["codex_lane"]["command"] = state.get("command")
    state["codex_lane"]["result"] = state.get("result", "pending")
    state["codex_lane"]["artifacts"] = state.get("artifacts", [])
    save_state(state)
    output({"success": not run_blocked, "run_id": run_id, "state": state})
    return 2 if run_blocked else 0


def command_status(args: argparse.Namespace) -> int:
    state = load_state(Path(args.state_root).resolve(), args.run_id)
    update_running_status(state)
    output({"success": True, "state": state})
    return 0


def command_logs(args: argparse.Namespace) -> int:
    state = load_state(Path(args.state_root).resolve(), args.run_id)
    names = ["builder", "review", "verify"] if args.phase == "all" else [args.phase]
    logs: dict[str, str] = {}
    for name in names:
        path = Path(state["logs"][name])
        logs[name] = path.read_text(encoding="utf-8")[-args.limit :] if path.exists() else ""
    output({"success": True, "run_id": args.run_id, "logs": logs})
    return 0


def command_stop(args: argparse.Namespace) -> int:
    state = load_state(Path(args.state_root).resolve(), args.run_id)
    pid = state.get("pid")
    outcome = "not_running"
    if pid:
        outcome = stop_pid(int(pid))
    state["pid"] = None
    state["status"] = "stopped" if outcome in {"stopped", "not_running"} else "human_review"
    state["result"] = "human_review"
    record_phase(state, "stop", outcome, {"pid": pid})
    state["codex_lane"]["result"] = state["result"]
    save_state(state)
    output({"success": outcome in {"stopped", "not_running"}, "outcome": outcome, "state": state})
    return 0 if outcome in {"stopped", "not_running"} else 3


def command_review(args: argparse.Namespace) -> int:
    state = load_state(Path(args.state_root).resolve(), args.run_id)
    worktree = Path(state["worktree"])
    log_path = Path(state["logs"]["review"])
    if args.simulate:
        verdict = args.simulate_verdict
        status = "review_passed" if verdict == "pass" else "rework_required"
        state["status"] = status
        state["result"] = "pending_verification" if verdict == "pass" else "rework"
        log_path.write_text(f"SIMULATED Claude review verdict: {verdict}\n", encoding="utf-8")
        record_phase(state, "claude_review", status, {"simulated": True, "verdict": verdict})
        state["codex_lane"]["result"] = state["result"]
        save_state(state)
        output({"success": True, "state": state})
        return 0

    preflight = preflight_checks(worktree, require_codex=False, require_claude=True, simulate=False)
    if not preflight["success"]:
        state["status"] = "human_review"
        state["result"] = "human_review"
        record_phase(state, "claude_review", "preflight_failed", {"preflight": preflight})
        state["codex_lane"]["result"] = state["result"]
        save_state(state)
        output({"success": False, "preflight": preflight, "state": state})
        return 2

    prompt = "\n".join(
        [
            "Review the current uncommitted implementation in this repository.",
            "Treat Codex self-report as untrusted.",
            "Compare the diff against the Hermes Codex Goal Lane objective.",
            "Report blocking defects, missing tests, unsafe changes, and acceptance risk.",
            "End with a JSON object containing verdict: pass or fail, and blocking_findings: list.",
            f"Goal file: {state['goal_path']}",
        ]
    )
    command = ["claude", "-p", prompt, "--output-format", "json", "--max-turns", str(args.max_turns)]
    result = run_command(command, cwd=worktree, timeout=args.timeout)
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    state["review_command"] = shlex.join(command)
    state["review_exit_code"] = result.returncode
    verdict, verdict_reason = parse_claude_verdict(result.stdout)
    state["review_verdict"] = verdict
    if result.returncode == 0 and verdict == "pass":
        state["status"] = "review_passed"
        state["result"] = "pending_verification"
        record_phase(
            state,
            "claude_review",
            "review_passed",
            {"exit_code": result.returncode, "verdict_reason": verdict_reason},
        )
    elif result.returncode == 0 and verdict == "fail":
        state["status"] = "rework_required"
        state["result"] = "rework"
        record_phase(
            state,
            "claude_review",
            "rework_required",
            {"exit_code": result.returncode, "verdict_reason": verdict_reason},
        )
    else:
        state["status"] = "human_review"
        state["result"] = "human_review"
        record_phase(
            state,
            "claude_review",
            "inconclusive",
            {
                "exit_code": result.returncode,
                "verdict": verdict,
                "verdict_reason": verdict_reason,
            },
        )
    state["codex_lane"]["result"] = state["result"]
    save_state(state)
    success = result.returncode == 0 and verdict == "pass"
    output({"success": success, "state": state})
    return 0 if success else 3


def command_verify(args: argparse.Namespace) -> int:
    state = load_state(Path(args.state_root).resolve(), args.run_id)
    worktree = Path(state["worktree"])
    log_path = Path(state["logs"]["verify"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.command and not args.simulate_pass:
        state["status"] = "human_review"
        state["result"] = "human_review"
        reason = "refusing to verify or accept without an independent verification command"
        record_phase(state, "hermes_verify", "failed", {"reason": reason})
        state["codex_lane"]["rejected_reason"] = reason
        state["codex_lane"]["result"] = state["result"]
        save_state(state)
        output({"success": False, "state": state})
        return 2

    if args.accept and not claude_review_passed(state):
        state["status"] = "human_review"
        state["result"] = "human_review"
        reason = "refusing to accept before a successful Claude review phase"
        record_phase(state, "hermes_verify", "failed", {"reason": reason})
        state["codex_lane"]["rejected_reason"] = reason
        state["codex_lane"]["result"] = state["result"]
        save_state(state)
        output({"success": False, "state": state})
        return 2

    tests_run: list[dict[str, Any]] = []
    with log_path.open("a", encoding="utf-8") as log:
        if args.simulate_pass:
            log.write("SIMULATED Hermes verification passed.\n")
            tests_run.append(
                {
                    "command": "simulated hermes verification",
                    "exit_code": 0,
                    "owner": "hermes",
                    "simulated": True,
                }
            )
        for command in args.command:
            started = utc_now()
            log.write(f"[{started}] running: {command}\n")
            try:
                result = run_shell_command(command, cwd=worktree, timeout=args.timeout)
            except subprocess.TimeoutExpired:
                tests_run.append(
                    {
                        "command": command,
                        "exit_code": None,
                        "owner": "hermes",
                        "timed_out": True,
                    }
                )
                log.write(f"[{utc_now()}] timed out after {args.timeout}s\n")
                continue
            log.write(result.stdout)
            log.write(result.stderr)
            log.write(f"[{utc_now()}] exit_code={result.returncode}\n")
            tests_run.append(
                {
                    "command": command,
                    "exit_code": result.returncode,
                    "owner": "hermes",
                    "started_at": started,
                }
            )

    state["codex_lane"]["tests_run"].extend(tests_run)
    failed = [item for item in tests_run if item.get("exit_code") != 0]
    if failed:
        state["status"] = "human_review"
        state["result"] = "human_review"
        state["codex_lane"]["rejected_reason"] = "Hermes verification failed"
        record_phase(state, "hermes_verify", "failed", {"tests_run": tests_run})
    elif args.accept:
        state["status"] = "accepted"
        state["result"] = "accepted"
        record_phase(state, "hermes_verify", "accepted", {"tests_run": tests_run})
    else:
        state["status"] = "verified"
        state["result"] = "pending_acceptance"
        record_phase(state, "hermes_verify", "verified", {"tests_run": tests_run})
    state["codex_lane"]["result"] = state["result"]
    state["codex_lane"]["artifacts"] = state.get("artifacts", []) + [str(log_path)]
    save_state(state)
    output({"success": not failed, "state": state})
    return 0 if not failed else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(func=None)
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--repo", required=True)
    preflight.add_argument("--require-codex", action="store_true", default=True)
    preflight.add_argument("--require-claude", action="store_true", default=True)
    preflight.add_argument("--simulate", action="store_true")
    preflight.set_defaults(func=command_preflight)

    run = subparsers.add_parser("run")
    run.add_argument("--repo", required=True)
    run.add_argument("--task-id", required=True)
    run.add_argument("--goal")
    run.add_argument("--goal-file")
    run.add_argument("--state-root", default=str(default_state_root()))
    run.add_argument("--run-id")
    run.add_argument("--worktree")
    run.add_argument("--branch")
    run.add_argument("--base")
    run.add_argument("--mode", choices=["auto", "exec", "goal"], default="auto")
    run.add_argument("--autonomy", choices=["full-auto", "yolo"], default="yolo")
    run.add_argument("--simulate", action="store_true")
    run.add_argument("--simulate-result", choices=["success", "timeout", "failure"], default="success")
    run.add_argument("--wait", action="store_true")
    run.add_argument("--timeout", type=int, default=1800)
    run.set_defaults(func=command_run)

    status = subparsers.add_parser("status")
    status.add_argument("--run-id", required=True)
    status.add_argument("--state-root", default=str(default_state_root()))
    status.set_defaults(func=command_status)

    logs = subparsers.add_parser("logs")
    logs.add_argument("--run-id", required=True)
    logs.add_argument("--state-root", default=str(default_state_root()))
    logs.add_argument("--phase", choices=["all", "builder", "review", "verify"], default="all")
    logs.add_argument("--limit", type=int, default=12000)
    logs.set_defaults(func=command_logs)

    stop = subparsers.add_parser("stop")
    stop.add_argument("--run-id", required=True)
    stop.add_argument("--state-root", default=str(default_state_root()))
    stop.set_defaults(func=command_stop)

    review = subparsers.add_parser("review")
    review.add_argument("--run-id", required=True)
    review.add_argument("--state-root", default=str(default_state_root()))
    review.add_argument("--simulate", action="store_true")
    review.add_argument("--simulate-verdict", choices=["pass", "fail"], default="pass")
    review.add_argument("--max-turns", type=int, default=10)
    review.add_argument("--timeout", type=int, default=1800)
    review.set_defaults(func=command_review)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--run-id", required=True)
    verify.add_argument("--state-root", default=str(default_state_root()))
    verify.add_argument("--command", action="append", default=[])
    verify.add_argument("--simulate-pass", action="store_true")
    verify.add_argument("--accept", action="store_true")
    verify.add_argument("--timeout", type=int, default=1800)
    verify.set_defaults(func=command_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        output({"success": False, "error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
