#!/usr/bin/env python3
"""Check and repair local agentmemory wiring for Codex/Hermes worktrees."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


HOME = Path.home()
AGENTMEMORY_ENV = HOME / ".agentmemory" / ".env"
CODEX_CONFIG = HOME / ".codex" / "config.toml"
CODEX_HOOKS = HOME / ".codex" / "hooks.json"
ZSHENV = HOME / ".zshenv"
AM_BIN_DIR = HOME / ".agentmemory" / "bin"
AM_URL = "http://localhost:3111"
HOOK_ENV_PREFIX = "env AGENTMEMORY_URL=http://localhost:3111 AGENTMEMORY_INJECT_CONTEXT=true"
HOOK_SCRIPT_FRAGMENT = "/@agentmemory/agentmemory/plugin/scripts/"


REQUIRED_FLAGS = {
    "AGENTMEMORY_AUTO_COMPRESS": "true",
    "AGENTMEMORY_INJECT_CONTEXT": "true",
    "CONSOLIDATION_ENABLED": "true",
    "GRAPH_EXTRACTION_ENABLED": "true",
    "AGENTMEMORY_REFLECT": "true",
    "AGENTMEMORY_SLOTS": "true",
}

LOCAL_OMLX_FLAGS = {
    "OPENAI_BASE_URL": "http://127.0.0.1:8000/v1",
    "OPENAI_MODEL": "qwen3-4b-instruct-2507-4bit",
    "EMBEDDING_PROVIDER": "local",
}


class Result:
    def __init__(self, name: str, ok: bool, detail: str = "", fix: str = "") -> None:
        self.name = name
        self.ok = ok
        self.detail = detail
        self.fix = fix


def run(cmd: list[str], *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)


def http_json(path: str, *, method: str = "GET", body: dict[str, Any] | None = None) -> tuple[int, Any]:
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(f"{AM_URL}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as res:
            raw = res.read().decode()
            return res.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        try:
            payload: Any = json.loads(raw)
        except json.JSONDecodeError:
            payload = raw
        return exc.code, payload
    except Exception as exc:
        return 0, {"error": str(exc)}


def parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if " #" in value:
            value = value.split(" #", 1)[0].strip()
        values[key.strip()] = value.strip("\"'")
    return values


def backup(path: Path) -> Path:
    stamp = time.strftime("%Y%m%d%H%M%S")
    target = path.with_name(f"{path.name}.bak-agentmemory-onboard-{stamp}")
    shutil.copy2(path, target)
    return target


def ensure_env_values(values: dict[str, str]) -> bool:
    AGENTMEMORY_ENV.parent.mkdir(parents=True, exist_ok=True)
    before = AGENTMEMORY_ENV.read_text() if AGENTMEMORY_ENV.exists() else ""
    lines = before.splitlines()
    seen: set[str] = set()
    after_lines: list[str] = []
    changed = False
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in values:
                replacement = f"{key}={values[key]}"
                after_lines.append(replacement)
                seen.add(key)
                changed = changed or line != replacement
                continue
        after_lines.append(line)

    missing = [key for key in values if key not in seen]
    if missing:
        if after_lines and after_lines[-1].strip():
            after_lines.append("")
        after_lines.append("# Added by scripts/agentmemory_onboard.py")
        changed = True
    for key in missing:
        after_lines.append(f"{key}={values[key]}")

    if not changed:
        return False
    if AGENTMEMORY_ENV.exists():
        backup(AGENTMEMORY_ENV)
    AGENTMEMORY_ENV.write_text("\n".join(after_lines).rstrip() + "\n")
    return True


def ensure_zshenv_path() -> bool:
    before = ZSHENV.read_text() if ZSHENV.exists() else ""
    bin_path = str(AM_BIN_DIR)
    current_path = os.environ.get("PATH", "")
    if bin_path not in current_path.split(":"):
        os.environ["PATH"] = f"{bin_path}:{current_path}" if current_path else bin_path
    if bin_path in before:
        return False
    if ZSHENV.exists():
        backup(ZSHENV)
    if "export PATH=" in before:
        after = before.replace('export PATH="', f'export PATH="{bin_path}:', 1)
    else:
        after = before.rstrip() + f'\nexport PATH="{bin_path}:$PATH"\n'
    ZSHENV.write_text(after)
    return True


def ensure_hook_env() -> bool:
    if not CODEX_HOOKS.exists():
        return False
    data = json.loads(CODEX_HOOKS.read_text())
    changed = False

    def visit(obj: Any) -> None:
        nonlocal changed
        if isinstance(obj, dict):
            command = obj.get("command")
            if (
                isinstance(command, str)
                and HOOK_SCRIPT_FRAGMENT in command
                and "AGENTMEMORY_INJECT_CONTEXT=true" not in command
            ):
                obj["command"] = f"{HOOK_ENV_PREFIX} {command}"
                changed = True
            for value in obj.values():
                visit(value)
        elif isinstance(obj, list):
            for value in obj:
                visit(value)

    visit(data)
    if changed:
        backup(CODEX_HOOKS)
        CODEX_HOOKS.write_text(json.dumps(data, indent=2) + "\n")
    return changed


def check_agentmemory_binary() -> Result:
    path = shutil.which("agentmemory")
    if not path:
        return Result("agentmemory binary", False, "not found", "Install @agentmemory/agentmemory.")
    version = run(["agentmemory", "--version"]).stdout.strip()
    return Result("agentmemory binary", True, f"{path} v{version}")


def check_server() -> Result:
    status, payload = http_json("/agentmemory/health")
    if status == 200 and payload.get("status") == "healthy":
        return Result("server health", True, "healthy")
    return Result("server health", False, f"HTTP {status}: {payload}", "Start agentmemory.")


def check_status() -> Result:
    proc = run(["agentmemory", "status"], timeout=20)
    if proc.returncode != 0:
        return Result("agentmemory status", False, proc.stderr.strip() or proc.stdout.strip())
    wanted = [
        "GRAPH_EXTRACTION_ENABLED",
        "CONSOLIDATION_ENABLED",
        "AGENTMEMORY_AUTO_COMPRESS",
        "AGENTMEMORY_INJECT_CONTEXT",
    ]
    missing = [flag for flag in wanted if flag not in proc.stdout]
    if missing:
        return Result("agentmemory status", False, f"missing flags: {', '.join(missing)}")
    return Result("agentmemory status", True, "core flags enabled")


def check_env(local_omlx: bool) -> list[Result]:
    values = parse_env(AGENTMEMORY_ENV)
    checks: list[Result] = []
    for key, want in REQUIRED_FLAGS.items():
        got = values.get(key)
        checks.append(Result(f"env {key}", got == want, got or "unset", f"Set {key}={want}"))
    if local_omlx:
        for key, want in LOCAL_OMLX_FLAGS.items():
            got = values.get(key)
            checks.append(Result(f"env {key}", got == want, got or "unset", f"Set {key}={want}"))
    return checks


def check_slots() -> Result:
    status, payload = http_json("/agentmemory/slots")
    if status == 200:
        count = len(payload.get("slots", [])) if isinstance(payload, dict) else 0
        return Result("slots endpoint", True, f"{count} slots")
    if status == 503:
        return Result("slots endpoint", False, "disabled", "Set AGENTMEMORY_SLOTS=true and restart.")
    return Result("slots endpoint", False, f"HTTP {status}: {payload}")


def check_iii_path() -> Result:
    path = shutil.which("iii")
    if not path:
        return Result("iii on PATH", False, "not found", f"Add {AM_BIN_DIR} to PATH.")
    proc = run(["iii", "--version"])
    return Result("iii on PATH", proc.stdout.strip() == "0.11.2", f"{path} {proc.stdout.strip()}")


def check_codex_mcp() -> Result:
    text = CODEX_CONFIG.read_text() if CODEX_CONFIG.exists() else ""
    ok = "[mcp_servers.agentmemory]" in text and 'AGENTMEMORY_URL = "http://localhost:3111"' in text
    return Result("Codex MCP", ok, str(CODEX_CONFIG), "Run agentmemory connect codex.")


def check_hooks() -> Result:
    if not CODEX_HOOKS.exists():
        return Result("Codex hooks", False, "missing", "Run agentmemory connect codex with hooks.")
    text = CODEX_HOOKS.read_text()
    ok = "AGENTMEMORY_INJECT_CONTEXT=true" in text and "session-start.mjs" in text
    return Result("Codex hooks", ok, str(CODEX_HOOKS), "Patch hooks with AGENTMEMORY_INJECT_CONTEXT=true.")


def check_local_omlx() -> Result:
    try:
        req = urllib.request.Request("http://127.0.0.1:8000/v1/models")
        with urllib.request.urlopen(req, timeout=3) as res:
            payload = json.loads(res.read().decode())
        models = [m.get("id") for m in payload.get("data", [])]
        return Result("local oMLX", bool(models), ", ".join(models))
    except Exception as exc:
        return Result("local oMLX", False, str(exc), "Start `omlx serve` or disable --local-omlx expectations.")


def restart_agentmemory() -> None:
    run(["agentmemory", "stop", "--force"], timeout=20)
    run(["tmux", "kill-session", "-t", "agentmemory"], timeout=10)
    proc = run(
        ["tmux", "new-session", "-d", "-s", "agentmemory", "CI=1 agentmemory --verbose"],
        timeout=10,
    )
    if proc.returncode != 0:
        print(f"restart: failed to start tmux session: {proc.stderr.strip()}", file=sys.stderr)
        return
    for _ in range(30):
        status, payload = http_json("/agentmemory/health")
        if status == 200 and payload.get("status") == "healthy":
            return
        time.sleep(1)
    print("restart: agentmemory did not become healthy within 30s", file=sys.stderr)


def install_skills() -> None:
    proc = run(["npx", "-y", "skills", "add", "rohitg00/agentmemory", "--all", "-g"], timeout=120)
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)


def short(text: str, *, default: str = "", limit: int = 500) -> str:
    value = text.strip()
    if not value:
        return default
    return value[:limit]


def find_session(session_id: str) -> dict[str, Any] | None:
    status, payload = http_json("/agentmemory/sessions")
    if status != 200 or not isinstance(payload, dict):
        return None
    sessions = payload.get("sessions", [])
    if not isinstance(sessions, list):
        return None
    for session in sessions:
        if isinstance(session, dict) and session.get("id") == session_id:
            return session
    return None


def run_probe(project: str, *, strict_context: bool = False) -> Result:
    session_id = f"agentmemory-onboard-{time.time_ns()}"
    payload = json.dumps({"session_id": session_id, "cwd": os.getcwd()})
    script = "/opt/homebrew/lib/node_modules/@agentmemory/agentmemory/plugin/scripts/session-start.mjs"
    if not Path(script).exists():
        return Result("SessionStart probe", False, f"missing {script}", "Reinstall @agentmemory/agentmemory.")

    try:
        proc = subprocess.run(
            [
                "env",
                "AGENTMEMORY_URL=http://localhost:3111",
                "AGENTMEMORY_INJECT_CONTEXT=true",
                "node",
                script,
            ],
            input=payload,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return Result("SessionStart probe", False, "hook timed out after 10s", "Check daemon health and hook script.")

    if proc.returncode != 0:
        detail = short(proc.stderr, default=short(proc.stdout, default=f"exit {proc.returncode}"))
        return Result("SessionStart probe", False, detail, "Check hook script, Node, and agentmemory daemon.")

    expected = f'<agentmemory-context project="{project}">'
    if expected in proc.stdout:
        return Result("SessionStart probe", True, short(proc.stdout, default="context injected"))

    if "<agentmemory-context" in proc.stdout:
        return Result(
            "SessionStart probe",
            False,
            short(proc.stdout),
            f"Hook returned context, but not for expected project `{project}`.",
        )

    session = find_session(session_id)
    if session is None:
        return Result(
            "SessionStart probe",
            False,
            "hook returned no context and session was not registered",
            "Check hooks/env, daemon health, and /agentmemory/session/start.",
        )

    actual_project = session.get("project")
    if actual_project != project:
        return Result(
            "SessionStart probe",
            False,
            f"registered project `{actual_project}`, expected `{project}`",
            "Run from the intended git/worktree root or pass --project.",
        )

    detail = f"registered session {session_id}; no context injected"
    if strict_context:
        return Result(
            "SessionStart probe",
            False,
            detail,
            "Project may have no useful memories yet; save one or rerun without --strict-probe.",
        )
    return Result("SessionStart probe", True, f"{detail} (expected for sparse/new projects)")


def print_results(results: list[Result]) -> int:
    failures = [r for r in results if not r.ok]
    for result in results:
        mark = "OK" if result.ok else "FAIL"
        print(f"[{mark}] {result.name}: {result.detail}")
        if not result.ok and result.fix:
            print(f"      fix: {result.fix}")
    print()
    print(f"{len(results) - len(failures)}/{len(results)} checks passing")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Patch local config files.")
    parser.add_argument("--restart", action="store_true", help="Restart agentmemory after applying env changes.")
    parser.add_argument("--install-skills", action="store_true", help="Install agentmemory skills globally.")
    parser.add_argument("--probe", action="store_true", help="Run a SessionStart context-injection probe.")
    parser.add_argument("--strict-probe", action="store_true", help="Require SessionStart to inject non-empty context.")
    parser.add_argument("--local-omlx", action="store_true", help="Require and optionally apply local oMLX settings.")
    parser.add_argument("--project", default=Path.cwd().name, help="Expected project key for probe output.")
    args = parser.parse_args()

    if args.apply:
        values = dict(REQUIRED_FLAGS)
        if args.local_omlx:
            values.update(LOCAL_OMLX_FLAGS)
        ensure_env_values(values)
        ensure_zshenv_path()
        ensure_hook_env()

    if args.install_skills:
        install_skills()

    if args.restart:
        restart_agentmemory()

    results: list[Result] = [
        check_agentmemory_binary(),
        check_server(),
        check_status(),
        check_iii_path(),
        check_codex_mcp(),
        check_hooks(),
        check_slots(),
    ]
    results.extend(check_env(args.local_omlx))
    if args.local_omlx:
        results.append(check_local_omlx())
    if args.probe:
        results.append(run_probe(args.project, strict_context=args.strict_probe))

    return print_results(results)


if __name__ == "__main__":
    raise SystemExit(main())
