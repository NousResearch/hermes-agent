import json
import re
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAST_SESSION_PATH = PROJECT_ROOT / "data" / "sessions" / "codex-last-session.json"
CODEX_EXECUTABLE = "/mnt/c/Users/AMT/AppData/Local/Programs/OpenAI/Codex/bin/codex.exe"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_last_session() -> dict[str, Any] | None:
    try:
        with LAST_SESSION_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None

    return data if isinstance(data, dict) else None


def save_last_session(session_summary: dict[str, Any]) -> None:
    ensure_parent(LAST_SESSION_PATH)
    with LAST_SESSION_PATH.open("w", encoding="utf-8") as handle:
        json.dump(session_summary, handle, indent=2)


def parse_session_id(output: str) -> str | None:
    for line in output.splitlines():
        match = re.search(r"session(?:\s+id|_id)?[:=\s]+([a-zA-Z0-9-]{8,})", line, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def run_exec(params: dict[str, Any]) -> dict[str, Any]:
    prompt = str(params.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Missing prompt")

    workdir = str(params.get("workdir") or r"D:\Codex")
    timeout = int(params.get("timeout") or 120)
    prompt_preview = prompt[:160]
    command = [
        CODEX_EXECUTABLE,
        "exec",
        "--skip-git-repo-check",
        "-C",
        workdir,
        prompt,
    ]

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"codex.exe not found at {CODEX_EXECUTABLE}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"codex.exe timed out after {timeout} seconds") from exc
    except OSError as exc:
        raise RuntimeError(f"Unable to start codex.exe: {exc}") from exc

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    combined_output = stdout if not stderr else f"{stdout}\n{stderr}".strip()
    if completed.returncode != 0:
        message = stderr.strip() or stdout.strip() or f"codex.exe exited with code {completed.returncode}"
        raise RuntimeError(message)

    session_id = parse_session_id(stdout) or parse_session_id(stderr) or str(uuid.uuid4())

    result = {
        "output": combined_output,
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": completed.returncode,
        "workdir": workdir,
        "prompt_preview": prompt_preview,
        "session_id": session_id,
        "timeout": timeout,
        "created_at": datetime.now(UTC).isoformat(),
    }
    save_last_session(result)
    return result


def handle_request(payload: dict[str, Any]) -> dict[str, Any]:
    request_id = payload.get("id", 1)
    method = payload.get("method")
    params = payload.get("params") or {}

    if method == "ping":
        return {"jsonrpc": "2.0", "result": "pong", "id": request_id}
    if method == "status":
        return {
            "jsonrpc": "2.0",
            "result": {
                "logged_in": True,
                "version": "0.142.0",
                "model": "gpt-5.4",
            },
            "id": request_id,
        }
    if method == "exec":
        try:
            return {"jsonrpc": "2.0", "result": run_exec(params), "id": request_id}
        except (ValueError, RuntimeError) as exc:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32602, "message": str(exc)},
                "id": request_id,
            }
    if method == "session.last":
        last_session = load_last_session()
        return {"jsonrpc": "2.0", "result": last_session, "id": request_id}

    return {
        "jsonrpc": "2.0",
        "error": {"code": -32601, "message": "Method not found"},
        "id": request_id,
    }


def main() -> None:
    for line in sys.stdin:
        raw = line.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
            response = handle_request(payload)
        except json.JSONDecodeError:
            response = {
                "jsonrpc": "2.0",
                "error": {"code": -32700, "message": "Parse error"},
                "id": None,
            }
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
