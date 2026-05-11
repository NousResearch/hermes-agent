from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_cli.config import get_hermes_home

_CONFIG_PATH = get_hermes_home() / "open_design.json"
_LOG_PATH = get_hermes_home() / "logs" / "open-design.log"
_DEFAULT_WORKSPACE = get_hermes_home() / "open-design-workspace"
_ALLOWED_ARTIFACT_SUFFIXES = {
    ".html", ".htm", ".png", ".jpg", ".jpeg", ".webp", ".svg",
    ".gif", ".pdf", ".md", ".json", ".txt",
}

_DEFAULT_CONFIG: Dict[str, Any] = {
    "repo_path": "",
    "start_command": "pnpm tools-dev run web",
    "ui_url": "http://127.0.0.1:3000",
    "health_url": "http://127.0.0.1:3000",
    "workspace_dir": str(_DEFAULT_WORKSPACE),
    "artifacts_dir": "",
    "env": {},
}

_PROC: Optional[subprocess.Popen] = None
_LAST_START_AT: float | None = None
_LAST_ERROR: str | None = None


def _ensure_parent_dirs() -> None:
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _DEFAULT_WORKSPACE.mkdir(parents=True, exist_ok=True)


def load_config() -> Dict[str, Any]:
    _ensure_parent_dirs()
    if not _CONFIG_PATH.exists():
        return dict(_DEFAULT_CONFIG)
    try:
        raw = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return dict(_DEFAULT_CONFIG)
    config = dict(_DEFAULT_CONFIG)
    if isinstance(raw, dict):
        config.update(raw)
    env = config.get("env")
    config["env"] = env if isinstance(env, dict) else {}
    return config


def save_config(updates: Dict[str, Any]) -> Dict[str, Any]:
    config = load_config()
    for key in _DEFAULT_CONFIG:
        if key in updates:
            config[key] = updates[key]
    config["env"] = config.get("env") if isinstance(config.get("env"), dict) else {}
    _ensure_parent_dirs()
    _CONFIG_PATH.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return config


def _tail_lines(path: Path, limit: int) -> List[str]:
    if not path.exists():
        return []
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    lines = data.splitlines()
    return lines[-limit:] if limit > 0 else lines


def _http_ok(url: str, timeout: float = 2.0) -> bool:
    if not url:
        return False
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "HermesDashboard/OpenDesign"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= getattr(resp, "status", 200) < 500
    except Exception:
        return False


def _proc_running() -> bool:
    global _PROC
    return _PROC is not None and _PROC.poll() is None


def _current_pid() -> int | None:
    if _PROC is None:
        return None
    try:
        return int(_PROC.pid)
    except Exception:
        return None


def _update_last_error_from_proc() -> None:
    global _LAST_ERROR
    if _PROC is None:
        return
    exit_code = _PROC.poll()
    if exit_code is not None and exit_code != 0:
        _LAST_ERROR = f"Open Design process exited with code {exit_code}."


def status() -> Dict[str, Any]:
    config = load_config()
    repo_path = str(config.get("repo_path") or "")
    repo_exists = bool(repo_path) and Path(repo_path).exists()
    process_running = _proc_running()
    health_ok = _http_ok(str(config.get("health_url") or ""))
    _update_last_error_from_proc()

    state = "stopped"
    if process_running and not health_ok:
        state = "starting"
    elif process_running or health_ok:
        state = "running"
    elif _LAST_ERROR:
        state = "error"

    return {
        "state": state,
        "configured": repo_exists,
        "repo_exists": repo_exists,
        "process_running": process_running,
        "health_ok": health_ok,
        "pid": _current_pid(),
        "last_start_at": _LAST_START_AT,
        "last_error": _LAST_ERROR,
        "log_path": str(_LOG_PATH),
        "config": config,
        "ui_url": str(config.get("ui_url") or ""),
        "health_url": str(config.get("health_url") or ""),
    }


def _build_env(config: Dict[str, Any]) -> Dict[str, str]:
    env = dict(os.environ)
    extra = config.get("env") or {}
    if isinstance(extra, dict):
        for key, value in extra.items():
            if value is None:
                continue
            env[str(key)] = str(value)
    env.setdefault("CI", "1")
    return env


def start() -> Dict[str, Any]:
    global _PROC, _LAST_START_AT, _LAST_ERROR
    config = load_config()
    repo_path = Path(str(config.get("repo_path") or "")).expanduser()
    if not str(repo_path):
        raise ValueError("repo_path is not configured")
    if not repo_path.exists():
        raise ValueError(f"repo_path does not exist: {repo_path}")
    if _proc_running() or _http_ok(str(config.get("health_url") or "")):
        return status()

    _ensure_parent_dirs()
    command = str(config.get("start_command") or "").strip()
    if not command:
        raise ValueError("start_command is empty")

    log_file = open(_LOG_PATH, "ab", buffering=0)
    log_file.write(
        f"\n=== open-design started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n".encode()
    )

    kwargs: Dict[str, Any] = {
        "cwd": str(repo_path),
        "stdin": subprocess.DEVNULL,
        "stdout": log_file,
        "stderr": subprocess.STDOUT,
        "env": _build_env(config),
        "shell": True,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
            | getattr(subprocess, "DETACHED_PROCESS", 0)
        )
    else:
        kwargs["start_new_session"] = True

    _PROC = subprocess.Popen(command, **kwargs)
    _LAST_START_AT = time.time()
    _LAST_ERROR = None
    return status()


def stop() -> Dict[str, Any]:
    global _PROC, _LAST_ERROR
    if _PROC is None or _PROC.poll() is not None:
        _PROC = None
        return status()
    try:
        if sys.platform == "win32":
            _PROC.terminate()
        else:
            os.killpg(_PROC.pid, signal.SIGTERM)
        _PROC.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if sys.platform == "win32":
            _PROC.kill()
        else:
            os.killpg(_PROC.pid, signal.SIGKILL)
        _PROC.wait(timeout=5)
    except Exception as exc:
        _LAST_ERROR = f"Failed to stop Open Design: {exc}"
        raise
    finally:
        _PROC = None
    return status()


def restart() -> Dict[str, Any]:
    try:
        stop()
    except Exception:
        pass
    return start()


def get_logs(lines: int = 200) -> Dict[str, Any]:
    return {
        "path": str(_LOG_PATH),
        "lines": _tail_lines(_LOG_PATH, min(max(lines, 1), 2000)),
    }


def push_brief(
    *,
    brief: str,
    project_name: str = "",
    skill: str = "",
    design_system: str = "",
) -> Dict[str, Any]:
    config = load_config()
    workspace_dir = Path(str(config.get("workspace_dir") or _DEFAULT_WORKSPACE)).expanduser()
    workspace_dir.mkdir(parents=True, exist_ok=True)

    slug = (project_name or f"brief-{time.strftime('%Y%m%d-%H%M%S')}").strip()
    safe_slug = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "-" for ch in slug).strip("-")
    safe_slug = safe_slug or f"brief-{time.strftime('%Y%m%d-%H%M%S')}"
    project_dir = workspace_dir / safe_slug
    project_dir.mkdir(parents=True, exist_ok=True)

    brief_path = project_dir / "brief.md"
    brief_path.write_text(brief.rstrip() + "\n", encoding="utf-8")

    metadata = {
        "project_name": project_name,
        "skill": skill,
        "design_system": design_system,
        "created_at": time.time(),
    }
    (project_dir / "brief.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if design_system.strip():
        (project_dir / "DESIGN.md").write_text(
            f"# Design System\n\n{design_system.strip()}\n",
            encoding="utf-8",
        )

    return {
        "ok": True,
        "project_dir": str(project_dir),
        "brief_path": str(brief_path),
        "files": [
            str(project_dir / "brief.md"),
            str(project_dir / "brief.json"),
            *(([str(project_dir / "DESIGN.md")] ) if design_system.strip() else []),
        ],
    }


def _artifact_roots(config: Dict[str, Any]) -> List[Path]:
    roots: List[Path] = []
    for raw in [config.get("artifacts_dir"), config.get("workspace_dir")]:
        if not raw:
            continue
        path = Path(str(raw)).expanduser()
        if path.exists() and path.is_dir():
            roots.append(path)
    return roots


def list_artifacts(limit: int = 100) -> List[Dict[str, Any]]:
    config = load_config()
    items: List[Dict[str, Any]] = []
    for root in _artifact_roots(config):
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in _ALLOWED_ARTIFACT_SUFFIXES:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            items.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "root": str(root),
                    "relative_path": str(path.relative_to(root)),
                    "size": int(stat.st_size),
                    "mtime": float(stat.st_mtime),
                }
            )
    items.sort(key=lambda item: item["mtime"], reverse=True)
    return items[: max(1, min(limit, 500))]


def resolve_artifact_path(raw_path: str) -> Path:
    config = load_config()
    candidate = Path(raw_path).expanduser().resolve()
    for root in _artifact_roots(config):
        resolved_root = root.resolve()
        try:
            candidate.relative_to(resolved_root)
            return candidate
        except ValueError:
            continue
    raise ValueError("Artifact path is outside the allowed roots")


def open_ui() -> Dict[str, Any]:
    config = load_config()
    url = str(config.get("ui_url") or "").strip()
    if not url:
        raise ValueError("ui_url is not configured")
    webbrowser.open(url)
    return {"ok": True, "url": url}
