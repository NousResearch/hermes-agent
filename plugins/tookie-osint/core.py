from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home


REPO_URL = "https://github.com/Alfredredbird/tookie-osint.git"
DEFAULT_TIMEOUT_SECONDS = 240
MAX_TIMEOUT_SECONDS = 3600
MAX_THREADS = 32
USERNAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]")


STATUS_SCHEMA = {
    "description": "Report Tookie-OSINT checkout and dependency readiness.",
    "type": "object",
    "properties": {},
}

SCAN_SCHEMA = {
    "description": "Scan public sites for a username through Tookie-OSINT.",
    "type": "object",
    "properties": {
        "username": {
            "type": "string",
            "description": "Username to scan. Path separators and shell fragments are rejected.",
        },
        "threads": {
            "type": "integer",
            "minimum": 1,
            "maximum": MAX_THREADS,
            "default": 4,
            "description": "Worker threads for non-browser scans.",
        },
        "output_format": {
            "type": "string",
            "enum": ["json", "csv", "txt"],
            "default": "json",
            "description": "Tookie output file format.",
        },
        "include_all": {
            "type": "boolean",
            "default": False,
            "description": "Include negative results as well as matches.",
        },
        "skip_headers": {
            "type": "boolean",
            "default": False,
            "description": "Skip Tookie's randomized User-Agent header list.",
        },
        "webscraper": {
            "type": "boolean",
            "default": False,
            "description": "Use Tookie's Selenium webscraper mode. Requires selenium and webdriver-manager.",
        },
        "harvest": {
            "type": "boolean",
            "default": False,
            "description": "Harvest configured fields when webscraper mode is enabled.",
        },
        "delay": {
            "type": "integer",
            "minimum": 0,
            "maximum": 60,
            "description": "Page-load delay for webscraper mode.",
        },
        "timeout_seconds": {
            "type": "integer",
            "minimum": 10,
            "maximum": MAX_TIMEOUT_SECONDS,
            "default": DEFAULT_TIMEOUT_SECONDS,
            "description": "Local subprocess timeout.",
        },
    },
    "required": ["username"],
}


def to_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _state_dir() -> Path:
    path = get_hermes_home() / "tookie-osint"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _root_file() -> Path:
    return _state_dir() / "root.txt"


def _runs_dir() -> Path:
    path = _state_dir() / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _coerce_path(value: str | os.PathLike[str]) -> Path:
    return Path(value).expanduser().resolve()


def _valid_root(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "brib.py").is_file()
        and (path / "sites" / "sites.json").is_file()
        and (path / "config" / "version").is_file()
    )


def _read_saved_root() -> Path | None:
    path = _root_file()
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8", errors="replace").strip()
    return _coerce_path(value) if value else None


def resolve_root() -> Path | None:
    env_root = os.environ.get("TOOKIE_OSINT_ROOT", "").strip()
    if env_root:
        return _coerce_path(env_root)
    return _read_saved_root()


def save_root(root: str | os.PathLike[str]) -> Path:
    path = _coerce_path(root)
    if not _valid_root(path):
        raise FileNotFoundError(
            f"{path} is not a Tookie-OSINT checkout with brib.py, sites/sites.json, and config/version"
        )
    root_file = _root_file()
    root_file.write_text(str(path), encoding="utf-8")
    return path


def _missing_imports(*, webscraper: bool = False) -> list[str]:
    packages = ["colorama", "requests"]
    if webscraper:
        packages.extend(["selenium", "webdriver_manager"])
    return [name for name in packages if importlib.util.find_spec(name) is None]


def _git_head(root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _site_count(root: Path) -> int:
    try:
        data = json.loads((root / "sites" / "sites.json").read_text(encoding="utf-8"))
    except Exception:
        return 0
    return len(data) if isinstance(data, list) else 0


def status_payload(_values: dict[str, Any] | None = None) -> dict[str, Any]:
    root = resolve_root()
    root_valid = bool(root and _valid_root(root))
    missing_basic = _missing_imports(webscraper=False)
    missing_browser = _missing_imports(webscraper=True)
    payload: dict[str, Any] = {
        "success": True,
        "available": root_valid and not missing_basic,
        "repo_url": REPO_URL,
        "root": str(root) if root else "",
        "root_exists": bool(root and root.exists()),
        "root_valid": root_valid,
        "saved_root_file": str(_root_file()),
        "missing_dependencies": missing_basic,
        "missing_webscraper_dependencies": missing_browser,
        "python": sys.executable,
        "runs_dir": str(_runs_dir()),
    }
    if root and root_valid:
        payload.update(
            {
                "version": (root / "config" / "version").read_text(
                    encoding="utf-8", errors="replace"
                ).strip(),
                "git_head": _git_head(root),
                "site_count": _site_count(root),
                "headers_file": str(root / "sites" / "headers.txt"),
                "headers_present": (root / "sites" / "headers.txt").is_file(),
            }
        )
    if not root:
        payload["setup_hint"] = "Run: hermes tookie-osint setup --root <path-to-tookie-osint>"
    elif not root_valid:
        payload["setup_hint"] = "Configured root is not a valid Tookie-OSINT checkout."
    elif missing_basic:
        payload["setup_hint"] = (
            f"Install dependencies with: {sys.executable} -m pip install -r "
            f"{root / 'requirements.txt'}"
        )
    return payload


def check_available() -> bool:
    status = status_payload({})
    return bool(status.get("available"))


def install_dependencies(root: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    resolved = _coerce_path(root) if root else resolve_root()
    if not resolved or not _valid_root(resolved):
        return {"success": False, "error": "Tookie-OSINT root is not configured or invalid."}
    req = resolved / "requirements.txt"
    if not req.is_file():
        return {"success": False, "error": f"requirements.txt not found: {req}"}
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(req)],
        cwd=str(resolved),
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    return {
        "success": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": _tail(proc.stdout),
        "stderr": _tail(proc.stderr),
    }


def _safe_filename(username: str) -> str:
    safe = UNSAFE_FILENAME_CHARS.sub("_", username)
    safe = safe.replace("..", "_").strip(".")
    return (safe or "output")[:128]


def _clean_username(value: Any) -> str:
    username = str(value or "").strip()
    if not USERNAME_RE.match(username):
        raise ValueError(
            "username must be 1-128 chars and contain only letters, digits, dot, underscore, or hyphen"
        )
    return username


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def _tail(value: str, *, limit: int = 12000) -> str:
    if len(value) <= limit:
        return value
    return value[-limit:]


def _output_file(run_dir: Path, username: str, output_format: str) -> Path:
    return run_dir / f"{_safe_filename(username)}.{output_format}"


def _load_result(path: Path, output_format: str) -> Any:
    if not path.is_file():
        return None
    if output_format == "json":
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return path.read_text(encoding="utf-8", errors="replace")


def scan_username(values: dict[str, Any] | None = None) -> dict[str, Any]:
    values = values or {}
    try:
        username = _clean_username(values.get("username"))
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    root = resolve_root()
    if not root or not _valid_root(root):
        return {
            "success": False,
            "error": "Tookie-OSINT root is not configured or invalid.",
            "status": status_payload({}),
        }

    output_format = str(values.get("output_format") or "json").lower()
    if output_format not in {"json", "csv", "txt"}:
        output_format = "json"
    webscraper = bool(values.get("webscraper", False))
    missing = _missing_imports(webscraper=webscraper)
    if missing:
        return {
            "success": False,
            "error": "Missing Tookie-OSINT Python dependencies.",
            "missing_dependencies": missing,
            "install_hint": (
                f"{sys.executable} -m pip install -r {root / 'requirements.txt'}"
            ),
        }

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = _runs_dir() / f"{timestamp}-{_safe_filename(username)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(root / "brib.py"), "-u", username, "-o", output_format]
    if webscraper:
        cmd.append("-W")
        if values.get("harvest"):
            cmd.append("-H")
        if values.get("delay") is not None:
            cmd.extend(["-D", str(_bounded_int(values.get("delay"), default=2, minimum=0, maximum=60))])
    else:
        cmd.extend(
            [
                "-t",
                str(
                    _bounded_int(
                        values.get("threads"), default=4, minimum=1, maximum=MAX_THREADS
                    )
                ),
            ]
        )
    if values.get("include_all"):
        cmd.append("-a")
    if values.get("skip_headers"):
        cmd.append("-sk")

    timeout_seconds = _bounded_int(
        values.get("timeout_seconds"),
        default=DEFAULT_TIMEOUT_SECONDS,
        minimum=10,
        maximum=MAX_TIMEOUT_SECONDS,
    )
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(run_dir),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "success": False,
            "error": f"Tookie-OSINT timed out after {timeout_seconds}s.",
            "run_dir": str(run_dir),
            "stdout": _tail(exc.stdout or ""),
            "stderr": _tail(exc.stderr or ""),
        }

    output_path = _output_file(run_dir, username, output_format)
    result = _load_result(output_path, output_format)
    return {
        "success": proc.returncode == 0,
        "returncode": proc.returncode,
        "command": cmd,
        "run_dir": str(run_dir),
        "output_path": str(output_path) if output_path.exists() else "",
        "result": result,
        "stdout": _tail(proc.stdout),
        "stderr": _tail(proc.stderr),
    }


def handle_slash(raw_args: str) -> str:
    parts = (raw_args or "").split()
    if not parts or parts[0] == "status":
        return to_json(status_payload({}))
    if parts[0] == "scan" and len(parts) >= 2:
        return to_json(scan_username({"username": parts[1]}))
    if parts[0] == "setup" and len(parts) >= 2:
        try:
            root = save_root(parts[1])
            return to_json({"success": True, "root": str(root), "status": status_payload({})})
        except Exception as exc:
            return to_json({"success": False, "error": str(exc)})
    return to_json(
        {
            "success": False,
            "usage": "/tookie-osint status | scan <username> | setup <path-to-tookie-osint>",
        }
    )
