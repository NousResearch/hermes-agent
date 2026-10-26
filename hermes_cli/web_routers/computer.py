"""Mission-control dashboard surface: the desktop screenshot ticker (computer view).

One file, one surface (``web/`` rules). Privacy posture: the dashboard is
token-authenticated, but the PHYSICAL screen is a step further — the whole
feature is opt-IN via ``dashboard.computer_view`` (config, default ``False``;
config_defaults.py carries the rationale). Capture uses the platform's own
screenshot command, is rate-capped so a stuck poller cannot hot-loop it, and
prunes to a small retention window so it cannot fill the disk. Everything the
command does lives inside ``~/.hermes/screenshots/``.

The router is surfaced-only; tests mount it standalone (web_server's auth
middleware is app-level, inherited by the mount — see web/ AGENTS.md).
"""

import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from hermes_cli.web_deps import late
from hermes_cli.web_routers._common import log as _log

router = APIRouter()

# Late-bound so a test's monkeypatch on the owning module wins at call time.
load_config = late("load_config", "hermes_cli.config")

_MIN_CAPTURE_INTERVAL_S = 3.0  # rate cap: within this window re-serve the latest capture
_MAX_KEEP = 30                 # retention cap (screenshots dir pruned to this many)
_SUBPROC_TIMEOUT_S = 15.0

_lock = threading.Lock()


def _capture_dir() -> Path:
    """Screenshots subdir INSIDE an already-served profile home (never creates a home —
    same refusal semantics as cron's _ensure_cron_dir: a missing home fails closed)."""
    from hermes_constants import get_hermes_dir, get_hermes_home

    home = get_hermes_home()
    if not home.is_dir():
        raise HTTPException(status_code=503, detail=(
            "Profile home is missing; refusing to create directories under it"))
    d = get_hermes_dir("screenshots", "screenshots")
    d.mkdir(parents=True, exist_ok=True)
    return d


def computer_view_enabled() -> bool:
    """The single reader of ``dashboard.computer_view``. Default False (opt-in)."""
    try:
        cfg = load_config() or {}
        if not isinstance(cfg, dict):
            return False
        return bool((cfg.get("dashboard") or {}).get("computer_view"))
    except Exception:  # health: allow BLE001 -- config read failure must not take the ticker down; logged with traceback, opt-out default
        _log.exception("computer_view config read failed; defaulting to disabled")
        return False


def _screenshots() -> list[Path]:
    cap = _capture_dir()
    return sorted(
        (p for p in cap.iterdir() if p.is_file() and p.suffix.lower() == ".png"),
        key=lambda p: p.stat().st_mtime,
    )


def _prune() -> None:
    extras = _screenshots()[: max(0, len(_screenshots()) - _MAX_KEEP)]
    for old in extras:
        old.unlink(missing_ok=True)


def _capture_command(dest: Path) -> list[str]:
    return ["grim", str(dest)]


def computer_status() -> dict[str, Any]:
    enabled = computer_view_enabled()
    shots = _screenshots() if enabled else []
    return {
        "enabled": enabled,
        "last_capture_at": (shots[-1].stat().st_mtime if shots else None),
        "count": len(shots),
    }


def computer_screenshot(force: bool = False) -> FileResponse:
    if not computer_view_enabled():
        raise HTTPException(status_code=409, detail=(
            "Computer view is disabled — set dashboard.computer_view: true in config.yaml"))
    with _lock:
        shots = _screenshots()
        latest = shots[-1] if shots else None
        now = time.time()
        # Rate cap: within the window, serve the existing capture instead of
        # re-shelling the screenshot command (a poller cannot hot-loop it).
        if latest and not force and (now - latest.stat().st_mtime) < _MIN_CAPTURE_INTERVAL_S:
            return FileResponse(latest, media_type="image/png")
        dest = _capture_dir() / time.strftime("screen-%Y%m%d-%H%M%S.png")
        try:
            proc = subprocess_run(_capture_command(dest))
        except FileNotFoundError:
            raise HTTPException(status_code=503, detail=(
                "No screenshot command available (grim not installed)"))
        if proc.returncode != 0 or not dest.exists():
            err = (proc.stderr or "").strip()[-300:] or f"exit {proc.returncode}"
            dest.unlink(missing_ok=True)
            raise HTTPException(status_code=503, detail=f"Screenshot capture failed: {err}")
        _prune()
        shots = _screenshots()
        latest = shots[-1] if shots else dest
        return FileResponse(latest, media_type="image/png")


def subprocess_run(cmd: list[str], env: Optional[dict] = None) -> Any:
    """Test seam around the capture subprocess call."""
    return subprocess.run(
        cmd, capture_output=True, check=False, text=True, encoding="utf-8",
        errors="replace", timeout=_SUBPROC_TIMEOUT_S)


@router.get("/api/computer/status")
def api_computer_status():
    """Opt-in state + capture inventory for the Mission Control page."""
    return computer_status()


@router.get("/api/computer/screenshot")
def api_computer_screenshot():
    """The ticker: capture a fresh screenshot (rate-capped) and serve the PNG."""
    return computer_screenshot()