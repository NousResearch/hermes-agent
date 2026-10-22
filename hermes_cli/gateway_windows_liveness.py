"""Bounded Windows Scheduled-Task consumer for gateway liveness files."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)
STALE_AFTER_S = 180
COOLDOWN_S = 600


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _timestamp(value: object) -> float | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).timestamp()
    except (TypeError, ValueError):
        return None


def _active_fence(home: Path, now: float) -> bool:
    marker = _read_json(home / ".gateway-planned-stop.json")
    written = _timestamp(marker.get("written_at"))
    return written is not None and 0 <= now - written <= 60


def inspect(home: Path, *, now: float | None = None) -> str:
    """Return healthy, dead, stalled, or unknown without treating missing data as death."""
    now = time.time() if now is None else now
    try:
        from hermes_cli.gateway import PROJECT_ROOT
        update_in_progress = (PROJECT_ROOT / ".hermes-update-in-progress").exists()
    except Exception:
        update_in_progress = False
    if _active_fence(home, now) or (home / ".update-incomplete").exists() or update_in_progress:
        return "fenced"
    heartbeat = _read_json(home / "state" / "gateway.heartbeat")
    seen = _timestamp(heartbeat.get("updated_at"))
    if seen is None or not 0 <= now - seen > STALE_AFTER_S:
        return "healthy" if seen is not None else "unknown"
    runtime = _read_json(home / "gateway_state.json")
    pid = runtime.get("pid", heartbeat.get("pid"))
    try:
        from gateway.status import _pid_exists
        alive = type(pid) is int and pid > 0 and bool(_pid_exists(pid))
    except Exception:
        return "unknown"
    return "stalled" if alive else "dead"


def _write(path: Path, value: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    except OSError:
        logger.warning("Could not persist Windows gateway liveness alert", exc_info=True)


def run(home: Path | None = None, *, now: float | None = None) -> str:
    """Alert once per cooldown and re-run the registered gateway task for a confirmed death."""
    if home is None:
        from hermes_cli.gateway_windows import _hermes_home
        home = _hermes_home()
    now = time.time() if now is None else now
    verdict = inspect(home, now=now)
    if verdict not in {"dead", "stalled"}:
        return verdict
    state_path = home / "state" / "windows-gateway-liveness.json"
    state = _read_json(state_path)
    if now - float(state.get("alerted_at", 0) or 0) >= COOLDOWN_S:
        alert = {"kind": verdict, "at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
                 "message": f"Windows gateway {verdict}; inspect hermes gateway status --deep"}
        _write(home / "state" / "gateway-needs-attention.json", alert)
        logger.warning(alert["message"])
        state["alerted_at"] = now
    if verdict == "dead" and now - float(state.get("recovery_at", 0) or 0) >= COOLDOWN_S:
        try:
            from hermes_cli.gateway_windows import _exec_schtasks, get_task_name
            code, _out, err = _exec_schtasks(["/Run", "/TN", get_task_name()])
            if code == 0:
                state["recovery_at"] = now
            else:
                state["recovery_error"] = err[:300]
        except Exception:
            logger.warning("Could not request Windows gateway recovery", exc_info=True)
    _write(state_path, state)
    return verdict
