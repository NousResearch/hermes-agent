"""Lightweight agent runtime status tracker.

Best-effort only: all public entrypoints swallow failures so agent/runtime
callers can safely invoke them from hot paths.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

from hermes_constants import get_hermes_home
from utils import atomic_json_write

_WRITE_DEBOUNCE_SECONDS = 0.5

_lock = threading.Lock()
_last_write_ts = 0.0
_pending_write = False
_flush_timer: threading.Timer | None = None
_status: dict[str, Any] = {
    "state": "offline",
    "state_detail": None,
    "gateway": {"running": False, "pid": None},
    "attention": {"needs_attention": False, "reason": None},
    "platforms": {},
    "updated_at": None,
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _snapshot_locked() -> dict[str, Any]:
    return {
        "state": _status.get("state", "offline"),
        "state_detail": _status.get("state_detail"),
        "gateway": dict(_status.get("gateway") or {}),
        "attention": dict(_status.get("attention") or {}),
        "platforms": dict(_status.get("platforms") or {}),
        "updated_at": _status.get("updated_at"),
    }


def _invoke_state_change_hook(payload: dict[str, Any]) -> None:
    try:
        from hermes_cli.plugins import invoke_hook

        invoke_hook("on_agent_state_change", status=dict(payload))
    except Exception:
        pass


def _get_status_path():
    return get_hermes_home() / "status.json"


def _schedule_flush(delay: float) -> None:
    global _flush_timer
    if delay <= 0:
        delay = 0.0
    with _lock:
        if _flush_timer is not None and _flush_timer.is_alive():
            return
        _flush_timer = threading.Timer(delay, _flush_pending_write)
        _flush_timer.daemon = True
        _flush_timer.start()


def _flush_pending_write() -> None:
    global _flush_timer
    try:
        _write_if_due(force=True)
    except Exception:
        pass
    finally:
        with _lock:
            _flush_timer = None


def _write_if_due(force: bool = False) -> None:
    global _last_write_ts, _pending_write
    payload = None
    should_write = False
    now = time.monotonic()
    delay = 0.0
    with _lock:
        _pending_write = True
        if force or (now - _last_write_ts) >= _WRITE_DEBOUNCE_SECONDS:
            _status["updated_at"] = _utc_now_iso()
            payload = _snapshot_locked()
            _last_write_ts = now
            _pending_write = False
            should_write = True
        else:
            delay = _WRITE_DEBOUNCE_SECONDS - (now - _last_write_ts)
    if not should_write:
        try:
            _schedule_flush(delay)
        except Exception:
            pass
    if not should_write or payload is None:
        return
    try:
        atomic_json_write(_get_status_path(), payload, indent=2)
    except Exception:
        return
    _invoke_state_change_hook(payload)


def set_state(state: str, detail: str | None = None, **extra: Any) -> None:
    try:
        with _lock:
            _status["state"] = state
            _status["state_detail"] = detail
            if extra:
                _status.update(extra)
        _write_if_due()
    except Exception:
        pass


def get_status() -> dict[str, Any]:
    try:
        with _lock:
            return _snapshot_locked()
    except Exception:
        return {
            "state": "offline",
            "state_detail": None,
            "gateway": {"running": False, "pid": None},
            "attention": {"needs_attention": False, "reason": None},
            "platforms": {},
            "updated_at": None,
        }


def set_gateway_info(running: bool, pid: int | None = None) -> None:
    try:
        with _lock:
            _status["gateway"] = {"running": bool(running), "pid": pid}
        _write_if_due()
    except Exception:
        pass


def set_platform_info(platforms_dict: dict[str, Any]) -> None:
    try:
        with _lock:
            _status["platforms"] = dict(platforms_dict or {})
        _write_if_due()
    except Exception:
        pass


def set_attention(needs: bool, reason: str | None = None) -> None:
    try:
        with _lock:
            _status["attention"] = {
                "needs_attention": bool(needs),
                "reason": reason,
            }
        _write_if_due()
    except Exception:
        pass
