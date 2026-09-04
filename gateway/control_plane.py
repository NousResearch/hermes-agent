"""Gateway control-plane helpers: origin marker, lock owner, stall, restart handoff.

These functions are deliberately pure/DI-friendly so tests can use temp
homes and process doubles. They must never restart a live gateway.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, MutableMapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

GATEWAY_ORIGIN_ENV = "_HERMES_GATEWAY_ORIGIN"
GATEWAY_PROCESS_ENV = "_HERMES_GATEWAY"
KANBAN_TASK_ENV = "HERMES_KANBAN_TASK"

RESTART_HANDOFF_FILENAME = ".restart_handoff.json"
DISPATCHER_LOCK_OWNER_SUFFIX = ".owner.json"

_SAFE_OWNER_KEYS = ("profile", "pid", "acquired_at")


def is_gateway_originated(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return str(env.get(GATEWAY_ORIGIN_ENV, "")).strip() == "1"


def stamp_gateway_origin(environ: MutableMapping[str, str]) -> MutableMapping[str, str]:
    environ[GATEWAY_ORIGIN_ENV] = "1"
    return environ


def propagate_gateway_origin(
    parent_env: Mapping[str, str] | None,
    child_env: MutableMapping[str, str],
) -> MutableMapping[str, str]:
    """Copy the origin marker into a child even if ``_HERMES_GATEWAY`` is stripped.

    A parent that *is* a gateway (or already originated) marks the child so
    later watcher/serve env scrubs cannot drop the self-restart guard.
    """
    parent = parent_env or {}
    if (
        str(parent.get(GATEWAY_ORIGIN_ENV, "")).strip() == "1"
        or str(parent.get(GATEWAY_PROCESS_ENV, "")).strip() == "1"
    ):
        child_env[GATEWAY_ORIGIN_ENV] = "1"
    return child_env


def scrub_gateway_markers_for_restart_watcher(
    environ: MutableMapping[str, str],
) -> MutableMapping[str, str]:
    """Restart watchers must be able to run ``hermes gateway restart``."""
    environ.pop(GATEWAY_PROCESS_ENV, None)
    environ.pop(GATEWAY_ORIGIN_ENV, None)
    return environ


def should_refuse_inline_gateway_lifecycle(
    environ: Mapping[str, str] | None = None,
    *,
    supervised: bool | None = None,
) -> bool:
    """True when this process must not stop/restart/uninstall its serving gateway.

    Order:
    1. Explicit origin marker (survives ``_HERMES_GATEWAY`` scrub).
    2. Dispatcher-spawned kanban worker (``HERMES_KANBAN_TASK``).
    3. Supervised gateway PID owner (existing in-process guard).
    """
    env = os.environ if environ is None else environ
    if is_gateway_originated(env):
        return True
    if str(env.get(KANBAN_TASK_ENV) or "").strip():
        return True
    if supervised is None:
        try:
            from tools.process_registry import _is_supervised_gateway_process

            supervised = bool(_is_supervised_gateway_process())
        except Exception:
            supervised = False
    return bool(supervised)


def current_profile_name(environ: Mapping[str, str] | None = None) -> str:
    """Best-effort active profile name. Never hard-codes a deployment roster."""
    env = os.environ if environ is None else environ
    named = str(env.get("HERMES_PROFILE") or "").strip()
    if named:
        return named
    try:
        from hermes_cli.profiles import get_active_profile_name

        resolved = get_active_profile_name()
        if resolved:
            return str(resolved)
    except Exception:
        pass
    return "default"


def dispatcher_lock_owner_path(lock_path: str | os.PathLike[str]) -> Path:
    path = Path(lock_path)
    return path.with_name(path.name + DISPATCHER_LOCK_OWNER_SUFFIX)


def write_dispatcher_lock_owner(
    lock_path: str | os.PathLike[str],
    *,
    profile: str,
    pid: int,
    acquired_at: float | None = None,
) -> Path:
    """Persist safe owner diagnostics next to the advisory lock file."""
    acquired = acquired_at if acquired_at is not None else time.time()
    payload = {
        "profile": str(profile or "default"),
        "pid": int(pid),
        "acquired_at": datetime.fromtimestamp(float(acquired), tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }
    owner_path = dispatcher_lock_owner_path(lock_path)
    owner_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = owner_path.with_suffix(owner_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=None), encoding="utf-8")
    tmp.replace(owner_path)
    return owner_path


def read_dispatcher_lock_owner(lock_path: str | os.PathLike[str]) -> Optional[dict[str, Any]]:
    owner_path = dispatcher_lock_owner_path(lock_path)
    try:
        data = json.loads(owner_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    if not isinstance(data, dict):
        return None
    return {key: data.get(key) for key in _SAFE_OWNER_KEYS}


def clear_dispatcher_lock_owner(lock_path: str | os.PathLike[str]) -> None:
    try:
        dispatcher_lock_owner_path(lock_path).unlink(missing_ok=True)
    except OSError:
        pass


def format_lock_owner(owner: Mapping[str, Any] | None) -> str:
    if not owner:
        return "unknown"
    profile = owner.get("profile") or "unknown"
    pid = owner.get("pid")
    acquired = owner.get("acquired_at") or "unknown"
    pid_txt = str(pid) if pid is not None else "unknown"
    return f"profile={profile} pid={pid_txt} acquired_at={acquired}"


class StalledDispatchTracker:
    """READY>0 and RUNNING==0 beyond one dispatch interval → warn, do not spawn."""

    def __init__(self) -> None:
        self.streak = 0

    def observe(
        self,
        *,
        ready: int,
        running: int,
        board: str,
        owner: Mapping[str, Any] | None = None,
    ) -> Optional[str]:
        if int(ready) > 0 and int(running) == 0:
            self.streak += 1
        else:
            self.streak = 0
            return None
        if self.streak <= 1:
            return None
        return (
            "kanban control-plane: persistent READY "
            f"ready={int(ready)} running=0 on board {board} "
            f"for {self.streak} dispatch intervals (beyond one interval). "
            f"lock_owner={format_lock_owner(owner)}. "
            "Not auto-running unsafe work."
        )


def build_restart_handoff(
    *,
    session_key: str | None = None,
    platform: str | None = None,
    chat_id: str | None = None,
    thread_id: str | None = None,
    message_id: str | None = None,
    acknowledged_at: float | None = None,
) -> dict[str, Any]:
    ts = acknowledged_at if acknowledged_at is not None else time.time()
    return {
        "session_key": session_key,
        "platform": platform,
        "chat_id": chat_id,
        "thread_id": thread_id,
        "message_id": message_id,
        "acknowledged_at": datetime.fromtimestamp(float(ts), tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "acknowledged": True,
        "outbound_flushed": False,
        "gateway_online": False,
        "notify_delivered": False,
        "sessions_restored": False,
        "launcher_exited": False,
    }


def persist_restart_handoff(path: str | os.PathLike[str], handoff: Mapping[str, Any]) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(handoff), indent=None), encoding="utf-8")
    tmp.replace(dest)
    return dest


def load_restart_handoff(path: str | os.PathLike[str]) -> Optional[dict[str, Any]]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    return data if isinstance(data, dict) else None


def mark_restart_handoff(path: str | os.PathLike[str], **fields: Any) -> Optional[dict[str, Any]]:
    handoff = load_restart_handoff(path)
    if handoff is None:
        return None
    handoff.update(fields)
    persist_restart_handoff(path, handoff)
    return handoff


def claim_restart_recovery(
    handoff: Mapping[str, Any] | None,
    *,
    gateway_online: bool = False,
    notify_delivered: bool = False,
    sessions_restored: bool = False,
    launcher_exited: bool = False,
) -> bool:
    """Recovery is real only after the new gateway is up and has reported.

    Launcher exit alone is never sufficient.
    """
    if not handoff or not handoff.get("acknowledged"):
        return False
    if launcher_exited and not gateway_online and not notify_delivered and not sessions_restored:
        return False
    if not gateway_online:
        return False
    return bool(notify_delivered or sessions_restored)
