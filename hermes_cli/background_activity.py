"""Process-backed leases for user-visible background AI work.

Each worker owns one small JSON file under its profile's cache.  Per-worker files
avoid cross-process lost updates; readers discard leases whose PID identity is no
longer alive, so crashes, cancellation and gateway restarts cannot leave a busy
indicator stuck on.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

_TITLE_LIMIT = 72
_SPACE_RE = re.compile(r"\s+")
_PATH_RE = re.compile(r"(?:[A-Za-z]:[\\/]|/)(?:[^\s/\\]+[/\\])+[^\s]*")
_SECRET_RE = re.compile(
    r"(?i)\b(?:sk|ghp|github_pat|xox[baprs]|AIza|AKIA)[-_A-Za-z0-9]{6,}\b"
)
_ACTIVE: dict[str, "ActivityLease"] = {}
_LOCK = threading.RLock()


def sanitize_title(value: Any, *, max_length: int = _TITLE_LIMIT) -> str:
    """Return compact single-line display text with path/token-shaped data removed."""
    text = _SPACE_RE.sub(" ", str(value or "")).strip()
    text = _PATH_RE.sub("[path]", text)
    text = _SECRET_RE.sub("[secret]", text)
    text = "".join(ch for ch in text if ch.isprintable())
    if len(text) > max_length:
        text = text[: max(1, max_length - 1)].rstrip() + "…"
    return text or "Background worker"


def format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _lease_dir(home: Path | None = None) -> Path:
    return Path(home or get_hermes_home()) / "cache" / "background-work"


def _process_start_time(pid: int) -> float | None:
    try:
        import psutil

        return float(psutil.Process(pid).create_time())
    except Exception:
        return None


def _same_live_process(pid: int, recorded: Any) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ValueError, TypeError):
        return False
    current = _process_start_time(pid)
    if current is None or recorded in (None, ""):
        return True
    try:
        return abs(current - float(recorded)) < 0.01
    except (TypeError, ValueError):
        return False


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    os.replace(tmp, path)


@dataclass
class ActivityLease:
    key: str
    path: Path
    payload: dict[str, Any]
    closed: bool = False

    @classmethod
    def start(
        cls, *, key: str, title: str, worker: str, profile: str = "default",
        model: str = "", provider: str = "", home: Path | None = None,
    ) -> "ActivityLease":
        safe_key = sanitize_title(key, max_length=128)
        digest = hashlib.sha256(f"{os.getpid()}:{safe_key}".encode()).hexdigest()[:20]
        path = _lease_dir(home) / f"{digest}.json"
        payload = {
            "key": safe_key,
            "title": sanitize_title(title),
            "worker": sanitize_title(worker, max_length=24),
            "profile": sanitize_title(profile or "default", max_length=32),
            "model": sanitize_title(model, max_length=48) if model else "",
            "provider": sanitize_title(provider, max_length=32) if provider else "",
            "pid": os.getpid(),
            "process_started_at": _process_start_time(os.getpid()),
            "started_at": time.time(),
            "state": "running",
        }
        lease = cls(key=safe_key, path=path, payload=payload)
        _atomic_json(path, payload)
        return lease

    def close(self, *, state: str = "completed") -> None:
        if self.closed:
            return
        self.closed = True
        self.payload["state"] = sanitize_title(state, max_length=24)
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            # A reader will remove it as soon as this process exits.
            pass


def list_active_work(home: Path | None = None, *, now: float | None = None) -> list[dict[str, Any]]:
    current = time.time() if now is None else now
    directory = _lease_dir(home)
    if not directory.is_dir():
        return []
    active: list[dict[str, Any]] = []
    for path in directory.glob("*.json"):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            pid = int(item.get("pid"))
            if item.get("state") != "running" or not _same_live_process(pid, item.get("process_started_at")):
                path.unlink(missing_ok=True)
                continue
            started = float(item.get("started_at") or current)
            item["elapsed_seconds"] = max(0, int(current - started))
            item["elapsed"] = format_elapsed(item["elapsed_seconds"])
            active.append(item)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
    active.sort(key=lambda item: (float(item.get("started_at") or 0), str(item.get("key") or "")))
    return active


def _activity_homes(home: Path) -> list[Path]:
    """Homes visible from a default or named-profile home, without creating paths."""
    resolved = home.resolve()
    root = resolved.parent.parent if resolved.parent.name == "profiles" else resolved
    homes = [root]
    profiles = root / "profiles"
    if profiles.is_dir():
        homes.extend(path for path in profiles.iterdir() if path.is_dir())
    if resolved not in homes:
        homes.append(resolved)
    return homes


def list_all_active_work(home: Path | None = None, *, now: float | None = None) -> list[dict[str, Any]]:
    """Aggregate live leases across the install's profile homes."""
    root = Path(home or get_hermes_home())
    items: dict[tuple[Any, Any], dict[str, Any]] = {}
    for candidate in _activity_homes(root):
        for item in list_active_work(candidate, now=now):
            items[(item.get("pid"), item.get("key"))] = item
    return sorted(items.values(), key=lambda item: (float(item.get("started_at") or 0), str(item.get("key") or "")))


def _profile_name() -> str:
    return (os.environ.get("HERMES_PROFILE") or "default").strip() or "default"


def _open(key: str, *, title: str, worker: str, model: str = "", provider: str = "") -> None:
    with _LOCK:
        previous = _ACTIVE.pop(key, None)
        if previous is not None:
            previous.close(state="replaced")
        _ACTIVE[key] = ActivityLease.start(
            key=key, title=title, worker=worker, profile=_profile_name(),
            model=model, provider=provider,
        )


def _close(key: str, state: str) -> None:
    with _LOCK:
        lease = _ACTIVE.pop(key, None)
    if lease is not None:
        lease.close(state=state)


def observe_lifecycle(hook_name: str, **kwargs: Any) -> None:
    """Consume existing lifecycle events without exposing prompts or conversation text."""
    from agent.delegation_context import owned_kanban_task

    task_id = owned_kanban_task()
    if hook_name == "on_session_start" and task_id:
        session_id = str(kwargs.get("session_id") or task_id)
        _open(
            f"kanban:{session_id}",
            title=os.environ.get("HERMES_BACKGROUND_WORK_TITLE") or "Kanban worker",
            worker="kanban",
            model=str(kwargs.get("model") or os.environ.get("HERMES_BACKGROUND_WORK_MODEL") or ""),
            provider=str(
                kwargs.get("provider") or os.environ.get("HERMES_BACKGROUND_WORK_PROVIDER") or ""
            ),
        )
    elif hook_name in {"on_session_end", "on_session_finalize", "on_session_reset"} and task_id:
        session_id = str(kwargs.get("session_id") or "")
        if session_id:
            state = "interrupted" if kwargs.get("interrupted") else str(kwargs.get("reason") or "completed")
            _close(f"kanban:{session_id}", state)
    elif hook_name == "subagent_start":
        child_id = str(kwargs.get("child_session_id") or kwargs.get("child_subagent_id") or "")
        if child_id:
            _open(
                f"delegate:{child_id}", title="Delegated worker", worker="delegate",
                model=str(kwargs.get("model") or ""),
                provider=str(kwargs.get("provider") or ""),
            )
    elif hook_name == "subagent_stop":
        child_id = str(kwargs.get("child_session_id") or kwargs.get("child_subagent_id") or "")
        if child_id:
            _close(f"delegate:{child_id}", str(kwargs.get("child_status") or "completed"))
