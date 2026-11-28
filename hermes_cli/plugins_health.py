"""Plugin health that outlives the process: load failures (deadline skips included) and per-hook
failure counts, readable from another process by ``hermes plugins list`` and ``hermes doctor``.

Before this, a plugin skipped by ``plugins.load_timeout_seconds`` or a hook raising on every turn
was visible only in that process's log (#140621). Each process that loads plugins for a home
writes only its own ``<home>/cache/plugin-health/<pid>.json`` (atomic replace, so no cross-process
lock), and only while it has something to report. Readers merge the records of processes that are
still running; a record whose process is gone is ignored and pruned by the next writer.
"""

from __future__ import annotations

import atexit
import copy
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_FLUSH_INTERVAL_SECS = 30.0  # repeat hook failures rewrite the record at most this often
_ERROR_CHARS = 300
_MEMORY_NAMESPACES = ("_hermes_user_memory.", "plugins.memory.")


def health_dir(home: Path) -> Path:
    return Path(home) / "cache" / "plugin-health"


def _clip(text: Any) -> str:
    text = str(text)
    return text if len(text) <= _ERROR_CHARS else text[: _ERROR_CHARS - 1] + "…"


def _start_time(pid: int) -> Optional[int]:
    from gateway.status import get_process_start_time
    return get_process_start_time(pid)


def _record_alive(record: dict) -> bool:
    """Same process incarnation as the one that wrote ``record`` (PID-reuse guarded by start time)."""
    from gateway.status import _pid_exists, start_time_fingerprints_match
    pid = record.get("pid")
    if not isinstance(pid, int) or not _pid_exists(pid):
        return False
    recorded, current = record.get("start_time"), _start_time(pid)
    if not isinstance(recorded, int) or current is None:
        return True
    return start_time_fingerprints_match(recorded, current)


class PluginHealth:
    """One process's plugin health for one Hermes home."""

    def __init__(self, home: Path):
        self.home = Path(home)
        self.path = health_dir(self.home) / f"{os.getpid()}.json"
        self._lock = threading.Lock()
        self.load_failures: dict[str, dict] = {}
        self.hook_failures: dict[str, dict[str, dict]] = {}
        self._last_flush = 0.0
        self._timer: Optional[threading.Timer] = None
        self._written = False

    def record_load(self, plugin_key: str, error: Optional[str]) -> None:
        """``error=None`` clears a previous failure (a later load of the same plugin succeeded)."""
        with self._lock:
            if error is None:
                changed = self.load_failures.pop(plugin_key, None) is not None
            else:
                error = _clip(error)
                changed = self.load_failures.get(plugin_key, {}).get("error") != error
                self.load_failures[plugin_key] = {"error": error, "at": time.time()}
        if changed:
            self.flush()

    def record_hook_failure(self, plugin_key: str, hook_name: str, error: str, *, timeout: bool = False) -> None:
        with self._lock:
            hooks = self.hook_failures.setdefault(plugin_key, {})
            first = hook_name not in hooks
            entry = hooks.setdefault(hook_name, {"failures": 0, "timeouts": 0})
            entry["timeouts" if timeout else "failures"] += 1
            entry.update(last_error=_clip(error), last_at=time.time())
            due = first or time.monotonic() - self._last_flush >= _FLUSH_INTERVAL_SECS
            if not due and self._timer is None:
                # Trailing write so the count a reader sees catches up once failures stop.
                self._timer = threading.Timer(_FLUSH_INTERVAL_SECS, self.flush)
                self._timer.daemon = True
                self._timer.start()
        if due:
            self.flush()

    def snapshot(self) -> dict:
        with self._lock:
            return {"pid": os.getpid(), "start_time": _start_time(os.getpid()), "updated_at": time.time(),
                    "load_failures": copy.deepcopy(self.load_failures),
                    "hook_failures": copy.deepcopy(self.hook_failures)}

    def flush(self) -> None:
        """Write this process's record, or remove it once nothing is left to report. A read-only or
        full disk must never break plugin loading or a hook dispatch, so I/O errors only log."""
        from utils import atomic_json_write
        record = self.snapshot()
        with self._lock:
            self._last_flush = time.monotonic()
            self._timer = None
        try:
            if record["load_failures"] or record["hook_failures"]:
                if not self._written:
                    _prune_dead_records(self.home)
                    atexit.register(self.discard)
                self.path.parent.mkdir(parents=True, exist_ok=True)
                atomic_json_write(self.path, record)
                self._written = True
            elif self._written:
                self.path.unlink(missing_ok=True)
        except OSError as exc:
            logger.debug("Could not write plugin health record %s: %s", self.path, exc)

    def discard(self) -> None:
        """Process exit: a stopped process's failures are no longer current."""
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            logger.debug("Could not remove plugin health record %s", self.path)


_HEALTH: dict[str, PluginHealth] = {}
_HEALTH_LOCK = threading.Lock()


def health_for(home: Path) -> PluginHealth:
    key = str(Path(home))
    with _HEALTH_LOCK:
        health = _HEALTH.get(key)
        if health is None:
            health = _HEALTH[key] = PluginHealth(Path(home))
        return health


def callback_owner(plugins: dict, callback: Callable) -> Optional[str]:
    """Plugin key whose module defined ``callback``; memory-loader copies map to the provider name.
    ``None`` for callbacks no plugin owns (config shell hooks, plugin-host proxies)."""
    module = getattr(callback, "__module__", None) or ""
    for key, loaded in plugins.items():
        name = getattr(getattr(loaded, "module", None), "__name__", None)
        if name and (module == name or module.startswith(name + ".")):
            return key
    for namespace in _MEMORY_NAMESPACES:
        if module.startswith(namespace):
            return module[len(namespace):].split(".", 1)[0].split("__source_", 1)[0] or None
    return None


def _prune_dead_records(home: Path) -> None:
    for path in health_dir(home).glob("*.json"):
        record = _read_record(path)
        if record is None or not _record_alive(record):
            path.unlink(missing_ok=True)


def _read_record(path: Path) -> Optional[dict]:
    try:
        record = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def read_plugin_health(home: Path) -> dict[str, dict]:
    """Merged health of every running process for ``home``:
    ``{plugin_key: {"load_error": str | None, "pids": [...], "hooks": {hook: {failures, timeouts, last_error}}}}``.
    """
    merged: dict[str, dict] = {}
    for path in sorted(health_dir(home).glob("*.json")):
        record = _read_record(path)
        if record is None or not _record_alive(record):
            continue
        pid = record.get("pid")
        for key, failure in (record.get("load_failures") or {}).items():
            entry = merged.setdefault(key, {"load_error": None, "pids": [], "hooks": {}})
            entry["load_error"] = (failure or {}).get("error") or entry["load_error"]
            entry["pids"].append(pid)
        for key, hooks in (record.get("hook_failures") or {}).items():
            entry = merged.setdefault(key, {"load_error": None, "pids": [], "hooks": {}})
            if pid not in entry["pids"]:
                entry["pids"].append(pid)
            for hook_name, counts in (hooks or {}).items():
                slot = entry["hooks"].setdefault(hook_name, {"failures": 0, "timeouts": 0, "last_error": ""})
                slot["failures"] += int(counts.get("failures") or 0)
                slot["timeouts"] += int(counts.get("timeouts") or 0)
                slot["last_error"] = counts.get("last_error") or slot["last_error"]
    return merged


def describe(health: dict) -> list[str]:
    """Plain-text problem lines for one plugin's merged health (no markup)."""
    lines = []
    if health.get("load_error"):
        lines.append(f"load failed: {health['load_error']}")
    for hook_name, counts in sorted((health.get("hooks") or {}).items()):
        parts = [f"{n} {label}" for n, label in ((counts["failures"], "failed"), (counts["timeouts"], "timed out")) if n]
        last = f" (last: {counts['last_error']})" if counts.get("last_error") else ""
        lines.append(f"hook {hook_name}: {', '.join(parts)}{last}")
    return lines
