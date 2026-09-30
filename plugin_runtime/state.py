"""Plugin-owned durable state runtime."""

from __future__ import annotations

import json
import os
import re
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

from hermes_constants import get_hermes_home
from plugin_runtime.manifest import _portable_skill_namespace

_PLUGIN_STATE_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_PLUGIN_STATE_QUOTA_BYTES = 10 * 1024 * 1024
_PLUGIN_STATE_LOCKS: Dict[str, threading.RLock] = {}
_PLUGIN_STATE_LOCKS_GUARD = threading.Lock()


def _plugin_data_namespace(plugin_id: str, skill_namespace: str) -> str:
    """Return one Windows-safe directory component for plugin-owned data."""
    candidate = skill_namespace or plugin_id
    portable = skill_namespace and candidate.startswith("agent-plugin-")
    if portable and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,191}", candidate):
        return candidate
    return _portable_skill_namespace(candidate)


@contextmanager
def _locked_plugin_state(path: Path):
    """Serialize state read-modify-write across threads/processes."""
    lock_path = path.with_name(f".{path.name}.lock")
    with _PLUGIN_STATE_LOCKS_GUARD:
        thread_lock = _PLUGIN_STATE_LOCKS.setdefault(
            str(lock_path.resolve(strict=False)), threading.RLock()
        )
    with thread_lock:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "a+b") as handle:
            if os.name == "nt":  # pragma: no cover - exercised on Windows CI
                import msvcrt

                if handle.seek(0, os.SEEK_END) == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == "nt":  # pragma: no cover - exercised on Windows CI
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class PluginState:
    """Atomic, quota-bounded JSON key/value state owned by one plugin."""

    def __init__(self, plugin_id: str, skill_namespace: str = "") -> None:
        self._data_namespace = _plugin_data_namespace(plugin_id, skill_namespace)

    @property
    def data_dir(self) -> Path:
        """Profile-scoped directory matching portable plugins' PLUGIN_DATA."""
        return get_hermes_home() / "plugin-data" / self._data_namespace

    @property
    def path(self) -> Path:
        return self.data_dir / "state.json"

    @property
    def quota_bytes(self) -> int:
        return _PLUGIN_STATE_QUOTA_BYTES

    @staticmethod
    def _validate_key(key: str) -> None:
        if not isinstance(key, str) or not _PLUGIN_STATE_KEY_RE.fullmatch(key) or ".." in key:
            raise ValueError(
                "Plugin state keys must be 1-128 characters using letters, "
                "numbers, '_', '-', '.', or ':' (without '..')"
            )

    def _read_unlocked(self) -> dict[str, Any]:
        try:
            with open(self.path, encoding="utf-8-sig") as handle:
                data = json.load(handle)
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Cannot parse plugin state {self.path}: {exc}") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"Cannot parse plugin state {self.path}: root must be an object")
        return data

    def get(self, key: str, default: Any = None) -> Any:
        """Read a JSON value, returning *default* when the key is absent."""
        self._validate_key(key)
        with _locked_plugin_state(self.path):
            return self._read_unlocked().get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Atomically set one JSON value without dropping concurrent updates."""
        self._validate_key(key)
        with _locked_plugin_state(self.path):
            data = self._read_unlocked()
            data[key] = value
            try:
                encoded = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Plugin state value for {key!r} is not JSON-serializable"
                ) from exc
            if len(encoded) > self.quota_bytes:
                raise ValueError(
                    f"Plugin state quota exceeded: {len(encoded)} bytes is greater "
                    f"than the {self.quota_bytes}-byte per-plugin quota"
                )
            from utils import atomic_json_write

            atomic_json_write(self.path, data, mode=0o600)
