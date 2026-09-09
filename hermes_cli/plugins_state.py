"""Plugin-owned durable state and namespaced settings helpers (split out of hermes_cli.plugins)."""

from __future__ import annotations

import json
import os
import re
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Mapping

from hermes_constants import get_hermes_home
from hermes_cli.plugins_manifest import _portable_skill_namespace

_PLUGIN_SETTING_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_PLUGIN_SETTING_RESERVED_ROOTS = frozenset({"model", "plugins", "security", "settings"})
_PLUGIN_STATE_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_PLUGIN_STATE_QUOTA_BYTES = 10 * 1024 * 1024
_PLUGIN_STATE_LOCKS: Dict[str, threading.RLock] = {}
_PLUGIN_STATE_LOCKS_GUARD = threading.Lock()


def _plugin_relative_segments(key: str) -> tuple[str, ...]:
    """Validate/split a plugin-relative settings key; global paths, traversal, and core roots are rejected
    before any config read.

    The public API accepts only relative keys (``endpoint`` or ``retry.policy``). See #64227.
    """
    if not isinstance(key, str):
        raise ValueError("Expected a plugin-relative config key string")
    segments = tuple(key.split("."))
    invalid = not key or "/" in key or "\\" in key or segments[0].lower() in _PLUGIN_SETTING_RESERVED_ROOTS
    if invalid or not all(_PLUGIN_SETTING_SEGMENT_RE.fullmatch(segment) for segment in segments):
        raise ValueError(
            "Expected a plugin-relative config key such as 'endpoint' or "
            "'retry.policy'; global, cross-plugin, and traversal paths are forbidden"
        )
    return segments


def _nested_plugin_value(root: object, segments: tuple[str, ...], default: Any) -> Any:
    """Walk ``segments`` through nested mappings; ``default`` on the first miss."""
    current = root
    for segment in segments:
        if not isinstance(current, Mapping) or segment not in current:
            return default
        current = current[segment]
    return current


def _nested_plugin_mapping(segments: tuple[str, ...], value: Any) -> dict[str, Any]:
    """Wrap ``value`` in nested single-key dicts, outermost first."""
    nested: Any = value
    for segment in reversed(segments):
        nested = {segment: nested}
    return nested


def _plugin_settings_entry(config: object, plugin_id: str) -> Mapping[str, Any] | None:
    """``plugins.entries.<plugin_id>`` as a mapping, else ``None``."""
    entry = _nested_plugin_value(config, ("plugins", "entries", plugin_id), None)
    return entry if isinstance(entry, Mapping) else None


def _plugin_data_namespace(plugin_id: str, skill_namespace: str) -> str:
    """Return one Windows-safe directory component for plugin-owned data. Portable Agent Plugins already
    receive this exact PLUGIN_DATA path; otherwise the fixed prefix avoids Windows reserved device names and
    the digest prevents fold collisions."""
    candidate = skill_namespace or plugin_id
    portable = skill_namespace and candidate.startswith("agent-plugin-")
    if portable and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,191}", candidate):
        return candidate
    return _portable_skill_namespace(candidate)


@contextmanager
def _locked_plugin_state(path: Path):
    """Serialize state read-modify-write across threads/processes (fcntl / msvcrt). The lock lives in a
    sibling file because atomic replacement changes the target's inode."""
    lock_path = path.with_name(f".{path.name}.lock")
    with _PLUGIN_STATE_LOCKS_GUARD:
        thread_lock = _PLUGIN_STATE_LOCKS.setdefault(str(lock_path.resolve(strict=False)), threading.RLock())
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
            with open(self.path, encoding="utf-8") as handle:
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
                raise ValueError(f"Plugin state value for {key!r} is not JSON-serializable") from exc
            if len(encoded) > self.quota_bytes:
                raise ValueError(
                    f"Plugin state quota exceeded: {len(encoded)} bytes is greater "
                    f"than the {self.quota_bytes}-byte per-plugin quota"
                )
            from utils import atomic_json_write
            atomic_json_write(self.path, data, mode=0o600)
