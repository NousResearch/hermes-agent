"""Machine-level Desktop/Webapp UI preferences (#145213).

One JSON document per machine, beside the ROOT Hermes home (``get_default_hermes_root()``, never a
profile's ``HERMES_HOME``): native Desktop writes it through its preload capability
(``apps/desktop/electron/ui-preferences.ts``) and every browser that opens this machine's Webapp
writes it through ``/api/ui-preferences``. It is deliberately neither per profile nor per gateway
connection; a key that is scoped (``….profile.<name>``) carries its scope in its own name.

Both writers share the on-disk contract, so keep the two in lockstep:

* document ``{"version": 1, "revision": <int>, "values": {<key>: <string>}}``;
* a write merges per key under the ``<store>.lock`` directory lock (``mkdir`` is atomic on every
  OS and in Node), bumps ``revision`` and lands by temp file + rename; a ``None`` value deletes;
* a ``seed`` write (the one-time localStorage import) applies only when no document exists.

Values are opaque strings the renderer already kept in localStorage. Only keys under the Desktop
prefixes are accepted; the renderer's allowlist decides which of those are preferences and keeps
credentials, grants and per-window state out.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping, Optional

from utils import atomic_json_write, read_json_or_empty

STORE_NAME = "ui-preferences.json"
KEY_PREFIXES = ("hermes.desktop.", "hermes-desktop-")
MAX_KEY_CHARS = 256
MAX_VALUE_CHARS = 256 * 1024
MAX_KEYS = 512
MAX_TOTAL_CHARS = 2 * 1024 * 1024
_LOCK_WAIT_S = 3.0
_LOCK_STALE_S = 10.0
_LOCK_POLL_S = 0.02
_thread_lock = threading.Lock()


class PreferencesError(ValueError):
    """A write the store refuses (bad key, value or size); the route answers 400."""


class PreferencesBusy(RuntimeError):
    """Another writer held the store lock past the wait; the route answers 503 and the client
    keeps the change pending for its next attempt."""


def store_path(root: Optional[Path] = None) -> Path:
    """The machine's preferences file. The root home is the same for every profile one process
    serves, so this read is profile-independent by design (not a launch-profile leak)."""
    if root is None:
        from hermes_constants import get_default_hermes_root

        root = get_default_hermes_root()
    return Path(root) / STORE_NAME


def _sanitize(values: object) -> dict[str, str]:
    if not isinstance(values, dict):
        return {}
    return {k: v for k, v in values.items() if isinstance(k, str) and isinstance(v, str)}


def _snapshot(path: Path) -> dict:
    exists = path.is_file()
    doc = read_json_or_empty(path) if exists else {}
    revision = doc.get("revision")
    return {
        "exists": exists,
        "revision": revision if isinstance(revision, int) else 0,
        "values": _sanitize(doc.get("values")),
    }


def read_preferences(path: Optional[Path] = None) -> dict:
    """``{"exists", "revision", "values"}`` for the store at *path* (the machine's by default)."""
    return _snapshot(path or store_path())


def _validate(changes: Mapping[str, Optional[str]]) -> None:
    for key, value in changes.items():
        if not isinstance(key, str) or not key.startswith(KEY_PREFIXES) or len(key) > MAX_KEY_CHARS:
            raise PreferencesError(f"unsupported preference key: {key!r}")
        if value is not None and (not isinstance(value, str) or len(value) > MAX_VALUE_CHARS):
            raise PreferencesError(f"unsupported value for {key}")


@contextmanager
def _store_lock(path: Path) -> Iterator[None]:
    """Cross-process lock shared with the Electron writer: the ``<store>.lock`` directory."""
    lock_dir = path.with_name(path.name + ".lock")
    deadline = time.monotonic() + _LOCK_WAIT_S
    with _thread_lock:
        while True:
            try:
                os.mkdir(lock_dir)
                break
            except FileExistsError:
                try:
                    stale = time.time() - lock_dir.stat().st_mtime > _LOCK_STALE_S
                except FileNotFoundError:
                    continue
                if stale:
                    # A writer that died holding the lock must not wedge every later write; the
                    # data file itself is only ever replaced whole, so breaking it loses nothing.
                    try:
                        os.rmdir(lock_dir)
                    except OSError:
                        pass
                    continue
                if time.monotonic() > deadline:
                    raise PreferencesBusy("preferences store is busy")
                time.sleep(_LOCK_POLL_S)
        try:
            yield
        finally:
            try:
                os.rmdir(lock_dir)
            except OSError:
                pass


def write_preferences(changes: Mapping[str, Optional[str]], *, seed: bool = False,
                      path: Optional[Path] = None) -> dict:
    """Merge *changes* per key (``None`` deletes) and return the resulting snapshot.

    ``seed=True`` is the one-time import of a window's existing localStorage: it writes only when
    the store does not exist yet and otherwise returns the existing store untouched.
    """
    _validate(changes)
    path = path or store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _store_lock(path):
        current = _snapshot(path)
        if seed and current["exists"]:
            return current
        values = dict(current["values"])
        for key, value in changes.items():
            if value is None:
                values.pop(key, None)
            else:
                values[key] = value
        if len(values) > MAX_KEYS or sum(len(k) + len(v) for k, v in values.items()) > MAX_TOTAL_CHARS:
            raise PreferencesError("preferences store is full")
        if current["exists"] and values == current["values"]:
            return current
        revision = current["revision"] + 1
        atomic_json_write(path, {"version": 1, "revision": revision, "values": values}, mode=0o600)
    return {"exists": True, "revision": revision, "values": values}
