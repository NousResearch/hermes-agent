"""Durable, machine-scoped UI state owned by Hermes Desktop.

The renderer keeps a localStorage cache for immediate sidebar responsiveness, but
that cache is disposable across Electron updates and browser-profile resets. This
module owns the backend copy used to restore pinned sessions after such a reset.
Pins intentionally span Hermes profiles: the Desktop sidebar can show all
profiles at once and its legacy localStorage key is machine-global.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


PINNED_SESSIONS_RELATIVE_PATH = Path("state") / "desktop-pinned-sessions.json"
_SCHEMA_VERSION = 1


def _normalize_session_ids(values: Any) -> list[str]:
    """Return stable, ordered, non-empty string ids without duplicates."""
    if not isinstance(values, list):
        return []

    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        session_id = value.strip()
        if not session_id or session_id in seen:
            continue
        seen.add(session_id)
        normalized.append(session_id)
    return normalized


def pinned_sessions_path(hermes_home: Path) -> Path:
    return Path(hermes_home) / PINNED_SESSIONS_RELATIVE_PATH


def read_pinned_sessions(hermes_home: Path) -> tuple[bool, list[str]]:
    """Read the persisted list, returning ``(exists, pinned_session_ids)``.

    A malformed or unreadable recovery file behaves as absent so a valid legacy
    renderer cache can repair it. An intentional unpin-all is represented by a
    valid existing file whose list is empty, so it remains authoritative.
    """
    path = pinned_sessions_path(hermes_home)
    if not path.is_file():
        return False, []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, []

    if not isinstance(payload, dict):
        return False, []
    values = payload.get("pinned_session_ids")
    if not isinstance(values, list):
        return False, []
    return True, _normalize_session_ids(values)


def write_pinned_sessions(hermes_home: Path, pinned_session_ids: Iterable[Any]) -> list[str]:
    """Atomically persist normalized pins with owner-only permissions."""
    home = Path(hermes_home)
    path = pinned_sessions_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_session_ids(list(pinned_session_ids))
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "pinned_session_ids": normalized,
    }

    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fchmod = getattr(os, "fchmod", None)
            if fchmod is not None:
                fchmod(handle.fileno(), 0o600)
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        os.chmod(path, 0o600)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return normalized
