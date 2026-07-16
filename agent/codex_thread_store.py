"""Persistent Hermes session -> Codex thread mapping."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_home


_LOCK = threading.Lock()
_STORE_NAME = "codex_thread_mappings.json"


def _store_path() -> Path:
    return Path(get_hermes_home()) / _STORE_NAME


def _read_all() -> dict[str, dict[str, str]]:
    try:
        data = json.loads(_store_path().read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        str(key): value
        for key, value in data.items()
        if isinstance(value, dict) and isinstance(value.get("codex_thread_id"), str)
    }


def get_codex_thread_id(hermes_session_id: str) -> Optional[str]:
    if not hermes_session_id:
        return None
    with _LOCK:
        record = _read_all().get(hermes_session_id)
    return record.get("codex_thread_id") if record else None


def save_codex_thread_id(
    hermes_session_id: str,
    codex_thread_id: str,
    *,
    cwd: str = "",
) -> None:
    if not hermes_session_id or not codex_thread_id:
        return
    with _LOCK:
        path = _store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = _read_all()
        data[hermes_session_id] = {
            "codex_thread_id": codex_thread_id,
            "cwd": cwd,
        }
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{_STORE_NAME}.", dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
