"""
sessions.json indexer for standalone CLI sessions.

When the CLI runs through the gateway, ``gateway.session.SessionStore`` is
responsible for keeping ``~/.hermes/sessions/sessions.json`` in sync with the
SQLite session DB.  When the CLI runs **standalone** -- e.g. via
``hermes.exe`` on Windows / PowerShell, or just ``hermes`` on macOS/Linux
without a gateway process attached -- nothing writes the CLI session into
that JSON index.  Downstream consumers that scan sessions.json (status,
mcp_serve, channel_directory, mirror) then can't see CLI conversations.

This module provides a tiny, dependency-light upsert that the CLI calls on
session start, update and exit.  It writes a SessionEntry-shaped record
keyed by a synthetic ``cli:<session_id>`` key so it can't collide with
gateway-managed keys.

Errors are intentionally swallowed -- failing to index must never break a
live CLI session.

See issue #29073.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_cli.config import get_hermes_home
from utils import atomic_replace

logger = logging.getLogger(__name__)

# Cross-thread guard for the read/modify/write cycle.  The on-disk write is
# atomic via tempfile + ``atomic_replace``, but two threads in the same
# process (e.g. the cleanup path racing the agent thread's persist hook)
# could still clobber each other's updates without this lock.
_INDEX_LOCK = threading.Lock()

_CLI_KEY_PREFIX = "cli:"


def _sessions_index_path() -> Path:
    return get_hermes_home() / "sessions" / "sessions.json"


def _now_iso() -> str:
    # Match gateway/session.py SessionEntry.to_dict() which uses naive
    # local-time isoformat via ``_now()`` -> datetime.now().  Aligning the
    # format keeps from_dict() round-trips happy.
    return datetime.now().isoformat()


def _load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("sessions.json read failed (%s); starting fresh dict", e)
    return {}


def _atomic_write(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp", prefix=".sessions_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                # fsync can fail on some Windows filesystems / WSL mounts;
                # the rename below is still atomic enough for our purposes.
                pass
        atomic_replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _build_entry(
    session_id: str,
    *,
    created_at: Optional[str],
    display_name: Optional[str],
    source: str,
) -> Dict[str, Any]:
    """Build a minimal SessionEntry-shaped dict for a CLI session.

    Uses ``platform: "local"`` because the ``Platform`` enum has a LOCAL
    member -- gateway/session.py's ``SessionEntry.from_dict()`` will accept
    it without raising.  ``chat_type`` stays "dm" (the default).
    """
    now = _now_iso()
    return {
        "session_key": f"{_CLI_KEY_PREFIX}{session_id}",
        "session_id": session_id,
        "created_at": created_at or now,
        "updated_at": now,
        "display_name": display_name,
        "platform": "local",
        "chat_type": "dm",
        "origin": {
            "platform": "local",
            "chat_id": session_id,
            "chat_type": "dm",
            "source_label": source,
        },
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "total_tokens": 0,
        "last_prompt_tokens": 0,
        "estimated_cost_usd": 0.0,
        "cost_status": "unknown",
        "expiry_finalized": False,
        "suspended": False,
        "resume_pending": False,
    }


def index_cli_session(
    session_id: Optional[str],
    *,
    display_name: Optional[str] = None,
    source: str = "cli",
) -> bool:
    """Upsert a CLI session entry into ``sessions.json``.

    Idempotent: re-indexing the same session_id refreshes ``updated_at`` but
    preserves the original ``created_at``.  Never raises -- returns False on
    any failure so the live CLI session keeps running.

    Returns True when the entry was written (or refreshed) successfully.
    """
    if not session_id or not isinstance(session_id, str):
        return False

    path = _sessions_index_path()
    key = f"{_CLI_KEY_PREFIX}{session_id}"

    try:
        with _INDEX_LOCK:
            data = _load(path)
            existing = data.get(key) if isinstance(data.get(key), dict) else None
            created_at = existing.get("created_at") if existing else None
            data[key] = _build_entry(
                session_id,
                created_at=created_at,
                display_name=display_name or (existing.get("display_name") if existing else None),
                source=source,
            )
            _atomic_write(path, data)
        return True
    except Exception as e:  # noqa: BLE001 -- indexing must never crash CLI
        logger.debug("Failed to index CLI session %s in sessions.json: %s", session_id, e)
        return False
