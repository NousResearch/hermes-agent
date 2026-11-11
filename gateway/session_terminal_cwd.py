"""A chat's terminal working directory survives a gateway restart (#41128).

The terminal tool records where each command finished (``tools.terminal_tool.record_session_cwd``,
keyed by the chat's session key) in a process-local dict, so a ``cd`` made in one turn holds for the
next. A restart empties that dict and the chat's next command runs in ``terminal.cwd`` again. The
turn now writes the recorded directory onto the chat's routing entry when it changed, and a turn
that finds no record (the first after a restart) seeds it back from there.

Local backend only: a host path can be checked before it is trusted. A remote or container
directory cannot be vouched for from here, and every session-env command starts with
``cd -- <cwd> || exit 126``, so a stale one would break the chat's commands instead of falling back.
A ``/new`` starts a fresh entry, so a new conversation after a restart starts in ``terminal.cwd``.
"""
from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

_METADATA_KEY = "terminal_cwd"


def _local_backend(policy) -> bool:
    """The backend this turn's terminal runs on: a local session's frozen policy, else (a messaging
    turn, which has none) the profile's live terminal config the tool itself reads."""
    if policy is not None:
        backend = json.loads(policy.terminal_json).get("TERMINAL_ENV")
    else:
        from tools.terminal_tool import _get_env_config
        backend = _get_env_config().get("env_type")
    return str(backend or "local").strip().lower() == "local"


def _entry_metadata(runner, session_key):
    """The chat's routing-entry metadata, or None when this runner keeps no entry for it (a one-shot
    or API runner without a routing store; the turn runner reads ``_entries`` the same way)."""
    entry = getattr(getattr(runner, "session_store", None), "_entries", {}).get(session_key)
    metadata = getattr(entry, "metadata", None)
    return metadata if isinstance(metadata, dict) else None


def restore_turn_cwd(runner, session_key: str | None, policy) -> None:
    """Seed the chat's cwd record from its routing entry when this process has none yet."""
    if not session_key or not _local_backend(policy):
        return
    from tools.terminal_tool import get_session_cwd, record_session_cwd
    if get_session_cwd(session_key) is not None:
        return
    saved = (_entry_metadata(runner, session_key) or {}).get(_METADATA_KEY)
    if isinstance(saved, str) and os.path.isabs(saved) and os.path.isdir(saved):
        record_session_cwd(session_key, saved)


def persist_turn_cwd(runner, session_key: str | None, policy) -> None:
    """Write the chat's recorded cwd onto its routing entry when it changed (no write otherwise)."""
    if not session_key or not _local_backend(policy):
        return
    from tools.terminal_tool import get_session_cwd
    cwd = get_session_cwd(session_key)
    metadata = _entry_metadata(runner, session_key) if cwd else None
    if metadata is None or metadata.get(_METADATA_KEY) == cwd:
        return
    try:
        runner.session_store.set_session_metadata(session_key, _METADATA_KEY, cwd)
    except Exception as exc:  # health: allow BLE001 -- runs in the turn's finally: a failed save must not fail the finished turn
        logger.warning("Could not save the terminal cwd of %s: %s", session_key, exc)
