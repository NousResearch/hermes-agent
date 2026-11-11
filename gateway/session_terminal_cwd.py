"""A chat's terminal working directory survives a gateway restart (#41128).

The terminal tool records where each command finished (``tools.terminal_tool.record_session_cwd``,
keyed by the chat's session key) in a process-local dict, so a ``cd`` made in one turn holds for the
next. A restart empties that dict and the chat's next command runs in ``terminal.cwd`` again. The
turn now writes the recorded directory onto the chat's routing entry when it changed, and a turn
that finds no record (the first after a restart) seeds it back from there.

Local backend only: a host path can be checked before it is trusted. A remote or container
directory cannot be vouched for from here, and every session-env command starts with
``cd -- <cwd> || exit 126``, so a stale one would break the chat's commands instead of falling back.
A ``/new`` starts a fresh entry, and the new conversation starts in ``terminal.cwd`` (the old record
is dropped at its first turn, never saved onto it).
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
        # Only the scope-aware backend name: the full tool config also parses timeouts, ports and
        # Docker JSON, and a typo there must not fail a turn that never runs a command.
        from tools.terminal_tool import _ensure_terminal_env_bridged
        from tools.terminal_tool_config import _tenv
        _ensure_terminal_env_bridged()
        backend = _tenv("TERMINAL_ENV", "local")
    return str(backend or "local").strip().lower() == "local"


def _routing_entry(runner, session_key):
    """The chat's routing entry, or None when this runner keeps none for it (a one-shot or API
    runner without a routing store; the turn runner reads ``_entries`` the same way)."""
    entry = getattr(getattr(runner, "session_store", None), "_entries", {}).get(session_key)
    return entry if isinstance(getattr(entry, "metadata", None), dict) else None


# Which conversation (routing entry, by its creation stamp) each chat's in-process cwd record
# belongs to. ``/new`` publishes a fresh entry: the record left by the old conversation is dropped at
# that conversation's next turn, inside the turn's own profile scope that qualifies the record's key.
_record_owner: dict[str, object] = {}


def restore_turn_cwd(runner, session_key: str | None, policy) -> object | None:
    """Seed the chat's cwd record from its routing entry when this process has none yet, after
    dropping one an earlier conversation on this chat left. Returns the conversation token
    :func:`persist_turn_cwd` saves against."""
    entry = _routing_entry(runner, session_key) if session_key else None
    if entry is None or not _local_backend(policy):
        return None
    from tools.terminal_tool import _qualify_task_key, clear_session_cwd, get_session_cwd, record_session_cwd
    owner, token = _qualify_task_key(session_key), entry.created_at
    if _record_owner.get(owner, token) != token:
        clear_session_cwd(session_key)
    _record_owner[owner] = token
    if get_session_cwd(session_key) is None:
        saved = entry.metadata.get(_METADATA_KEY)
        if isinstance(saved, str) and os.path.isabs(saved) and os.path.isdir(saved):
            record_session_cwd(session_key, saved)
    return token


def persist_turn_cwd(runner, session_key: str | None, token: object | None) -> None:
    """Write the chat's recorded cwd onto its routing entry when it changed. Only onto the entry the
    turn started on: a ``/new`` during the turn published a fresh one, which starts in ``terminal.cwd``."""
    entry = _routing_entry(runner, session_key) if token is not None else None
    if entry is None or entry.created_at != token:
        return
    from tools.terminal_tool import get_session_cwd
    cwd = get_session_cwd(session_key)
    if not cwd or entry.metadata.get(_METADATA_KEY) == cwd:
        return
    try:
        runner.session_store.set_session_metadata(session_key, _METADATA_KEY, cwd)
    except Exception as exc:  # health: allow BLE001 -- runs in the turn's finally: a failed save must not fail the finished turn
        logger.warning("Could not save the terminal cwd of %s: %s", session_key, exc)
