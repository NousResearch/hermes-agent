"""External resume-request marker contract (dashboard -> gateway).

Same problem as ``drain_control.py`` (its module docstring: "there is no HTTP
control channel into a running gateway... driven only by the gateway
reacting to its own inputs"), for a different action: the Telegram Mini
App's "make this the active session for this chat" button needs to switch a
LIVE gateway's ``SessionStore`` routing entry, but ``SessionStore`` loads its
routing table once at gateway startup and never re-reads it from disk (see
``gateway/session.py``'s ``SessionStore._ensure_loaded_locked``) -- a direct
write to the underlying ``gateway_routing`` table from the dashboard process
would be silently clobbered the next time the live gateway calls its own
``_save()`` for any unrelated reason (a full-table replace from its own
stale in-memory picture).

So, like drain: the dashboard endpoint writes a marker, and a gateway
background watcher (``GatewayRunner._resume_control_watcher``) observes it
and performs the actual switch in-process, reusing the exact same
``_apply_session_switch`` the ``/resume`` slash command calls.

Contract (a JSON object keyed by session_key, unlike drain's single
presence-based flag -- multiple chats can have independent pending resume
requests at once):

    {
      "<session_key>": {
        "target_session_id": <str>,
        "requested_at": <iso>,
        "principal": <str>,
        "epoch": <instantiation-epoch>
      },
      ...
    }

Each entry is cleared (see :func:`clear_resume_request`) once the watcher
has applied it -- successfully or not (a target that no longer resolves is
not retried forever). The epoch stamp mirrors drain's NS-570 fix: a request
that survives a machine restart on a durable HERMES_HOME volume must not
fire against a freshly-restarted gateway whose in-memory session_store may
not even have loaded that session_key's entry yet, and whose admin no longer
necessarily wants that switch applied out of context.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from gateway.drain_control import current_instantiation_epoch
from hermes_constants import get_hermes_home
from utils import atomic_json_write

_log = logging.getLogger(__name__)

_RESUME_REQUESTS_FILENAME = ".miniapp_resume_requests.json"


def resume_requests_path(home: Optional[Path] = None) -> Path:
    """Absolute path to the resume-requests marker, respecting HERMES_HOME."""
    base = home if home is not None else get_hermes_home()
    return Path(base) / _RESUME_REQUESTS_FILENAME


def read_resume_requests(*, home: Optional[Path] = None) -> dict[str, Any]:
    """Return the full pending-requests mapping. Never raises.

    A missing file returns ``{}``. A present-but-unparseable file (or one
    whose top-level JSON isn't an object) also returns ``{}`` rather than
    raising -- unlike drain's single marker, a corrupt requests file has no
    safe "fail toward acting" reading (there's no single flag to default to
    True), so failing toward "no pending requests" is the only sound choice.
    """
    path = resume_requests_path(home)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as e:
        _log.warning("resume-control: failed to read %s: %s", path, e)
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_resume_request(
    session_key: str,
    target_session_id: str,
    *,
    principal: str = "dashboard",
    home: Optional[Path] = None,
) -> dict[str, Any]:
    """Add (or replace) one pending resume request. Returns the entry written.

    Atomic read-modify-write of the whole file so a concurrent write for a
    DIFFERENT session_key is never lost. Re-requesting the same session_key
    before the watcher applies it just replaces the pending target -- last
    write wins, matching the drain marker's own re-write-refreshes semantics.
    """
    entry = {
        "target_session_id": target_session_id,
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "principal": principal,
        "epoch": current_instantiation_epoch(),
    }
    requests = read_resume_requests(home=home)
    requests[session_key] = entry
    atomic_json_write(resume_requests_path(home), requests)
    return entry


def clear_resume_request(session_key: str, *, home: Optional[Path] = None) -> bool:
    """Remove one pending request by session_key. Returns True if one existed.

    Best-effort and idempotent, like ``drain_control.clear_drain_request``.
    Leaves every OTHER pending session_key's entry untouched.
    """
    requests = read_resume_requests(home=home)
    if session_key not in requests:
        return False
    del requests[session_key]
    try:
        atomic_json_write(resume_requests_path(home), requests)
    except OSError as e:
        _log.warning("resume-control: failed to clear %s: %s", session_key, e)
        return False
    return True


def pending_resume_requests(*, home: Optional[Path] = None) -> dict[str, str]:
    """session_key -> target_session_id for every request whose epoch is
    current (or lenient-absent), mirroring ``drain_requested``'s staleness
    handling. A request whose epoch is a DEFINITE mismatch (survived a
    machine restart on a durable volume) is silently dropped from the
    returned mapping -- the caller should still explicitly clear it via
    :func:`clear_resume_request` so it doesn't linger in the file forever.
    """
    current = current_instantiation_epoch()
    out: dict[str, str] = {}
    for session_key, entry in read_resume_requests(home=home).items():
        if not isinstance(entry, dict):
            continue
        target = entry.get("target_session_id")
        if not target:
            continue
        entry_epoch = entry.get("epoch")
        if current and entry_epoch and entry_epoch != current:
            continue
        out[session_key] = target
    return out
