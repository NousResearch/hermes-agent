"""Resume-request dropbox — external writers ask the gateway to resume a session.

Fixes the SGR-6EA95669 clobber race (2026-07-10): external tools (the
safe-restart watcher) used to write ``resume_pending`` directly into
``sessions.json`` — the gateway's OWN persisted state. Two writers on one
last-writer-wins file: the old gateway's final save during a drain window
clobbered the watcher's flag, and the new gateway reads the file exactly once
at boot, so a re-asserted flag was invisible. The initiating session of a
deferred restart silently never resumed.

The dropbox inverts the ownership: external writers drop small request files
into ``<HERMES_HOME>/gateway/resume_requests/`` (atomic tmp+rename, no shared
file mutated); the gateway — the single writer of its own session state —
sweeps the directory at boot and from the housekeeping loop, marks each
requested session ``resume_pending`` through the normal
``session_store.mark_resume_pending()`` path, and deletes the consumed files.

Requests never bypass the existing auto-resume gates: unknown session keys are
skipped, ``suspended`` still wins, the reason must be in
``_AUTO_RESUME_REASONS`` to auto-fire, and the allowlist / adapter / loop-
breaker checks in ``_schedule_resume_pending_sessions`` apply unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

DROPBOX_DIRNAME = "resume_requests"

# A resume request is a point-in-time ask; honoring one long after it was
# written would wake a session nobody is waiting on. One hour is generous for
# any restart/reconnect window while still bounded.
MAX_AGE_SECONDS = 3600.0

_KEY_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def dropbox_dir(hermes_home: Path) -> Path:
    return Path(hermes_home) / "gateway" / DROPBOX_DIRNAME


def submit_resume_request(
    hermes_home: Path,
    session_key: str,
    reason: str = "restart_interrupted",
) -> Path:
    """Write a resume request for *session_key*. For EXTERNAL writers.

    Atomic (tmp+rename) and collision-free (pid+monotonic suffix). Duplicate
    requests for the same key are harmless — the sweep dedups. Never touches
    sessions.json.
    """
    if not session_key:
        raise ValueError("session_key is required")
    directory = dropbox_dir(hermes_home)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "session_key": str(session_key),
        "reason": str(reason),
        "requested_at": time.time(),
    }
    stem = _KEY_SANITIZE_RE.sub("_", str(session_key))[:120]
    final = directory / f"{stem}-{os.getpid()}-{time.monotonic_ns()}.json"
    fd, tmp_name = tempfile.mkstemp(prefix=".resume-req-", dir=str(directory))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, final)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return final


def sweep_resume_requests(
    hermes_home: Path,
    *,
    max_age_seconds: float = MAX_AGE_SECONDS,
) -> List[Tuple[str, str]]:
    """Consume all pending requests. For the GATEWAY (single consumer).

    Returns deduped ``[(session_key, reason)]`` (first reason wins per key).
    Consumed files are deleted; malformed files are renamed ``*.rejected`` so
    they are never re-parsed; stale files (older than *max_age_seconds*) are
    deleted without being returned. Fail-open: any per-file error skips that
    file, never raises out of the sweep.
    """
    directory = dropbox_dir(hermes_home)
    try:
        names = sorted(os.listdir(directory))
    except FileNotFoundError:
        return []
    except OSError as exc:
        logger.warning("resume-request dropbox unreadable (%s): %s", directory, exc)
        return []

    now = time.time()
    results: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for name in names:
        if not name.endswith(".json"):
            continue
        path = directory / name
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            # Typed deferred SELF restarts have a persisted multi-stage
            # lifecycle owned by gateway.deferred_restart. The legacy tuple
            # sweep must never unlink them.
            if payload.get("kind") == "deferred_restart":
                continue
            session_key = str(payload["session_key"])
            reason = str(payload.get("reason") or "restart_interrupted")
            requested_at = float(payload.get("requested_at") or 0.0)
        except Exception as exc:  # noqa: BLE001 — quarantine, don't crash
            logger.warning("resume request %s malformed (%s) — quarantining", name, exc)
            try:
                os.replace(path, str(path) + ".rejected")
            except OSError:
                pass
            continue
        try:
            os.unlink(path)
        except OSError:
            # Another sweep already consumed it (or perms) — don't double-honor.
            continue
        if now - requested_at > max_age_seconds:
            logger.info(
                "resume request for %s dropped as stale (age %.0fs > %.0fs)",
                session_key, now - requested_at, max_age_seconds,
            )
            continue
        if session_key in seen:
            continue
        seen.add(session_key)
        results.append((session_key, reason))
    return results
