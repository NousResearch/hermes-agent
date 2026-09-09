"""Suggested cron jobs — proposed automations the user accepts with one tap.

A suggestion is a ready-to-run cron job spec the user accepts (creates the real job) or dismisses
(latched by ``dedup_key`` so it is never re-offered). Every proposal flows through here regardless
of source: ``catalog`` (curated starters), ``blueprint`` (skill ``blueprint:`` blocks, see
``tools/blueprints.py``), ``usage`` (self-improvement review), ``integration`` (connected account).
Accepting calls ``cron.jobs.create_job`` with the stored ``job_spec`` — no second job engine;
nothing auto-creates (consent-first). Storage mirrors ``cron/jobs.py`` (atomic replace, 0600).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home
from hermes_time import now as _hermes_now
from utils import atomic_replace

logger = logging.getLogger(__name__)

# Per-profile by design (anchored on get_hermes_home(), see cron/jobs.py). Optional test override;
# production resolves the path at CALL time so multiplexed profile ticks (set_hermes_home_override)
# cannot leak one profile's suggestions into the import-time home.
# Per-profile by design (issue #4707): suggestions live alongside the active profile's cron store. Anchor on
# get_hermes_home() (profile home), not the shared default root. Same pattern as cron/executions.py.
SUGGESTIONS_FILE: Optional[Path] = None

# Protects load->modify->save cycles (the background review fork and the main agent can both write).
_suggestions_lock = threading.Lock()

# Cap pending suggestions so the list never becomes a nag wall; when full, new ones are dropped.
MAX_PENDING = 5

VALID_SOURCES = frozenset({"catalog", "blueprint", "usage", "integration"})
_STATUS_PENDING = "pending"
_STATUS_ACCEPTED = "accepted"
_STATUS_DISMISSED = "dismissed"


def _current_suggestions_file() -> Path:
    return SUGGESTIONS_FILE or (get_hermes_home().resolve() / "cron" / "suggestions.json")


def _secure_file(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _ensure_dir() -> None:
    from cron.jobs import _ensure_cron_dir

    _ensure_cron_dir(_current_suggestions_file().parent)


def _load_raw() -> Dict[str, Any]:
    suggestions_file = _current_suggestions_file()
    if not suggestions_file.exists():
        return {"suggestions": []}
    try:
        with open(suggestions_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("suggestions.json unreadable (%s); starting empty", e)
        return {"suggestions": []}
    if isinstance(data, dict) and isinstance(data.get("suggestions"), list):
        return data
    if isinstance(data, list):
        return {"suggestions": data}
    logger.warning("suggestions.json malformed; starting empty")
    return {"suggestions": []}


def _save_raw(suggestions: List[Dict[str, Any]]) -> None:
    _ensure_dir()
    suggestions_file = _current_suggestions_file()
    fd, tmp_path = tempfile.mkstemp(dir=str(suggestions_file.parent), suffix=".tmp", prefix=".sugg_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            payload = {"suggestions": suggestions, "updated_at": _hermes_now().isoformat()}
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        atomic_replace(tmp_path, suggestions_file)
        _secure_file(suggestions_file)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_suggestions() -> List[Dict[str, Any]]:
    """Return all suggestion records (any status)."""
    return _load_raw().get("suggestions", [])


def _pending(suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [s for s in suggestions if s.get("status") == _STATUS_PENDING]


def list_pending() -> List[Dict[str, Any]]:
    """Return pending suggestions in creation order (oldest first)."""
    return _pending(load_suggestions())


def add_suggestion(
    *, title: str, description: str, source: str, job_spec: Dict[str, Any], dedup_key: str,
) -> Optional[Dict[str, Any]]:
    """Register a pending suggestion. Returns the record, or None when skipped: the same
    ``dedup_key`` was already decided on or is still pending (never re-offer, never duplicate), or
    the pending list is full (``MAX_PENDING``). ``job_spec`` is passed straight to
    ``cron.jobs.create_job`` on accept."""
    if source not in VALID_SOURCES:
        raise ValueError(f"unknown suggestion source: {source!r}")
    if not title.strip() or not dedup_key.strip():
        raise ValueError("title and dedup_key are required")

    with _suggestions_lock:
        suggestions = _load_raw().get("suggestions", [])
        if any(
            existing.get("dedup_key") == dedup_key
            and existing.get("status") in (_STATUS_DISMISSED, _STATUS_ACCEPTED, _STATUS_PENDING)
            for existing in suggestions
        ):
            return None
        if len(_pending(suggestions)) >= MAX_PENDING:
            logger.info("Suggestion backlog full (%d); dropping %r", MAX_PENDING, title)
            return None

        record = {
            "id": uuid.uuid4().hex[:12],
            "title": title.strip(),
            "description": description.strip(),
            "source": source,
            "job_spec": job_spec,
            "dedup_key": dedup_key.strip(),
            "status": _STATUS_PENDING,
            "created_at": _hermes_now().isoformat(),
        }
        suggestions.append(record)
        _save_raw(suggestions)
        return record


def get_suggestion(ref: str) -> Optional[Dict[str, Any]]:
    """Resolve a suggestion by id, 1-based pending index, or exact (case-insensitive) title."""
    suggestions = load_suggestions()
    for s in suggestions:
        if s.get("id") == ref:
            return s
    if ref.isdigit():
        pending = _pending(suggestions)
        idx = int(ref) - 1
        if 0 <= idx < len(pending):
            return pending[idx]
    for s in suggestions:
        if s.get("title", "").lower() == ref.lower():
            return s
    return None


def _set_status(suggestion_id: str, status: str) -> bool:
    with _suggestions_lock:
        suggestions = _load_raw().get("suggestions", [])
        for s in suggestions:
            if s.get("id") == suggestion_id:
                s["status"] = status
                s["resolved_at"] = _hermes_now().isoformat()
                _save_raw(suggestions)
                return True
        return False


def dismiss_suggestion(ref: str) -> bool:
    """Dismiss a suggestion (latched — never re-offered for its dedup_key)."""
    s = get_suggestion(ref)
    return bool(s) and _set_status(s["id"], _STATUS_DISMISSED)


def accept_suggestion(ref: str, *, origin: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Accept a suggestion: create the real cron job from its ``job_spec``. Returns the job dict, or
    None if not found / not pending. ``origin`` (platform/chat) is merged so "origin" delivery
    routes back to the chat where the user accepted."""
    s = get_suggestion(ref)
    if not s or s.get("status") != _STATUS_PENDING:
        return None

    from cron.scheduler import (
        CronSchedulerRegistrationError, create_job_with_scheduler_registration,
    )

    spec = dict(s.get("job_spec") or {})
    if origin is not None and "origin" not in spec:
        spec["origin"] = origin

    try:
        job = create_job_with_scheduler_registration(**spec)
    except CronSchedulerRegistrationError:
        # The job is already durable: resolve the suggestion so a retry cannot create a second copy.
        _set_status(s["id"], _STATUS_ACCEPTED)
        raise
    _set_status(s["id"], _STATUS_ACCEPTED)
    return job


def clear_resolved() -> int:
    """Drop ACCEPTED records from disk (they served their purpose once the job exists); dismissed
    records are RETAINED for their dedup_key. Returns the count removed."""
    with _suggestions_lock:
        suggestions = _load_raw().get("suggestions", [])
        kept = [s for s in suggestions if s.get("status") != _STATUS_ACCEPTED]
        removed = len(suggestions) - len(kept)
        if removed:
            _save_raw(kept)
        return removed
