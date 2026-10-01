"""Interruption flags and persisted outcomes for active cron executions."""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("cron.scheduler")


def mark_running_jobs_interrupted(
    reason: str, *, only_owners: Optional[set] = None,
) -> list:
    """Best-effort: mark every in-flight cron job interrupted; returns the job IDs marked.

    Called by gateway shutdown right after ``process_registry.kill_all()``: a job whose tool was
    killed must never report success. ``only_owners`` (``(job_id, fire_owner)`` pairs) restricts
    marking. Tokens go into ``_interrupted_job_ids`` BEFORE ``last_status`` is written so
    ``run_one_job`` sees them.
    """
    from cron.scheduler import (
        _inflight_home_path, _interrupted_job_ids, _restart_safe_waiter_job_ids,
        _running_fire_owners, _running_job_ids, _running_lock, mark_job_run, use_cron_store,
    )
    with _running_lock:
        restart_safe_waiters = set(_restart_safe_waiter_job_ids)
        active_fires = [
            (token, key, owner, profile_home)
            for key, executions in _running_fire_owners.items()
            if key not in restart_safe_waiters
            for token, (owner, profile_home) in executions.items()
        ]
        if only_owners is not None:
            active_fires = [fire for fire in active_fires if (fire[1][1], fire[2]) in only_owners]
        registered_keys = {key for _t, key, _o, _p in active_fires}
        if only_owners is None:
            # The key's home half IS the profile home this claim belongs to — the only record of
            # it for a claim that never reached ``_running_fire_owners``. Read the real Path back
            # from the key, never ``Path(key[0])``: the key is normcased.
            active_fires.extend(
                (None, key, None, _inflight_home_path(key[0]))
                for key in (
                    _running_job_ids - registered_keys - restart_safe_waiters
                )
            )
        _interrupted_job_ids.update(
            token if token is not None else key
            for token, key, _owner, _profile_home in active_fires
        )
    marked = []
    for _token, key, fire_owner, profile_home in active_fires:
        job_id = key[1]
        if not fire_owner:
            logger.warning(
                "Job '%s' interrupted before its durable fire owner was registered; "
                "leaving persisted state untouched",
                job_id)
            # Still report it: shutdown uses the returned IDs for the interrupted-cron notice. The
            # in-memory flag WAS recorded above; only the persisted last_status write is skipped.
            # See #82232.
            marked.append(job_id)
            continue
        try:
            with use_cron_store(profile_home):
                if mark_job_run(
                    job_id, False, reason, expected_fire_owner=fire_owner):
                    marked.append(job_id)
        except Exception as e:
            logger.warning("Failed to mark job %s interrupted: %s", job_id, e)
    return marked

def _is_interrupted(job_id: str, token: Optional[object] = None) -> bool:
    """Non-destructive peek: has shutdown marked THIS execution interrupted? Used before deciding
    what to deliver; does not clear the flag (the authoritative pre-``last_status`` check needs it).
    ``token`` scopes to one execution so a fresh run reusing the job ID isn't poisoned."""
    from cron.scheduler import _inflight_key, _running_lock, _interrupted_job_ids
    key = _inflight_key(job_id)
    with _running_lock:
        if token is not None and token in _interrupted_job_ids:
            return True
        return key in _interrupted_job_ids

def _consume_interrupted_flag(job_id: str, token: Optional[object] = None) -> bool:
    """Return True and clear the flag if shutdown marked THIS execution interrupted. Called right
    before ``last_status`` is written; consuming stops the flag leaking into a later run."""
    from cron.scheduler import _inflight_key, _running_lock, _interrupted_job_ids
    key = _inflight_key(job_id)
    with _running_lock:
        hit = False
        if token is not None and token in _interrupted_job_ids:
            _interrupted_job_ids.discard(token)
            hit = True
        if key in _interrupted_job_ids:
            _interrupted_job_ids.discard(key)
            hit = True
        return hit
