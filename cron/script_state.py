"""Per-job identity and private state directory for cron scripts.

Every ``script`` / ``monitor_script`` / ``no_agent`` run gets ``HERMES_CRON_JOB_ID`` and
``HERMES_CRON_STATE_DIR`` (``<cron dir>/state/<job_id>/``, owner-only) in its environment, so a
gate script can keep "last seen" cursors between ticks without inventing a path. The directory
lives beside ``jobs.json`` in the job's own cron store, so it follows the job's profile, and
``cron.jobs.remove_job`` deletes it with the job (``clear_job_side_state``).
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Job ids are generated hex; hand-edited stores may hold other ids, but only a single safe path
# component may ever name a directory.
_SAFE_JOB_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


def job_state_dir(job_id: Optional[str]) -> Optional[Path]:
    """``<cron dir>/state/<job_id>`` for a safe job id, else None (nothing is created)."""
    text = str(job_id or "")
    if not _SAFE_JOB_ID.fullmatch(text):
        return None
    from cron.jobs import _current_cron_store

    return _current_cron_store().cron_dir / "state" / text


def script_identity_env(job_id: Optional[str]) -> dict[str, str]:
    """Env overlay naming the job and its state dir (created 0700); empty for an unsafe id."""
    state_dir = job_state_dir(job_id)
    if state_dir is None:
        return {}
    from cron.jobs import _ensure_cron_dir, _secure_dir

    _ensure_cron_dir(state_dir.parent)
    state_dir.mkdir(mode=0o700, exist_ok=True)
    _secure_dir(state_dir)
    return {"HERMES_CRON_JOB_ID": str(job_id), "HERMES_CRON_STATE_DIR": str(state_dir)}


def clear_job_side_state(job_id: str) -> None:
    """Best-effort removal of a deleted job's notepad rows and script state dir: neither may
    block the removal itself."""
    from cron import notepad

    try:
        notepad.clear_notepad(job_id)
    except Exception:
        logger.debug("Failed to clear notepad for removed job %s", job_id, exc_info=True)
    state_dir = job_state_dir(job_id)
    if state_dir is not None and state_dir.is_dir():
        shutil.rmtree(state_dir, ignore_errors=True)
