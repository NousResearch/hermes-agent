"""Post-update cron safety net for field-level loss: restore agent-job prompts an update
collapsed to the job name while the job count stayed the same (issue #82990).

Split out of ``hermes_cli.backup`` (which owns the quick snapshots this compares against and
the count-based net); ``backup`` names are late-imported so its patch seams
(``_sibling_profile_homes``) hold.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home

# Log-record parity with the origin module.
logger = logging.getLogger("hermes_cli.backup")


def _load_cron_jobs_doc(path: Path) -> Optional[Any]:
    """Parse ``path`` as the canonical ``{"jobs": [...]}`` doc (legacy bare list honoured).

    ``None`` = missing/unreadable/non-dict-with-list — same dialect rules as
    :func:`_count_cron_jobs` (utf-8-sig for Windows BOMs). Never raises.
    """
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(data, dict):
        jobs = data.get("jobs", [])
        return data if isinstance(jobs, list) else None
    if isinstance(data, list):
        return data
    return None


def _cron_jobs_list(doc: Any) -> list[Any]:
    """The job list out of either document shape. Empty when malformed."""
    if isinstance(doc, list):
        return doc
    if isinstance(doc, dict):
        jobs = doc.get("jobs", [])
        return jobs if isinstance(jobs, list) else []
    return []


def _prompt_degraded(job: dict[str, Any]) -> bool:
    """True when an agent job's prompt field is unusable: blank, missing, or
    collapsed to the job's own name (a name is not a prompt)."""
    if job.get("no_agent"):
        return False
    prompt = job.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return True
    return prompt.strip() == str(job.get("name", "")).strip()


def _snapshot_prompts_by_id(snap_doc: Any) -> dict[str, str]:
    """Usable (non-blank) snapshot prompts keyed by job id."""
    prompts: dict[str, str] = {}
    for job in _cron_jobs_list(snap_doc):
        if not isinstance(job, dict):
            continue
        prompt = job.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            prompts[str(job.get("id", ""))] = prompt
    return prompts


def _restore_degraded_prompts(
    jobs: list[Any], snap_prompts: dict[str, str], *, apply: bool
) -> list[str]:
    """Ids of jobs whose prompt is degraded AND the snapshot has a usable prompt for;
    with ``apply`` the prompt field (only) is restored in place."""
    restored: list[str] = []
    for job in jobs:
        if not isinstance(job, dict) or not _prompt_degraded(job):
            continue
        job_id = str(job.get("id", ""))
        snap_prompt = snap_prompts.get(job_id)
        if snap_prompt is None:
            continue
        if apply:
            job["prompt"] = snap_prompt
        restored.append(job_id)
    return restored


def _restore_under_cron_lock(home: Path, snap_prompts: dict[str, str]) -> list[str]:
    """Re-read ``home``'s jobs under the canonical cron writer lock, restore the prompts that
    are STILL degraded, and publish through the locked, merge-aware cron save path.

    Publishing a document read before the lock would erase any cron edit committed in between
    (create, remove, schedule/prompt edit). ``use_cron_store`` pins both the lock file and the
    store to ``home`` so a sibling profile's recovery serializes with THAT profile's writers.
    Raises ``RuntimeError`` on an unreadable store and ``OSError`` on a failed write.
    """
    from cron import jobs as cron_jobs

    with cron_jobs.use_cron_store(home), cron_jobs._jobs_lock():
        jobs = cron_jobs.load_jobs()
        restored_ids = _restore_degraded_prompts(jobs, snap_prompts, apply=True)
        if restored_ids:
            cron_jobs.save_jobs(jobs)
    return restored_ids


def restore_cron_prompt_fields_if_degraded(
    snapshot_id: str,
    hermes_home: Optional[Path] = None,
) -> Optional[dict[str, Any]]:
    """Safety net for field-level cron-job degradation across ``hermes update``.

    A writer active during the update's mutation window replaced every
    agent-job ``prompt`` with the job's own ``name`` while the job COUNT
    stayed identical, so the count-based net
    (:func:`restore_cron_jobs_if_emptied`) passed the loss undetected
    (issue #82990): 6 jobs before, 6 jobs after, every one of them with an
    empty prompt wearing its name.

    Mirrors the field-level pattern of
    :func:`restore_config_model_settings_if_rewritten`: compare the live
    file against the pre-update snapshot taken minutes earlier by this same
    update run, and restore ONLY the ``prompt`` field of a live agent job
    whose id matches a snapshot job — never the whole record, never jobs the
    snapshot does not know. Conservative on purpose:

    - a live prompt is only restored when it is blank/missing or exactly
      equal to the job's own name — a legitimate user edit that merely
      differs from the snapshot is never stomped;
    - ``no_agent`` script jobs are never touched (they have no prompt);
    - a blank snapshot prompt restores nothing (there is nothing better to
      put back).

    Args:
        snapshot_id: The pre-update quick-snapshot id (from
            :func:`create_quick_snapshot`).
        hermes_home: Override for the Hermes home directory (tests/siblings).

    Returns:
        ``None`` when no action was taken (the common, healthy path). On a
        successful restore, ``{"restored": True, "prompts": N,
        "snapshot_id": ...}`` so the caller can warn the user.
    """
    if not snapshot_id:
        return None

    from hermes_cli.backup import _CRON_JOBS_REL, _quick_snapshot_root

    home = hermes_home or get_hermes_home()
    live_path = home / _CRON_JOBS_REL
    snap_path = _quick_snapshot_root(home) / snapshot_id / _CRON_JOBS_REL

    snap_doc = _load_cron_jobs_doc(snap_path)
    if snap_doc is None:
        return None
    snap_prompts = _snapshot_prompts_by_id(snap_doc)
    # Unlocked pre-check keeps the healthy path free of lock/dir side effects. It only gates:
    # the patch itself is decided on a fresh read under the cron writer lock below.
    live_doc = _load_cron_jobs_doc(live_path)
    if live_doc is None or not _restore_degraded_prompts(
        _cron_jobs_list(live_doc), snap_prompts, apply=False
    ):
        return None

    try:
        restored_ids = _restore_under_cron_lock(home, snap_prompts)
    except (OSError, RuntimeError) as exc:
        logger.error(
            "Cron job prompts were degraded during update but auto-restore "
            "failed: %s",
            exc,
        )
        return None
    if not restored_ids:
        return None

    logger.warning(
        "Restored %d cron job prompt(s) from pre-update snapshot %s — job(s) "
        "%s had their prompt replaced by the job name (#82990)",
        len(restored_ids),
        snapshot_id,
        ", ".join(restored_ids),
    )
    return {
        "restored": True,
        "prompts": len(restored_ids),
        "job_ids": restored_ids,
        "snapshot_id": snapshot_id,
    }


def restore_cron_prompt_fields_all_profiles(
    profile_snapshots: dict[str, str],
    invoking_home: Optional[Path] = None,
) -> list[dict[str, Any]]:
    """Run the cron prompt-field safety net for every sibling profile.

    Same contract as :func:`restore_cron_jobs_all_profiles`: each profile's
    live ``cron/jobs.json`` is compared against ITS OWN same-generation
    pre-update snapshot. Returns one result dict per restored profile, each
    with a ``profile`` key added. Never raises.
    """
    restored: list[dict[str, Any]] = []
    if not profile_snapshots:
        return restored
    from hermes_cli.backup import _sibling_profile_homes

    home = invoking_home or get_hermes_home()
    by_name = dict(_sibling_profile_homes(home))
    for name, snap_id in profile_snapshots.items():
        profile_home = by_name.get(name)
        if profile_home is None:
            continue
        try:
            result = restore_cron_prompt_fields_if_degraded(
                snap_id, hermes_home=profile_home
            )
        except Exception as exc:
            logger.debug(
                "Cron prompt-field restore check for profile %s failed: %s",
                name,
                exc,
            )
            continue
        if result:
            result["profile"] = name
            restored.append(result)
    return restored
