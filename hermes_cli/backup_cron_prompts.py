"""Post-update cron safety net for field-level loss: restore agent-job prompts an update
collapsed to the job name while the job count stayed the same (issue #82990).

Split out of ``hermes_cli.backup`` (which owns the quick snapshots this compares against and
the count-based net); ``backup`` names are late-imported so its patch seams
(``_sibling_profile_homes``) hold.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

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


def _prompt_degraded(job: Dict[str, Any]) -> bool:
    """True when an agent job's prompt field is unusable: blank, missing, or
    collapsed to the job's own name (a name is not a prompt)."""
    if job.get("no_agent"):
        return False
    prompt = job.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return True
    return prompt.strip() == str(job.get("name", "")).strip()


def restore_cron_prompt_fields_if_degraded(
    snapshot_id: str,
    hermes_home: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
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

    from hermes_cli.backup import _CRON_JOBS_REL, _atomic_output_path, _quick_snapshot_root

    home = hermes_home or get_hermes_home()
    live_path = home / _CRON_JOBS_REL
    snap_path = _quick_snapshot_root(home) / snapshot_id / _CRON_JOBS_REL

    live_doc = _load_cron_jobs_doc(live_path)
    if live_doc is None:
        return None
    snap_doc = _load_cron_jobs_doc(snap_path)
    if snap_doc is None:
        return None

    snap_by_id: Dict[str, Dict[str, Any]] = {}
    for job in _cron_jobs_list(snap_doc):
        if isinstance(job, dict):
            snap_by_id[str(job.get("id", ""))] = job

    restored_ids: list[str] = []
    live_jobs = _cron_jobs_list(live_doc)
    for job in live_jobs:
        if not isinstance(job, dict):
            continue
        snap_job = snap_by_id.get(str(job.get("id", "")))
        if snap_job is None:
            continue
        if not _prompt_degraded(job):
            continue
        snap_prompt = snap_job.get("prompt")
        if not isinstance(snap_prompt, str) or not snap_prompt.strip():
            continue
        job["prompt"] = snap_prompt
        restored_ids.append(str(job.get("id", "")))

    if not restored_ids:
        return None

    try:
        live_path.parent.mkdir(parents=True, exist_ok=True)
        with _atomic_output_path(live_path) as tmp_path:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(live_doc, f, indent=2)
                f.write("\n")
    except (OSError, PermissionError) as exc:
        logger.error(
            "Cron job prompts were degraded during update but auto-restore "
            "failed: %s",
            exc,
        )
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
    profile_snapshots: Dict[str, str],
    invoking_home: Optional[Path] = None,
) -> list[Dict[str, Any]]:
    """Run the cron prompt-field safety net for every sibling profile.

    Same contract as :func:`restore_cron_jobs_all_profiles`: each profile's
    live ``cron/jobs.json`` is compared against ITS OWN same-generation
    pre-update snapshot. Returns one result dict per restored profile, each
    with a ``profile`` key added. Never raises.
    """
    restored: list[Dict[str, Any]] = []
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
