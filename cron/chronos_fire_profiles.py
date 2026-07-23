"""Bounded job-to-profile hints for authenticated Chronos callbacks."""

import os
import uuid
from pathlib import Path
from typing import Optional

from cron.jobs import _job_output_dir
from hermes_constants import get_default_hermes_root
from utils import atomic_replace


def _cron_fire_profile_index_dir() -> Path:
    """Shared per-job profile hints for public Chronos fire callbacks."""
    return get_default_hermes_root() / "cron" / "fire_profile_index"


def _cron_fire_profile_hint_path(job_id: str) -> Optional[Path]:
    clean = str(job_id or "").strip()
    if not clean:
        return None
    try:
        _job_output_dir(clean)
    except ValueError:
        return None
    return _cron_fire_profile_index_dir() / f"{clean}.profile"


def resolve_cron_fire_profile_hint(job_id: str) -> Optional[str]:
    """Return a bounded profile hint for a Chronos fire job, if known."""
    path = _cron_fire_profile_hint_path(job_id)
    if path is None or not path.is_file():
        return None
    try:
        profile = path.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    return profile or None


def record_cron_fire_profile_hint(job_id: str, profile: str) -> None:
    """Persist a best-effort profile hint for a Chronos-armed job."""
    clean_profile = str(profile or "").strip()
    if not clean_profile or clean_profile == "custom":
        return
    path = _cron_fire_profile_hint_path(job_id)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        tmp_path.write_text(f"{clean_profile}\n", encoding="utf-8")
        atomic_replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def forget_cron_fire_profile_hint(job_id: str) -> None:
    """Remove a Chronos fire profile hint when a job is cancelled or deleted."""
    path = _cron_fire_profile_hint_path(job_id)
    if path is None:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


