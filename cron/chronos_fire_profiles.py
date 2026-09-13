"""Bounded job-to-profile hints for authenticated Chronos callbacks."""

import hashlib
from pathlib import Path
from typing import Optional

from hermes_constants import get_default_hermes_root
from utils import atomic_write_text


def _cron_fire_profile_hint_path(job_id: str) -> Optional[Path]:
    if not isinstance(job_id, str) or not job_id or len(job_id) > 128:
        return None
    try:
        key = hashlib.sha256(job_id.encode("utf-8")).hexdigest()
    except UnicodeError:
        return None
    return get_default_hermes_root() / "cron" / "fire_profile_index" / f"{key}.profile"


def resolve_cron_fire_profile_hint(job_id: str) -> Optional[str]:
    """Read one bounded hint, never a profile/job scan or an authentication grant."""
    from hermes_cli.profiles import validate_profile_name

    path = _cron_fire_profile_hint_path(job_id)
    if path is None:
        return None
    try:
        with path.open(encoding="utf-8") as stream:
            profile = stream.read(66).strip()  # profile ids are at most 64 characters
        validate_profile_name(profile)
    except (OSError, UnicodeError, ValueError):
        return None
    return profile


def record_cron_fire_profile_hint(job_id: str, profile: Optional[str] = None) -> None:
    """Persist routing before arming, retaining it for authenticated late callbacks.

    A cancelled job still needs to authenticate to receive `gone`; a provision timeout
    may also have armed NAS successfully. Existence and JWT checks remain with the receiver.
    """
    from hermes_cli.profiles import get_active_profile_name, validate_profile_name

    profile = profile or get_active_profile_name()
    if profile == "custom":
        return
    validate_profile_name(profile)
    path = _cron_fire_profile_hint_path(job_id)
    if path is None or resolve_cron_fire_profile_hint(job_id) == profile:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, f"{profile}\n", create_mode=0o600)
