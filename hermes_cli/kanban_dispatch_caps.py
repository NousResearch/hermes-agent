"""Board-level kanban WIP caps.

The dispatcher lock lives at the shared kanban root
(``kanban_home()/kanban/.dispatcher.lock``). ``config.yaml`` does not: each
gateway reads the profile that happened to win that lock. A lock winner with
``kanban.max_in_progress`` / ``kanban.max_in_progress_per_profile`` unset used
to fall through to the memory-derived default (uncapped where memory cannot
be read) even when another profile had set a real cap.

Those two keys bound the shared board, not the lock winner. Precedence:

1. Collect every explicit positive integer from the default home's
   ``config.yaml`` and each live named profile's ``config.yaml``, plus the
   dispatching process's already-parsed values.
2. The effective cap for each key is the minimum of those explicit values.
   A profile that leaves the key unset has no opinion — it does not uncap
   a profile that set one, and it does not override a tighter value.
3. When no profile has ``max_in_progress`` set, callers still apply
   ``resolve_max_in_progress`` (memory-derived default).
   ``max_in_progress_per_profile`` stays unlimited only when nobody set it.
4. A file that cannot be read is skipped. Its last successful values are
   kept, so a mid-write or permission error cannot open the gate. A missing
   file or a successful read that omits the key is intentional and clears
   that file's contribution.

Other ``kanban.*`` settings stay profile-local to the lock winner. This is
not general config inheritance between profiles.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_CAP_KEYS = ("max_in_progress", "max_in_progress_per_profile")

# Last successful (max_in_progress, max_in_progress_per_profile) per config
# path. A read error reuses this instead of treating the file as unset.
_file_caps: dict[str, tuple[Optional[int], Optional[int]]] = {}
_known_paths: list[str] = []
_lock = threading.Lock()


def positive_cap(raw: Any) -> Optional[int]:
    """Positive int cap, or None when unset / invalid. Bool is not a cap."""
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 1 else None


def _min_cap(*values: Optional[int]) -> Optional[int]:
    present = [v for v in values if v is not None]
    return min(present) if present else None


def _path_key(path: Path) -> str:
    try:
        return str(path.expanduser().resolve())
    except OSError:
        return str(path.expanduser())


def _config_paths() -> list[Path]:
    """Default-home config plus every live named profile's config.yaml.

    Paths come from ``get_default_hermes_root()`` — the same root the shared
    board uses — never from the dispatching process's ``HERMES_HOME`` alone.
    """
    from hermes_constants import (
        PROFILE_ID_RE,
        get_default_hermes_root,
        named_profile_is_deleted,
    )

    root = get_default_hermes_root()
    paths = [root / "config.yaml"]
    profiles = root / "profiles"
    if not profiles.is_dir():
        return paths
    for entry in profiles.iterdir():
        if not entry.is_dir() or not PROFILE_ID_RE.match(entry.name):
            continue
        try:
            if named_profile_is_deleted(entry):
                continue
        except OSError:
            continue
        cfg = entry / "config.yaml"
        if cfg.is_file():
            paths.append(cfg)
    return paths


def _read_file_caps(path: Path) -> tuple[Optional[int], Optional[int]]:
    """Explicit caps from one file. Raises on a read/parse failure.

    ``FileNotFoundError`` is not a failure: the file is gone, so it
    contributes nothing. A parsed file with the key absent contributes
    nothing as well (intentional unset).
    """
    from utils import fast_safe_load

    with open(path, encoding="utf-8-sig") as fh:
        data = fast_safe_load(fh)
    if data is None:
        return (None, None)
    if not isinstance(data, dict):
        raise ValueError(
            f"top-level YAML must be a mapping, got {type(data).__name__}"
        )
    kanban = data.get("kanban", {})
    if kanban is None:
        return (None, None)
    if not isinstance(kanban, dict):
        raise ValueError(
            f"kanban must be a mapping, got {type(kanban).__name__}"
        )
    return (positive_cap(kanban.get("max_in_progress")),
            positive_cap(kanban.get("max_in_progress_per_profile")))


def _caps_for_path(path: Path) -> tuple[Optional[int], Optional[int]]:
    key = _path_key(path)
    try:
        caps = _read_file_caps(path)
    except FileNotFoundError:
        with _lock:
            _file_caps.pop(key, None)
        return (None, None)
    except Exception as exc:
        with _lock:
            cached = _file_caps.get(key)
        if cached is not None:
            logger.warning(
                "kanban dispatcher: cannot re-read %s (%s); keeping last "
                "explicit caps %s",
                path, exc, cached,
            )
            return cached
        logger.warning(
            "kanban dispatcher: cannot read %s (%s); skipping that profile's "
            "WIP caps",
            path, exc,
        )
        return (None, None)
    with _lock:
        _file_caps[key] = caps
    return caps


def _scan_files() -> tuple[Optional[int], Optional[int]]:
    try:
        paths = _config_paths()
    except Exception as exc:
        logger.warning(
            "kanban dispatcher: cannot list profile configs (%s); using last "
            "known paths",
            exc,
        )
        with _lock:
            paths = [Path(p) for p in _known_paths]
    else:
        with _lock:
            _known_paths[:] = [_path_key(p) for p in paths]
    mip: Optional[int] = None
    per_profile: Optional[int] = None
    for path in paths:
        file_mip, file_pp = _caps_for_path(path)
        mip = _min_cap(mip, file_mip)
        per_profile = _min_cap(per_profile, file_pp)
    return mip, per_profile


def explicit_dispatch_caps(
    own_max_in_progress: Optional[int] = None,
    own_max_in_progress_per_profile: Optional[int] = None,
) -> tuple[Optional[int], Optional[int]]:
    """Most restrictive explicit WIP caps for the shared board.

    ``own_*`` are the dispatching process's already-parsed values (so a
    live re-read of its config still counts). Either may be None. Never
    raises: a profile that cannot be read is skipped.
    """
    try:
        scanned_mip, scanned_pp = _scan_files()
    except Exception as exc:
        logger.warning(
            "kanban dispatcher: board cap scan failed (%s); using this "
            "profile's caps only",
            exc,
        )
        scanned_mip, scanned_pp = None, None
    return (
        _min_cap(positive_cap(own_max_in_progress), scanned_mip),
        _min_cap(positive_cap(own_max_in_progress_per_profile), scanned_pp),
    )
