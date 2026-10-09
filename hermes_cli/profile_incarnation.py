"""Persisted identity for one published named-profile generation.

A profile name/path is reusable after deletion, so it cannot identify the
lifecycle object held by a deferred callback.  Each create/import publishes a
fresh token inside the profile home; existing profiles receive one lazily on
first use.  Callers retain the token and compare it before touching profile
state so an old callback cannot cross a delete/recreate ABA boundary.
"""

from __future__ import annotations

import os
import re
import secrets
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from hermes_constants import (
    named_profile_home_is_unavailable,
    profile_deletion_marker_path,
)
from hermes_cli.profile_lifecycle import (
    _profile_lifecycle_modes,
    _write_token_file,
    profile_lifecycle_lease as _profile_mutation_lease,
)
from pm.filesystem import hard_link_refused

PROFILE_INCARNATION_FILENAME = ".profile-incarnation"
_INCARNATION_RE = re.compile(r"^[0-9a-f]{32}$")


def _incarnation_path(profile_home: Path | str) -> Path:
    return Path(profile_home) / PROFILE_INCARNATION_FILENAME


def _temp_marker_path(profile_home: Path) -> Path:
    return profile_home / (
        f"{PROFILE_INCARNATION_FILENAME}.{os.getpid()}.{secrets.token_hex(6)}.tmp"
    )


def _validate_incarnation(value: str, path: Path) -> str:
    token = value.strip().lower()
    if not _INCARNATION_RE.fullmatch(token):
        raise RuntimeError(f"Invalid profile incarnation marker: {path}")
    return token


def read_incarnation_marker(path: Path | str) -> str | None:
    """Read and validate one incarnation marker file."""
    marker = Path(path)
    try:
        value = marker.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return None
    return _validate_incarnation(value, marker)


def read_profile_deletion_incarnation(profile_home: Path | str) -> str | None:
    """Read a deletion token, accepting only the two historical tokenless fences.

    A tokenless fence authorizes deletion retry, never incarnation backfill or
    resource admission. Other invalid contents remain corruption errors.
    """
    marker = profile_deletion_marker_path(Path(profile_home))
    if marker is None:
        raise ValueError(f"Not a named profile home: {profile_home}")
    value = marker.read_text(encoding="utf-8-sig")
    if value in ("", "deleted\n"):
        return None
    return _validate_incarnation(value, marker)


def read_profile_incarnation(profile_home: Path | str) -> str | None:
    """Read a named profile's persisted incarnation, or None if not initialized."""
    home = Path(profile_home)
    if profile_deletion_marker_path(home) is None:
        return None
    return read_incarnation_marker(_incarnation_path(home))


@contextmanager
def profile_incarnation_lease(
    profile_home: Path | str,
    expected_incarnation: str | None = None,
    *,
    require_incarnation: bool = False,
) -> Iterator[Path]:
    """Hold the profile-mutation lease while binding a checked named path.

    Custom/default homes are not reusable named lifecycle objects and keep
    their legacy lock-free behavior. Contention raises TimeoutError; only a
    missing/retired generation raises FileNotFoundError.
    """
    home = Path(profile_home)
    marker = profile_deletion_marker_path(home)
    if marker is None:
        yield home
        return

    # Leases fence keystrokes, uploads and cold SessionDB opens: the checks
    # under the lock reuse this resolution instead of walking the path again.
    with _profile_mutation_lease(home):
        if named_profile_home_is_unavailable(home, marker=marker):
            raise FileNotFoundError(
                f"Named profile home is missing or being deleted: {home}"
            )
        if expected_incarnation is None and require_incarnation:
            raise FileNotFoundError(f"Named profile incarnation is required: {home}")
        assert_profile_incarnation_current(home, expected_incarnation, named=True)
        yield home


def write_fresh_profile_incarnation(profile_home: Path | str) -> str:
    """Atomically replace a profile home's marker with a fresh incarnation."""
    home = Path(profile_home)
    if not home.is_dir():
        raise FileNotFoundError(f"Profile home does not exist: {home}")
    token = secrets.token_hex(16)
    path = _incarnation_path(home)
    mode = _profile_lifecycle_modes(home)[1]
    temp = _temp_marker_path(home)
    from utils import atomic_replace

    try:
        _write_token_file(temp, token, mode)
        # Antivirus and indexers briefly hold a just-written file on Windows.
        atomic_replace(temp, path)
        try:
            os.chmod(path, mode)
        except OSError:
            pass
    finally:
        temp.unlink(missing_ok=True)
    return token


def ensure_profile_incarnation(profile_home: Path | str) -> str | None:
    """Return a stable incarnation, lazily backfilling pre-marker profiles.

    Only named profiles participate.  Publication/deletion tombstones are
    checked before and after the O_EXCL write so this compatibility backfill
    never turns into authority to revive a retired home.
    """
    home = Path(profile_home)
    if profile_deletion_marker_path(home) is None:
        return None
    with profile_incarnation_lease(home):
        current = read_profile_incarnation(home)
        if current is not None:
            return current

        token = secrets.token_hex(16)
        path = _incarnation_path(home)
        temp = _temp_marker_path(home)
        try:
            _write_token_file(temp, token, _profile_lifecycle_modes(home)[1])
            try:
                # Hard-link publication is atomic and never replaces another
                # process's winner. The marker becomes visible only after its
                # complete token has been flushed.
                os.link(temp, path)
            except FileExistsError:
                pass
            except OSError as exc:
                if not hard_link_refused(exc):
                    raise
                # FAT/exFAT, Android app data, some SMB/FUSE mounts. The lease
                # serializes every live-marker writer and the marker was absent
                # above, so a rename cannot replace a winner either; unlike an
                # O_EXCL reservation it never exposes an empty marker to the
                # lease-free readers or strands one after a crash.
                from utils import atomic_replace

                atomic_replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)

        current = read_profile_incarnation(home)
        if current is None or named_profile_home_is_unavailable(home):
            raise FileNotFoundError(
                f"Named profile home disappeared during incarnation setup: {home}"
            )
        return current


def profile_incarnation_matches(
    profile_home: Path | str,
    expected_incarnation: str,
    *,
    named: bool = False,
) -> bool:
    """Return whether the currently published home is the expected generation.

    ``named``: the caller already resolved *profile_home* as a named home, so skip that resolve.
    """
    try:
        current = (read_incarnation_marker(_incarnation_path(profile_home)) if named
                   else read_profile_incarnation(profile_home))
    except (OSError, RuntimeError):
        return False
    return current is not None and secrets.compare_digest(current, expected_incarnation)


def assert_profile_incarnation_current(
    profile_home: Path | str,
    expected_incarnation: str | None,
    *,
    named: bool = False,
) -> None:
    """Refuse a caller bound to a retired generation; ``None`` binds no generation."""
    if expected_incarnation is not None and not profile_incarnation_matches(
        profile_home, expected_incarnation, named=named,
    ):
        raise FileNotFoundError(f"Named profile incarnation is stale: {profile_home}")
