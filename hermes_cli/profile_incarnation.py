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
import stat
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Iterator

from hermes_constants import (
    named_profile_home_is_unavailable,
    profile_deletion_marker_path,
)
from hermes_cli.profile_lifecycle import (
    profile_lifecycle_lease as _profile_mutation_lease,
)

PROFILE_INCARNATION_FILENAME = ".profile-incarnation"
_INCARNATION_RE = re.compile(r"^[0-9a-f]{32}$")


def _incarnation_path(profile_home: Path | str) -> Path:
    return Path(profile_home) / PROFILE_INCARNATION_FILENAME


def _marker_mode(profile_home: Path) -> int:
    try:
        return 0o660 if profile_home.stat().st_mode & stat.S_IWGRP else 0o600
    except OSError:
        return 0o600


def _temp_marker_path(profile_home: Path) -> Path:
    return profile_home / (
        f"{PROFILE_INCARNATION_FILENAME}.{os.getpid()}.{secrets.token_hex(6)}.tmp"
    )


def _write_token_file(path: Path, token: str, mode: int) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(token + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validate_incarnation(value: str, path: Path) -> str:
    token = value.strip().lower()
    if not _INCARNATION_RE.fullmatch(token):
        raise RuntimeError(f"Invalid profile incarnation marker: {path}")
    return token


def read_incarnation_marker(path: Path | str) -> str | None:
    """Read and validate one incarnation marker file."""
    marker = Path(path)
    try:
        value = marker.read_text(encoding="utf-8")
    except FileNotFoundError:
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
    their legacy lock-free behavior. The shared lease lives in the focused
    lifecycle owner so marker code never imports the profile command godfile.
    """
    home = Path(profile_home)
    if profile_deletion_marker_path(home) is None:
        yield home
        return

    lease_stack = ExitStack()
    try:
        lease_stack.enter_context(_profile_mutation_lease())
    except TimeoutError as exc:
        raise FileNotFoundError(
            f"Named profile lifecycle lease is unavailable: {home}"
        ) from exc
    try:
        if named_profile_home_is_unavailable(home):
            raise FileNotFoundError(
                f"Named profile home is missing or being deleted: {home}"
            )
        if expected_incarnation is None:
            if require_incarnation:
                raise FileNotFoundError(
                    f"Named profile incarnation is required: {home}"
                )
        elif not profile_incarnation_matches(home, expected_incarnation):
            raise FileNotFoundError(f"Named profile incarnation is stale: {home}")
        yield home
    finally:
        lease_stack.close()


def write_fresh_profile_incarnation(profile_home: Path | str) -> str:
    """Atomically replace a profile home's marker with a fresh incarnation."""
    home = Path(profile_home)
    if not home.is_dir():
        raise FileNotFoundError(f"Profile home does not exist: {home}")
    token = secrets.token_hex(16)
    path = _incarnation_path(home)
    mode = _marker_mode(home)
    temp = _temp_marker_path(home)
    try:
        _write_token_file(temp, token, mode)
        os.replace(temp, path)
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
            _write_token_file(temp, token, _marker_mode(home))
            try:
                # Hard-link publication is atomic and never replaces another
                # process's winner. The marker becomes visible only after its
                # complete token has been flushed.
                os.link(temp, path)
            except FileExistsError:
                pass
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
) -> bool:
    """Return whether the currently published home is the expected generation."""
    try:
        current = read_profile_incarnation(profile_home)
    except (OSError, RuntimeError):
        return False
    return current is not None and secrets.compare_digest(current, expected_incarnation)
