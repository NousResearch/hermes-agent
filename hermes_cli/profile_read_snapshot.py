"""Non-mutating admission for a read-only handle to one named profile generation."""
from contextlib import contextmanager
from pathlib import Path

from hermes_constants import assert_named_profile_home_available, profile_deletion_marker_path
from hermes_cli.profile_incarnation import read_profile_incarnation


@contextmanager
def profile_read_snapshot(home: Path, expected_incarnation: str | None):
    """Check both sides of opening a read-only handle without creating lock files.

    The opened SQLite descriptor pins its file, and tracked handles participate
    in lifecycle retirement. If publication races the open, reject before the
    handle escapes. Directory identity also fences legacy profiles without a
    marker; inspection must never backfill one.
    """
    if profile_deletion_marker_path(home) is None:
        yield expected_incarnation
        return

    def snapshot():
        assert_named_profile_home_available(home)
        before = home.stat()
        token = read_profile_incarnation(home)
        after = home.stat()
        identity = (before.st_dev, before.st_ino)
        if identity != (after.st_dev, after.st_ino):
            raise FileNotFoundError(f"Named profile changed during read admission: {home}")
        return identity, token

    captured = snapshot()
    if expected_incarnation is not None and captured[1] != expected_incarnation:
        raise FileNotFoundError(f"Named profile incarnation is stale: {home}")
    yield captured[1]
    if snapshot() != captured:
        raise FileNotFoundError(f"Named profile changed during read admission: {home}")
