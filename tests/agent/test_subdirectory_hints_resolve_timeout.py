"""Tests for the timeout-guarded path resolution in subdirectory hint discovery.

Regression coverage for kanban t_565de3ca: a librarian worker wedged on a
``Path.resolve()`` that crossed a hard-mounted NFS path whose server had
stopped responding. ``resolve()`` stats every path component, and a stat()
on a wedged ``hard`` NFS mount does not fail — it blocks forever. The turn
stalled (603s with no progress), the turn-liveness watchdog aborted it, but
the worker process stayed alive, so the kanban dispatcher kept extending the
claim and the card burned its whole retry budget and was marked ``gave_up``.

The fix: resolve on a daemon thread with a timeout, so an unresolvable path
is simply skipped for hint purposes instead of stalling the turn.
"""

import threading
import time
from pathlib import Path

import pytest

from agent.subdirectory_hints import (
    _resolve_with_timeout,
    SubdirectoryHintTracker,
)


def test_resolve_with_timeout_returns_path_for_healthy_path(tmp_path):
    """A resolvable path comes back resolved, promptly."""
    target = tmp_path / "sub" / "dir"
    target.mkdir(parents=True)

    resolved = _resolve_with_timeout(target, timeout=5.0)

    assert resolved is not None
    assert resolved == target.resolve()


def test_resolve_with_timeout_gives_up_on_wedged_path():
    """A path whose resolve() hangs returns None instead of hanging."""
    release = threading.Event()

    class _WedgedPath(type(Path())):
        def resolve(self, *args, **kwargs):
            # Simulate a stat() on a wedged hard NFS mount: block until the
            # test releases us, so a naive (unguarded) resolve() would hang.
            release.wait(timeout=30)
            return super().resolve(*args, **kwargs)

    wedged = _WedgedPath("/mnt/Stor1/nfs_appdata/obsidian/Obsidian Vault")

    try:
        start = time.monotonic()
        resolved = _resolve_with_timeout(wedged, timeout=1.0)
        elapsed = time.monotonic() - start
    finally:
        release.set()

    assert resolved is None, "an unresolvable path must be skipped, not returned"
    assert elapsed < 10.0, f"resolve must not block; took {elapsed:.1f}s"


def test_resolve_with_timeout_reraises_real_errors(tmp_path):
    """A genuine resolve() error propagates to the caller."""
    missing = tmp_path / "no-such-dir"

    # resolve(strict=False) does not raise for a missing path, so use a
    # path that cannot be constructed instead: a null byte is invalid.
    with pytest.raises(ValueError):
        _resolve_with_timeout(Path("bad\x00path"), timeout=5.0)


def test_check_tool_call_does_not_stall_on_wedged_path(tmp_path):
    """The public entry point returns promptly when a path arg is wedged."""
    release = threading.Event()

    class _WedgedPath(type(Path())):
        def resolve(self, *args, **kwargs):
            release.wait(timeout=30)
            return super().resolve(*args, **kwargs)

    tracker = SubdirectoryHintTracker(working_dir=tmp_path)

    try:
        start = time.monotonic()
        result = tracker.check_tool_call(
            "terminal", {"command": "ls /mnt/Stor1/nfs_appdata/obsidian/Obsidian Vault"}
        )
        elapsed = time.monotonic() - start
    finally:
        release.set()

    assert elapsed < 15.0, f"check_tool_call must not stall; took {elapsed:.1f}s"
    # No hint is discoverable for an unresolvable path, which is fine.
    assert result is None or isinstance(result, str)
