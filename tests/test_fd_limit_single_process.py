"""Regression guard for the single-process FD-limit fix (QA campaign F0-1).

macOS defaults RLIMIT_NOFILE to 256, which a single-process run of the whole
suite (or a subsystem like tests/gateway/) exhausts → ``OSError: [Errno 24] Too
many open files`` masquerading as thousands of unrelated errors. ``conftest.py``'s
``pytest_configure`` raises the soft limit at session start. This guard proves the
limit was actually raised, so the fix can't silently regress.
"""
import pytest

resource = pytest.importorskip("resource")  # POSIX-only; skip the module on Windows


def test_soft_fd_limit_raised_for_single_process_runs():
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    # The conftest targets 65536 (or the hard cap if lower). On any POSIX host
    # the soft limit must now be comfortably above the macOS 256 default that
    # caused the FD exhaustion.
    expected = 65536 if hard == resource.RLIM_INFINITY else min(hard, 65536)
    assert soft >= min(expected, 65536), (
        f"soft FD limit {soft} not raised (hard={hard}); single-process suite "
        f"runs will exhaust file descriptors"
    )
    assert soft > 256, "soft FD limit still at/below the macOS default 256"
