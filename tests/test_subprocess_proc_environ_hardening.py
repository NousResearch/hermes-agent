"""Tests for /proc/<ppid>/environ leak hardening (#4427).

Verifies that constructing a subprocess env via _sanitize_subprocess_env() or
_make_run_env() drops the PR_SET_DUMPABLE flag on the parent process so a
same-UID child cannot recover the parent's initial environment by reading
/proc/<ppid>/environ.

See: https://github.com/NousResearch/hermes-agent/issues/4427
"""

import os
import platform

import pytest


_LINUX_ONLY = pytest.mark.skipif(
    platform.system() != "Linux",
    reason="/proc/<pid>/environ leak is Linux-specific",
)


def _prctl_or_skip():
    """Resolve libc.prctl via the same path production uses, or skip the test.

    Production routes through `ctypes.util.find_library("c")` with a
    `CDLL(None)` fallback so musl-based distros (Alpine) work; mirroring that
    here keeps the test from spuriously failing on hosts where `libc.so.6`
    doesn't exist.
    """
    from tools.environments.local import _resolve_prctl
    prctl = _resolve_prctl()
    if prctl is None:
        pytest.skip("libc.prctl unavailable on this host")
    return prctl


_PR_GET_DUMPABLE = 3
_PR_SET_DUMPABLE = 4


def _get_dumpable() -> int:
    return _prctl_or_skip()(_PR_GET_DUMPABLE, 0, 0, 0, 0)


def _set_dumpable(value: int) -> None:
    _prctl_or_skip()(_PR_SET_DUMPABLE, value, 0, 0, 0)


@pytest.fixture
def restore_dumpable():
    original = _get_dumpable()
    _set_dumpable(1)
    yield
    _set_dumpable(original)


@_LINUX_ONLY
class TestProcEnvironHardening:
    """Verify subprocess env construction clears PR_SET_DUMPABLE."""

    def test_sanitize_subprocess_env_clears_dumpable(self, restore_dumpable):
        assert _get_dumpable() == 1
        from tools.environments.local import _sanitize_subprocess_env
        _sanitize_subprocess_env({"PATH": "/usr/bin"})
        assert _get_dumpable() == 0

    def test_make_run_env_clears_dumpable(self, restore_dumpable):
        assert _get_dumpable() == 1
        from tools.environments.local import _make_run_env
        _make_run_env({"PATH": "/usr/bin"})
        assert _get_dumpable() == 0

    def test_hardening_reapplies_when_dumpable_resets(self, restore_dumpable):
        from tools.environments.local import _sanitize_subprocess_env
        _sanitize_subprocess_env({})
        assert _get_dumpable() == 0
        _set_dumpable(1)
        _sanitize_subprocess_env({})
        assert _get_dumpable() == 0

    def test_mcp_build_safe_env_clears_dumpable(self, restore_dumpable):
        assert _get_dumpable() == 1
        from tools.mcp_tool import _build_safe_env
        _build_safe_env(None)
        assert _get_dumpable() == 0

    @pytest.mark.skipif(
        os.geteuid() == 0,
        reason="root bypasses PR_SET_DUMPABLE=0; security property only verifiable as non-root",
    )
    def test_child_cannot_read_parent_environ_after_hardening(self, restore_dumpable):
        import subprocess
        import sys
        from tools.environments.local import _harden_against_proc_environ_leak

        _harden_against_proc_environ_leak()
        assert _get_dumpable() == 0

        ppid = os.getpid()
        child = subprocess.run(
            [sys.executable, "-c", f"open('/proc/{ppid}/environ', 'rb').read()"],
            capture_output=True,
            text=True,
        )
        assert child.returncode != 0, (
            f"child unexpectedly read /proc/{ppid}/environ; "
            f"stdout={child.stdout!r} stderr={child.stderr!r}"
        )
        assert "Permission denied" in child.stderr or "EACCES" in child.stderr
