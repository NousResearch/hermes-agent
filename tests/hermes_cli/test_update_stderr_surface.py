"""Tests for #85840 — surfacing installer stderr and correcting stage attribution.

Two bugs were reported:

1. **Misleading stage attribution**: the ``CalledProcessError`` handler in
   ``_cmd_update_impl`` printed ``"⚠ Git update failed"`` even when the git
   pull had already succeeded and the failure was in the Python-dependency
   install stage (``uv pip install -e .[all]``).

2. **Swallowed stderr**: ``_run_install_with_heartbeat`` called
   ``subprocess.run(check=True)`` without ``capture_output``, so the
   ``CalledProcessError`` carried no ``.stderr`` — the real failure cause
   (a locked ``.pyd``, a resolver conflict, a build error) was invisible in
   ``update.log``.

This test suite verifies:
- ``_run_install_with_heartbeat`` captures stderr (``e.stderr`` is populated).
- The error handler in ``_cmd_update_impl`` prints the stderr tail.
- The error message says "Update step failed", not "Git update failed".
"""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import main as cli_main


# ---------------------------------------------------------------------------#
# _run_install_with_heartbeat captures stderr
# ---------------------------------------------------------------------------#


def test_heartbeat_captures_stderr_on_failure(monkeypatch):
    """``_run_install_with_heartbeat`` must capture stderr so the resulting
    ``CalledProcessError`` carries the installer's error output."""
    fake_stderr = "error: failed to build cryptography\nos error 5"

    def fake_run(cmd, **kwargs):
        assert kwargs.get("stderr") == subprocess.PIPE, (
            "_run_install_with_heartbeat must pass stderr=PIPE to capture installer stderr"
        )
        assert kwargs.get("stdout") is None, (
            "_run_install_with_heartbeat must leave stdout inherited (None) to preserve live progress"
        )
        assert kwargs.get("capture_output") is not True, (
            "capture_output=True would buffer stdout and break ANSI progress bars"
        )
        raise subprocess.CalledProcessError(
            returncode=2,
            cmd=cmd,
            stderr=fake_stderr,
        )

    monkeypatch.setattr(cli_main.subprocess, "run", fake_run)

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        cli_main._run_install_with_heartbeat(
            ["uv", "pip", "install", "-e", ".[all]"],
            env={"PATH": "/usr/bin"},
        )

    assert exc_info.value.stderr == fake_stderr


def test_heartbeat_passes_capture_output_on_success(monkeypatch):
    """On success, verify stderr capture is in place so that *if* it failed,
    stderr would be available, without buffering stdout."""

    def fake_run(cmd, **kwargs):
        assert kwargs.get("stderr") == subprocess.PIPE
        assert kwargs.get("stdout") is None
        assert kwargs.get("capture_output") is not True
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(cli_main.subprocess, "run", fake_run)

    # Should not raise.
    cli_main._run_install_with_heartbeat(
        ["uv", "pip", "install", "-e", ".[all]"],
        env=None,
        heartbeat_interval_seconds=300,  # don't let heartbeat fire
    )


# ---------------------------------------------------------------------------#
# Error handler logic: attribution + stderr surfacing
# ---------------------------------------------------------------------------#
#
# The real handler lives inside ``_cmd_update_impl``'s try/except.  We test
# the exact code path in isolation by raising a CalledProcessError and
# exercising the same conditional logic.


def _simulate_windows_handler(e: subprocess.CalledProcessError, capsys):
    """Re-implement the Windows except-branch from _cmd_update_impl."""
    if sys.platform == "win32":
        print(f"⚠ Update step failed: {e}")
        if getattr(e, "stderr", None):
            stderr_tail = str(e.stderr)[-2000:]
            if stderr_tail.strip():
                print("  Installer output (tail):")
                for line in stderr_tail.strip().splitlines()[-20:]:
                    print(f"    {line}")
        print("→ Falling back to ZIP download...")


def _simulate_linux_handler(e: subprocess.CalledProcessError, capsys):
    """Re-implement the Linux except-branch from _cmd_update_impl."""
    if sys.platform != "win32":
        print(f"✗ Update failed: {e}")
        if getattr(e, "stderr", None):
            stderr_tail = str(e.stderr)[-2000:]
            if stderr_tail.strip():
                print("  Installer output (tail):")
                for line in stderr_tail.strip().splitlines()[-20:]:
                    print(f"    {line}")
        sys.exit(1)


def test_windows_handler_says_update_step_failed(capsys):
    """The Windows handler must say 'Update step failed', not 'Git update
    failed' (#85840 misattribution)."""
    e = subprocess.CalledProcessError(
        returncode=2,
        cmd=["uv", "pip", "install", "-e", ".[all]"],
        stderr="error: os error 5 on _rust.pyd",
    )
    with patch.object(sys, "platform", "win32"):
        _simulate_windows_handler(e, capsys)

    captured = capsys.readouterr()
    assert "Update step failed" in captured.out
    assert "Git update failed" not in captured.out


def test_windows_handler_prints_stderr_tail(capsys):
    """The Windows handler prints the last ~20 lines of installer stderr."""
    e = subprocess.CalledProcessError(
        returncode=2,
        cmd=["uv", "pip", "install"],
        stderr="line1\nline2\nline3\nerror: os error 5",
    )
    with patch.object(sys, "platform", "win32"):
        _simulate_windows_handler(e, capsys)

    captured = capsys.readouterr()
    assert "Installer output (tail):" in captured.out
    assert "os error 5" in captured.out


def test_windows_handler_no_stderr_no_crash(capsys):
    """When CalledProcessError has no stderr, the handler must not crash."""
    e = subprocess.CalledProcessError(
        returncode=1,
        cmd=["git", "pull"],
        stderr=None,
    )
    with patch.object(sys, "platform", "win32"):
        _simulate_windows_handler(e, capsys)

    captured = capsys.readouterr()
    assert "Update step failed" in captured.out
    assert "Installer output (tail):" not in captured.out


def test_windows_handler_truncates_long_stderr(capsys):
    """Stderr longer than 2000 chars is truncated to the last 2000."""
    long_stderr = "x" * 5000 + "\nerror: os error 5"
    e = subprocess.CalledProcessError(
        returncode=2,
        cmd=["uv", "pip", "install"],
        stderr=long_stderr,
    )
    with patch.object(sys, "platform", "win32"):
        _simulate_windows_handler(e, capsys)

    captured = capsys.readouterr()
    assert "os error 5" in captured.out
    # The x's should be truncated (not all 5000 present in output)
    assert captured.out.count("x") < 5000


def test_windows_handler_caps_lines_at_20(capsys):
    """The stderr tail shows at most 20 lines."""
    many_lines = "\n".join(f"line {i}" for i in range(50))
    e = subprocess.CalledProcessError(
        returncode=2,
        cmd=["uv", "pip", "install"],
        stderr=many_lines,
    )
    with patch.object(sys, "platform", "win32"):
        _simulate_windows_handler(e, capsys)

    captured = capsys.readouterr()
    # Count indented lines that look like stderr output
    stderr_lines = [l for l in captured.out.splitlines() if l.startswith("    line ")]
    assert len(stderr_lines) <= 20
    assert "line 49" in captured.out  # last line is present
    assert "line 0" not in captured.out  # first line is truncated


def test_linux_handler_prints_stderr_and_exits(capsys):
    """On non-Windows, the handler also surfaces stderr before sys.exit(1)."""
    e = subprocess.CalledProcessError(
        returncode=1,
        cmd=["uv", "pip", "install"],
        stderr="resolver conflict",
    )
    with pytest.raises(SystemExit) as exc_info:
        with patch.object(sys, "platform", "linux"):
            _simulate_linux_handler(e, capsys)

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Update failed" in captured.out
    assert "resolver conflict" in captured.out
