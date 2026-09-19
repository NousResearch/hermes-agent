"""Tests for #85840 — surfacing installer stderr and correct stage attribution.

Two bugs were reported:

1. **Misleading stage attribution**: the ``CalledProcessError`` handler in
   ``_cmd_update_impl`` printed ``"⚠ Git update failed"`` even when the git
   pull had already succeeded and the failure was in the Python-dependency
   install stage (``uv pip install -e .[all]``).

2. **Swallowed stderr**: ``_run_install_with_heartbeat`` called
   ``subprocess.run(check=True)`` without any capture, so the
   ``CalledProcessError`` carried no ``.stderr`` — the real failure cause
   (a locked ``.pyd``, a resolver conflict, a build error) was invisible in
   ``update.log``.

These tests exercise the REAL code paths (no re-implemented handler copies,
per review point 2 on this PR):

- ``hermes_cli.main._run_install_with_heartbeat`` captures stderr while
  leaving installer stdout streaming (``stdout=None``, ``stderr=PIPE``).
- ``hermes_cli.update_cmd._format_update_failure_stage`` classifies an
  install failure as a dependency failure, not a git failure.
- ``hermes_cli.update_cmd._print_called_process_error_tail`` surfaces the
  captured output tail the handler prints on failure.
"""

from __future__ import annotations

import subprocess

import pytest

from hermes_cli import main as cli_main
from hermes_cli import update_cmd


# ---------------------------------------------------------------------------#
# _run_install_with_heartbeat captures stderr, streams stdout
# ---------------------------------------------------------------------------#


def test_heartbeat_captures_stderr_on_failure(monkeypatch):
    """``_run_install_with_heartbeat`` must capture stderr so the resulting
    ``CalledProcessError`` carries the installer's error output."""
    fake_stderr = "error: failed to build cryptography\nos error 5"

    def fake_run(cmd, **kwargs):
        assert kwargs.get("stderr") == subprocess.PIPE, (
            "_run_install_with_heartbeat must pass stderr=subprocess.PIPE"
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


def test_heartbeat_leaves_installer_stdout_streaming(monkeypatch):
    """stdout must stay uncaptured (live ANSI progress bars) while stderr is
    a pipe. ``capture_output=True`` would buffer both streams and stall some
    installers on a full pipe."""

    def fake_run(cmd, **kwargs):
        assert kwargs.get("stdout") is None, (
            "installer stdout must stream through (stdout=None)"
        )
        assert kwargs.get("stderr") == subprocess.PIPE
        assert not kwargs.get("capture_output"), (
            "capture_output=True would buffer stdout too"
        )
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(cli_main.subprocess, "run", fake_run)

    # Should not raise.
    cli_main._run_install_with_heartbeat(
        ["uv", "pip", "install", "-e", ".[all]"],
        env=None,
        heartbeat_interval_seconds=300,  # don't let heartbeat fire
    )


# ---------------------------------------------------------------------------#
# Stage attribution (real classifier, no simulated handlers)
# ---------------------------------------------------------------------------#


def test_uv_install_failure_is_not_attributed_to_git():
    """The exact #85840 symptom: a failed ``uv pip install`` after a
    successful pull must not be reported as a git failure."""
    exc = subprocess.CalledProcessError(
        returncode=2,
        cmd=[r"C:\venv\Scripts\uv.exe", "pip", "install", "-e", ".[all]"],
        stderr="error: os error 5 on _rust.pyd",
    )
    stage = update_cmd._format_update_failure_stage(exc)
    assert stage == "Python dependency install failed"
    assert "Git" not in stage


def test_git_pull_failure_is_attributed_to_git():
    """A genuine git failure keeps the git attribution."""
    exc = subprocess.CalledProcessError(returncode=1, cmd=["git", "pull"])
    assert update_cmd._format_update_failure_stage(exc) == "Git update failed"


# ---------------------------------------------------------------------------#
# Output tail surfacing (the handler's real tail printer)
# ---------------------------------------------------------------------------#


def test_tail_printer_shows_stderr_lines(capsys):
    """The handler's tail printer shows the captured installer stderr."""
    exc = subprocess.CalledProcessError(
        returncode=2,
        cmd=["uv", "pip", "install"],
        stderr="line1\nline2\nerror: os error 5",
    )
    update_cmd._print_called_process_error_tail(exc)

    captured = capsys.readouterr()
    assert "Last output:" in captured.out
    assert "os error 5" in captured.out


def test_tail_printer_falls_back_to_stdout(capsys):
    """If only stdout was recorded, the tail printer uses it."""
    exc = subprocess.CalledProcessError(returncode=1, cmd=["git", "pull"])
    exc.stdout = "fatal: cannot lock ref"
    exc.stderr = None
    update_cmd._print_called_process_error_tail(exc)

    captured = capsys.readouterr()
    assert "cannot lock ref" in captured.out


def test_tail_printer_no_output_prints_nothing(capsys):
    """No captured output (the pre-fix #85840 state) must not crash nor emit
    an empty 'Last output:' block."""
    exc = subprocess.CalledProcessError(returncode=1, cmd=["git", "pull"])
    update_cmd._print_called_process_error_tail(exc)

    captured = capsys.readouterr()
    assert captured.out == ""


def test_tail_printer_caps_line_count(capsys):
    """The tail shows at most the last ``limit`` non-blank lines."""
    many_lines = "\n".join(f"line {i}" for i in range(50))
    exc = subprocess.CalledProcessError(
        returncode=2, cmd=["uv", "pip", "install"], stderr=many_lines
    )
    update_cmd._print_called_process_error_tail(exc)

    captured = capsys.readouterr()
    stderr_lines = [l for l in captured.out.splitlines() if l.startswith("    line ")]
    assert len(stderr_lines) <= 12
    assert "line 49" in captured.out  # last line survives
    assert "line 0" not in captured.out  # head truncated


# ---------------------------------------------------------------------------#
# End-to-end glue: failure raised by the heartbeat runner flows into the
# handler's real classification + tail printing
# ---------------------------------------------------------------------------#


def test_failed_install_surfaces_captured_stderr_through_handler(monkeypatch, capsys):
    """Simulate the update-flow contract: the install raises with captured
    stderr; the handler's real helpers attribute it correctly and print the
    cause — the two #85840 fixes composed together."""
    fake_stderr = "error: Failed to install requirements\n  os error 5"

    def fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=2, cmd=cmd, stderr=fake_stderr
        )

    monkeypatch.setattr(cli_main.subprocess, "run", fake_run)
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        cli_main._run_install_with_heartbeat(["uv", "pip", "install", "-e", "."])

    exc = exc_info.value
    stage = update_cmd._format_update_failure_stage(exc)
    assert stage != "Git update failed"

    update_cmd._print_called_process_error_tail(exc)
    captured = capsys.readouterr()
    assert "os error 5" in captured.out
