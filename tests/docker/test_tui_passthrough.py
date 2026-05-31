"""Harness: interactive TUI TTY passthrough.

Uses ``script -qc`` on the host to allocate a PTY for the docker client,
which then allocates a container-side PTY via ``-t``. The probe inside
the container is ``tput cols``, which returns a real column count when
stdout is a TTY and either prints ``80`` (the terminfo fallback) or
nothing when it is not.

These tests MUST pass on the current tini-based image AND continue to
pass after the Phase 2 s6 migration. Any drift is a regression.
"""
from __future__ import annotations

import shlex
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("script") is None,
    reason="`script` command not available on this host",
)


def test_tty_passthrough_to_container(built_image: str) -> None:
    """``docker run -t`` must deliver a real TTY to the container process."""
    probe = "if [ -t 1 ]; then tput cols; else echo NO_TTY; fi"
    cmd = (
        f"docker run --rm -t -e COLUMNS=123 {built_image} "
        f"sh -c {shlex.quote(probe)}"
    )
    r = subprocess.run(
        ["script", "-qc", cmd, "/dev/null"],
        capture_output=True, text=True, timeout=120,
    )
    output = r.stdout.strip()
    assert "NO_TTY" not in output, f"TTY passthrough failed: {output!r}"
    numeric_lines = [s for s in output.split() if s.strip().isdigit()]
    assert numeric_lines, f"No numeric width in output: {output!r}"
    assert int(numeric_lines[0]) > 0


def test_tui_flag_recognized(built_image: str) -> None:
    """``docker run -it <image> --help`` should run without crashing."""
    cmd = f"docker run --rm -t {built_image} --help"
    r = subprocess.run(
        ["script", "-qc", cmd, "/dev/null"],
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0
