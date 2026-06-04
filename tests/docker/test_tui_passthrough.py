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

import re
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
    # Emit the probe result behind a unique marker. The container's s6 boot
    # output (cont-init diagnostics, skills-sync summaries like
    # "Done: 90 new, 0 updated, ...", the preinit "uid=0 ... egid=0" line)
    # is written to the SAME PTY stream before this runs, so we must NOT
    # scan the whole stream for "the first number" — that picks up a stray
    # 0 from the boot log and flips the assertion (assert 0 > 0) whenever
    # boot output shifts (e.g. a new bundled dep changes the skills-sync
    # counts). Parse only the value tagged with our marker.
    marker = "HERMES_TTY_COLS"
    probe = (
        f'if [ -t 1 ]; then echo "{marker}=$(tput cols)"; else echo "{marker}=NO_TTY"; fi'
    )
    cmd = (
        f"docker run --rm -t -e COLUMNS=123 {built_image} "
        f"sh -c {shlex.quote(probe)}"
    )
    r = subprocess.run(
        ["script", "-qc", cmd, "/dev/null"],
        capture_output=True, text=True, timeout=120,
    )
    output = r.stdout
    matches = re.findall(rf"{marker}=(\S+)", output)
    assert matches, f"No {marker} marker in output: {output!r}"
    value = matches[-1].strip()
    assert value != "NO_TTY", f"TTY passthrough failed: {output!r}"
    assert value.isdigit(), f"Non-numeric column width {value!r} in: {output!r}"
    assert int(value) > 0


def test_tui_flag_recognized(built_image: str) -> None:
    """``docker run -it <image> --help`` should run without crashing."""
    cmd = f"docker run --rm -t {built_image} --help"
    r = subprocess.run(
        ["script", "-qc", cmd, "/dev/null"],
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0
