"""marker.sh reads macOS creation times with BSD date even under GNU coreutils (#137925).

Desktop hands the update off with the user's login-shell PATH. With Homebrew's
coreutils `gnubin` first on it, a bare `date` is GNU date, which has no `-j`:
`proc_ct` printed nothing, the bridge-marker identity check refused the claim and
`posix.sh` exited 2 on every update. A `date` that rejects `-j` sits first on
PATH here; the creation time must still match the one read with the system PATH.
"""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess

import pytest

MARKER_SH = Path(__file__).resolve().parents[3] / "scripts" / "desktop-update" / "marker.sh"


def _proc_ct(path: str) -> str:
    script = f"log() {{ :; }}; . {shlex.quote(str(MARKER_SH))}; proc_ct {os.getpid()}"
    proc = subprocess.run(
        ["/bin/bash", "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env={**os.environ, "PATH": path},
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


@pytest.mark.platforms("macos")
def test_proc_ct_ignores_a_gnu_date_first_on_path(tmp_path):
    gnubin = tmp_path / "gnubin"
    gnubin.mkdir()
    fake = gnubin / "date"
    fake.write_text(
        '#!/bin/sh\nfor a in "$@"; do [ "$a" = -j ] && { echo "date: invalid option -- \'j\'" >&2; exit 1; }; done\n'
        'exec /bin/date "$@"\n',
        encoding="utf-8",
    )
    fake.chmod(0o755)

    system_path = "/usr/bin:/bin:/usr/sbin:/sbin"
    baseline = _proc_ct(system_path)
    assert baseline, "proc_ct must read a creation time with the system PATH"
    assert _proc_ct(f"{gnubin}:{system_path}") == baseline
