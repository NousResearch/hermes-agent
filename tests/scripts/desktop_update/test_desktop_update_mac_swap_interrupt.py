"""An interrupted macOS app swap always comes back as one complete bundle.

The Desktop hand-off (``scripts/desktop-update/posix.sh::mac_swap``) replaces
the installed ``Hermes.app`` by staging a ditto copy, renaming the old bundle
aside and renaming the copy in. A hand-off killed anywhere in that sequence is
finished or rolled back by ``mac_bundle_recover`` on the next run.

``posix.sh --self-test-swap-interrupt`` runs the REAL ``mac_swap`` on a fixture
bundle tree, SIGKILLs its process group at every step boundary and mid-copy /
mid-cleanup, then runs the REAL ``mac_bundle_recover`` and checks the
invariant: the target is the complete old or the complete new bundle, no
aside/staged copy is left, and a second recovery is a no-op. ``posix`` because
the swap and recovery are mv/rm/ditto; on Linux the self-test stages with
``cp -R`` (ditto is macOS-only), and the macOS lane of tests-os.yml runs the
same arm with the real ``/usr/bin/ditto``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
POSIX_SH = REPO_ROOT / "scripts" / "desktop-update" / "posix.sh"

# Every interruption point, in swap order (swap-selftest.sh header).
CELLS = (
    "before-stage",
    "mid-stage",
    "staged",
    "aside",
    "installed",
    "mid-cleanup",
    "complete",
)


@pytest.mark.platforms("posix")
def test_killed_app_swap_recovers_to_a_complete_bundle(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    proc = subprocess.run(
        ["/bin/bash", str(POSIX_SH), "--self-test-swap-interrupt"],
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "HOME": str(home),
            "HERMES_HOME": str(home),
            "TMPDIR": str(tmp_path),
        },
        capture_output=True,
        text=True,
        timeout=240,
    )
    report = proc.stdout + proc.stderr
    assert proc.returncode == 0, report
    for cell in CELLS:
        assert f"ok   [{cell}]" in proc.stdout, f"cell {cell} did not pass:\n{report}"
