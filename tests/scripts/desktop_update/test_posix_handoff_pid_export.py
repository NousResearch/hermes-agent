"""The update a POSIX Desktop hand-off spawns is told the marker's live owner as its hand-off pid.

``hermes update`` adopts an existing claim only when the live owner is its own pid, the
``HERMES_UPDATE_HANDOFF_PID`` value, or an ancestor (``update_lock.UpdateLock._is_partner``).
``posix.sh`` hands marker line 1 to a custodian before any update work starts, and the custodian is
a sibling of the update child, never an ancestor. On macOS the delegate line cannot rescue it (a
whole-second creation time never matches the child's own), so a hand-off pid naming the script
instead of the custodian made every Desktop-initiated update refuse its own claim with exit 2
(#133992, #134268, #134309, #134381).
"""
from __future__ import annotations

import pytest

from tests.scripts.desktop_update.test_desktop_update_posix_marker import FAKE_CLI, _custodian, _install, _run

pytestmark = pytest.mark.platforms("linux")  # /proc creation times, like the rest of the posix marker suite

RECORD_UPDATE_VIEW = """
import os, sys
from pathlib import Path
if sys.argv[1:2] == ['update']:
    marker = Path(os.environ['HERMES_HOME']) / '.hermes-update-in-progress'
    owner = marker.read_text(encoding='utf-8-sig').splitlines()[0]
    Path(os.environ['HANDOFF_VIEW']).write_text(
        f"{os.environ.get('HERMES_UPDATE_HANDOFF_PID', '')} {owner}", encoding='utf-8')
"""


def test_the_update_child_is_told_the_live_marker_owner(tmp_path):
    home, install = _install(tmp_path)
    (install / "hermes_cli" / "main.py").write_text(RECORD_UPDATE_VIEW + FAKE_CLI, encoding="utf-8")
    view = tmp_path / "view.txt"

    result = _run(tmp_path, home, install, HANDOFF_VIEW=str(view))

    assert result.returncode == 0, result.stdout + result.stderr
    custodian = _custodian(home)
    assert custodian, "the hand-off never named a custodian"
    handoff_pid, owner = view.read_text(encoding="utf-8-sig").split()
    assert owner == custodian, "line 1 must name the custodian while the update runs"
    assert handoff_pid == owner, "the update child must be told the marker's owner, not the hand-off script"
