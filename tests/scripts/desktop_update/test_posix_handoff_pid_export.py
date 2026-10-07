"""posix.sh must export the marker's OWNER as HERMES_UPDATE_HANDOFF_PID, not its own pid.

`hermes update` adopts an existing claim only when the live owner is its own pid, the
``HERMES_UPDATE_HANDOFF_PID`` value, or one of its ancestors
(``update_lock.UpdateLock._is_partner``). The posix hand-off hands marker line 1 to a
*custodian* subshell (``marker_refresher_start``) that outlives the hand-off script, so the
owner is a **sibling** of that script: never its own pid, and never an ancestor of the
``hermes update`` it spawns. Exporting ``$$`` therefore left the custodian unrecognisable and
every desktop-initiated update refused its own claim with exit code 2 --
``Another Hermes update is already running (started Ns ago, process <custodian>)``.

The runtime hand-off protocol suite (``test_desktop_update_posix_handoff_protocol.py``) is
``platforms("linux")`` because it decides liveness through ``/proc``; these assertions pin the
same invariant on every platform.
"""
from __future__ import annotations

from pathlib import Path

import pytest

POSIX = Path(__file__).resolve().parents[3] / "scripts" / "desktop-update" / "posix.sh"
EXPORT = "export HERMES_UPDATE_HANDOFF_PID="


@pytest.fixture(scope="module")
def body() -> str:
    return POSIX.read_text(encoding="utf-8-sig")


def test_handoff_pid_names_the_marker_owner_not_the_script(body):
    """Line 1 of the marker carries $MY_PID, so that is what the update child must be told."""
    assert f'{EXPORT}"$MY_PID"' in body
    assert f'{EXPORT}"$$"' not in body


def test_custodian_handover_precedes_the_export(body):
    """$MY_PID only becomes the custodian when marker_refresher_start runs; an export placed
    before that call site would still name the hand-off script itself."""
    call_site = body.index("\nmarker_refresher_start\n")  # the call, not the definition
    assert call_site < body.index(EXPORT)


def test_the_owner_the_marker_names_is_what_my_pid_is_set_to(body):
    """marker.sh writes line 1 from "$MY_PID" (marker_custody_take_locked / marker_canonical),
    and the handover is what re-points it at the custodian."""
    assert 'MY_PID="$MARKER_REFRESHER"' in body
