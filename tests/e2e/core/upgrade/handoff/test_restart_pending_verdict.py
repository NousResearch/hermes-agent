"""The handoff cell's restart-pending property judges ``hermes update``'s own output (#155532): it passes
only when the update left the busy gateway running, reported its restart as pending, named the held
kanban worker's managed session, and listed the gateway as restart pending in the fleet check."""

from __future__ import annotations

from tests.e2e.core.upgrade.handoff import _handoff as X

SESSION = "local-9eec26b610ab80ce87c935fc27922cf0c530af240b9aa560dccf3fad701f0dc0"
LEFT = (f"  ↻ gateway PID 6: still running live work (managed run {SESSION} pid 440, running 2m00s) — left running; "
        "it exits once idle (start it again with `hermes gateway run` if nothing relaunches it). "
        "`hermes update --force` stops it now.")
FLEET = ("  ↻ default (pid 6) @ 611f0ab7 — restart pending (it restarts itself once its live work is done, "
         "or once this update exits when the update runs inside it)")


def _verdict(*lines: str, session: str = SESSION) -> str:
    return X.restart_pending_verdict("\n".join(lines) + "\n", 6, session)


def test_a_pending_restart_that_names_the_held_worker_passes():
    assert _verdict("Fleet version check:", LEFT, FLEET) == ""


def test_no_pending_restart_report_fails():
    assert "never reported its restart as pending" in _verdict("Fleet version check:", FLEET)


def test_a_report_without_the_held_worker_fails():
    bare = LEFT.replace(f" (managed run {SESSION} pid 440, running 2m00s)", "")
    assert f"not the held kanban worker's session {SESSION}" in _verdict(bare, FLEET)


def test_a_report_naming_another_session_fails():
    assert "not the held kanban worker's session local-other" in _verdict(LEFT, FLEET, session="local-other")


def test_a_gateway_missing_from_the_fleet_pending_rows_fails():
    assert "does not list gateway PID 6 as restart pending" in _verdict(LEFT, FLEET.replace("(pid 6)", "(pid 7)"))
