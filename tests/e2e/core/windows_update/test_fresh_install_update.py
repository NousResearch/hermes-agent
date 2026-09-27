"""Fresh source install on native Windows, then ``hermes update`` HEAD -> NEXT.

Failure class: the first things a new Windows user does. ``install.ps1 -NonInteractive``
on a clean profile (no git, no Python, no uv on PATH), the first agent launch, then the
first update and the turn after it. Verdicts come from what a user sees: exit codes,
which commit the checkout serves, what the launcher prints, and what reached the
loopback provider.
"""

from __future__ import annotations

import json

import pytest

from tests.e2e.core._pending_fixes import known_failure
from tests.e2e.core.windows_update._machine import (
    NEXT_MARKER,
    REQUIRES_OPT_IN,
    Journey,
    fail_with,
    failure_line,
    new_machine,
    one_shot_turn,
    source_completion_detour,
)
from tests.fakes.fake_llm_provider import FakeLLMServer

pytestmark = [pytest.mark.platforms("windows"), pytest.mark.integration,
              pytest.mark.live_system_guard_bypass, REQUIRES_OPT_IN]



def _installed_state(machine) -> dict:
    marker = machine.install_dir / ".hermes-bootstrap-complete"
    pinned = (json.loads(marker.read_text(encoding="utf-8-sig")).get("pinnedCommit")
              if marker.is_file() else None)
    return {"head": machine.installed_head(), "marker": marker.is_file(), "pinned": pinned}


@pytest.fixture(scope="module")
def journey(tmp_path_factory):
    with FakeLLMServer() as srv:
        machine = new_machine(tmp_path_factory.mktemp("fresh"), srv.base_url, label="fresh")
        j = Journey(machine)
        try:
            j.step("install", machine.install)
            # Snapshot what the installer left before the update moves the checkout on.
            j.step("installed", lambda: _installed_state(machine))
            j.step("version", lambda: machine.hermes("--version"))
            j.step("first_turn", lambda: one_shot_turn(machine, srv, "first-turn"))
            machine.advance()
            j.step("update", machine.update)
            j.step("turn_after_update", lambda: one_shot_turn(machine, srv, "turn-after-update"))
            yield j
        finally:
            machine.teardown()


def test_install_lands_on_head_and_publishes_hermes(journey: Journey) -> None:
    m, run = journey.machine, journey["install"]
    assert run.returncode == 0, fail_with(m, f"install.ps1 exited {run.returncode}", run)
    installed = journey["installed"]
    head = installed["head"]
    assert head == m.head, fail_with(m, f"install.ps1 checked out {head}, expected HEAD {m.head}", run)
    assert installed["marker"], fail_with(m, "install.ps1 reported success but wrote no bootstrap marker", run)
    pinned = installed["pinned"]
    assert pinned == m.head, fail_with(m, f"bootstrap marker pins {pinned}, expected {m.head}", run)
    version = journey["version"]
    assert version.returncode == 0 and "Hermes Agent v" in version.stdout, fail_with(
        m, "the published hermes.exe cannot report its version", version)


def test_first_agent_launch_runs_the_turn(journey: Journey) -> None:
    m, turn = journey.machine, journey["first_turn"]
    detour = source_completion_detour(turn.run)
    assert detour is None, fail_with(
            m, f"first agent launch after a pristine install detoured through source-update completion "
           f"(printed {detour!r})", turn.run)
    assert turn.ok, fail_with(
        m, f"first agent launch did not complete a turn (reply printed={turn.reply_id in turn.run.stdout}, "
           f"prompt reached provider={turn.reached_wire})", turn.run)


def test_update_moves_checkout_to_next(journey: Journey) -> None:
    m, run = journey.machine, journey["update"]
    with known_failure(r"^hermes update exited 1: ✗ Update failed: \[WinError 2\]",
                       "gated on #124634: on a machine without system Git, the pinned Git install.ps1 "
                       "staged never reaches `hermes update`'s PATH"):
        assert run.returncode == 0, fail_with(
            m, f"hermes update exited {run.returncode}: {failure_line(run)}", run)
    head = m.installed_head()
    assert head == m.next, fail_with(
        m, f"after hermes update the checkout is at {head}, expected NEXT {m.next}", run)
    assert (m.install_dir / NEXT_MARKER).is_file(), fail_with(
        m, "NEXT's marker file is missing from the checkout", run)
    receipt_path = m.hermes_home / "logs" / "update_receipts" / "latest.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.is_file() else {}
    assert receipt.get("outcome") == "success", fail_with(
        m, f"update receipt outcome is {receipt.get('outcome')!r}, expected 'success'", run)


def test_turn_after_update(journey: Journey) -> None:
    m, turn = journey.machine, journey["turn_after_update"]
    assert source_completion_detour(turn.run) is None, fail_with(
        m, "the first turn after a completed update re-ran source-update completion", turn.run)
    assert turn.ok, fail_with(
        m, f"turn after update failed (reply printed={turn.reply_id in turn.run.stdout}, "
           f"prompt reached provider={turn.reached_wire})", turn.run)
