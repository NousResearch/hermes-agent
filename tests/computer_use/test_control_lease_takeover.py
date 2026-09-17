"""Takeover and hand-back through the REAL lease/control-epoch abstraction (#112734).

The lease module is vendored from open PR #108914 (``tools/bot_desktop/lease.py``) instead of
waiting for it to merge. These tests exercise the real acquire/release/epoch machinery — a
stale viewer's release is rejected, a valid takeover transfers control with an epoch bump, and
the computer_use revision facts track the lease epoch so a mid-flight takeover invalidates the
admitted revision.
"""
import pytest

from tools.bot_desktop import lease
from tools.bot_desktop.lease import AGENT, HUMAN, HumanHasControl
from tools.computer_use import tool as tool_mod
from tools.computer_use.execution_revision import CONTROL


@pytest.fixture
def lease_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME per test; the lease file never touches a real profile."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    lease._reset_for_tests()
    yield home
    lease._reset_for_tests()


def epoch_of(session: str) -> int:
    return tool_mod._current_revision_facts(session)["control_epoch"]


def test_takeover_transfers_control_and_bumps_epoch(lease_home):
    first = lease.get()
    assert (first.holder, first.epoch) == (AGENT, 0)
    taken = lease.acquire("viewer-1", reason="2FA login")
    assert taken.holder == HUMAN and taken.viewer_id == "viewer-1"
    assert taken.epoch == first.epoch + 1
    assert epoch_of("sid-takeover") == taken.epoch  # computer_use facts read the real lease


def test_second_viewer_takeover_evicts_first(lease_home):
    first = lease.acquire("viewer-a")
    second = lease.acquire("viewer-b")  # last writer wins: eviction, not rejection
    assert second.holder == HUMAN and second.viewer_id == "viewer-b"
    assert second.epoch == first.epoch + 1


def test_stale_viewer_release_is_rejected(lease_home):
    lease.acquire("viewer-a")
    second = lease.acquire("viewer-b")
    after = lease.release(viewer_id="viewer-a")  # stale holder tries to hand back
    assert after.holder == HUMAN and after.viewer_id == "viewer-b"  # ignored, not an error
    assert after.epoch == second.epoch  # no transition, no epoch bump


def test_handback_by_current_holder_returns_control(lease_home):
    taken = lease.acquire("viewer-1")
    handed = lease.release(viewer_id="viewer-1")
    assert handed.holder == AGENT and handed.viewer_id is None
    assert handed.epoch == taken.epoch + 1
    assert epoch_of("sid-handback") == handed.epoch


def test_bare_release_noop_while_human_holds(lease_home):
    taken = lease.acquire("viewer-1")
    same = lease.release(unless_human=True)
    assert same.holder == HUMAN and same.viewer_id == "viewer-1"
    assert same.epoch == taken.epoch


def test_redundant_handback_does_not_bump_epoch(lease_home):
    first = lease.get()
    again = lease.release(viewer_id="viewer-1")  # e.g. a double-clicked Hand back
    assert again.holder == AGENT and again.epoch == first.epoch


def test_same_viewer_reacquire_is_noop(lease_home):
    taken = lease.acquire("viewer-1", reason="login")
    same = lease.acquire("viewer-1", reason="changed reason")
    assert same.epoch == taken.epoch and same.reason == "login"


def test_agent_refused_while_human_holds(lease_home):
    lease.acquire("viewer-1")
    with pytest.raises(HumanHasControl):
        lease.assert_agent_may_act()


def test_midflight_takeover_invalidates_control_revision(lease_home):
    state = tool_mod._execution_state("sid-midflight")
    admitted = state.admit(app="xterm")
    assert state.validate(admitted, requires={CONTROL}).ok
    lease.acquire("viewer-1")  # human takes over under the admitted action
    verdict = state.validate(admitted, requires={CONTROL})
    assert not verdict.ok and verdict.reason == "control_epoch_changed"


def test_revision_epoch_tracks_every_transition(lease_home):
    epochs = [epoch_of("sid-track")]
    lease.acquire("viewer-1")
    epochs.append(epoch_of("sid-track"))
    lease.release(viewer_id="viewer-1")
    epochs.append(epoch_of("sid-track"))
    assert epochs[0] < epochs[1] < epochs[2]


def test_corrupt_lease_fails_closed(lease_home):
    path = lease_home / "bot-desktop" / "lease.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert lease.get().holder == HUMAN  # torn write must never let the agent act
    with pytest.raises(HumanHasControl):
        lease.assert_agent_may_act()
