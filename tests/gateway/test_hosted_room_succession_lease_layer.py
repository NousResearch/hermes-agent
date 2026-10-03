"""The lease layer's own health: a host whose lease layer isn't running pauses to stay safe and says why,
its owner can still continue it, and the upkeep keeps trying to install the layer."""

import pytest

from gateway import hosted_room_custody as custody
from gateway import hosted_room_succession as succession
from gateway import hosted_room_succession_automatic as automatic
from tests.gateway.fixtures.succession import ROOM
from tests.gateway.fixtures.succession_world import World

SILENCE = succession.CAREFUL_SILENCE_SECONDS


@pytest.fixture
def world(tmp_path, monkeypatch):
    value = World(tmp_path, monkeypatch, ("h", "s", "t"), voters=("h", "s", "t"))
    yield value
    value.close()
    automatic._without_leases.clear()


def test_a_host_without_its_lease_layer_pauses_says_why_and_can_be_continued_anyway(world):
    world.advance(10)
    host = world.gateways["h"]
    assert world.admitting() == {("h", 1)}
    # The lease layer is gone and doesn't come back (an install that keeps failing).
    with host.acting():
        automatic.uninstall(world.automatics["h"])
    world.upkeeps["h"]._installed = True
    with host.acting():
        assert succession.paused_reason(host.db, ROOM) == "room_host_paused"
    current = world.status("h")
    assert current["state"] == "paused" and current["paused"]["reason"] == "no_lease_layer"
    assert {"action": "continue_anyway", "turns_off_automatic": True} in current["actions"]
    assert not world.send("h", "user:paused")
    with host.acting():
        assert not custody.serving(ROOM, mode=custody.room_mode(host.db, ROOM))  # nothing may start here
    # Its owner continues it anyway, at its own epoch: that also turns automatic moves off for the group.
    with host.acting():
        automatic.continue_anyway(world.context(host), ROOM)
        assert succession.paused_reason(host.db, ROOM) is None
        assert custody.room_mode(host.db, ROOM) == "ask"
        assert custody.serving(ROOM, mode="ask")  # its turns start again
    assert world.send("h", "user:anyway") and world.head("h")["authority_epoch"] == 1
    assert world.status("h")["automatic"]["state"] == "off"
    # Every standby learns it with the next push, so none moves the group by itself beside a host that
    # holds no lease, even once the host goes quiet.
    world.advance(10)
    assert world.configuration("s")["automatic"] is False and world.configuration("t")["automatic"] is False
    world.network.stopped.add("h")
    world.advance(SILENCE + 60, observe=False)
    assert not world.head("s")["authoritative"] and not world.head("t")["authoritative"]


def test_the_upkeep_keeps_trying_to_install_the_lease_layer(tmp_path):
    from gateway.hosted_room_succession_status import SuccessionUpkeep
    contexts = [None]
    upkeep = SuccessionUpkeep(lambda: None, lease_context=lambda: contexts[-1])
    try:
        assert upkeep._ensure_installed() is False  # no room store yet: nothing to hold
        contexts.append(type("Context", (), {"db_path": tmp_path / "state.db", "runs_store": None})())
        assert upkeep._ensure_installed() is True
        assert automatic.instance_for(tmp_path / "state.db") is upkeep.automatic
    finally:
        upkeep.stop()
