"""Careful automatic moves for exactly two voters, across simulated gateways on a virtual clock."""

import pytest

from gateway import hosted_room_succession as succession
from gateway import hosted_room_succession_move as move
from tests.gateway.fixtures.succession import ROOM, events
from tests.gateway.fixtures.succession_world import Network, World

SILENCE = succession.CAREFUL_SILENCE_SECONDS


@pytest.fixture
def world(tmp_path, monkeypatch):
    value = World(tmp_path, monkeypatch, ("h", "s", "p"), voters=("h", "s"), others=("p",))
    yield value
    value.close()


def moved_by(world, name):
    return next(event["payload"] for event in reversed(events(world.gateways[name], "hosted_room_events"))
                if event["kind"] == "authority.transition")


def test_two_voters_move_carefully_after_three_minutes_of_silence(world):
    world.advance(10)
    current = world.status("h")
    assert current["automatic"]["mode"] == "careful" and current["automatic"]["state"] == "ready"
    world.network.stopped.add("h")
    took = world.until(lambda: world.head("s")["authoritative"], limit=SILENCE + 60, observe=False)
    assert SILENCE - 10 <= took <= SILENCE + 15
    payload = moved_by(world, "s")
    assert payload["proof_kind"] == "evidence" and payload["reason"] == "automatic"
    assert payload["proof"]["statement"]["silent_for_s"] >= SILENCE
    # The owner is offered going back while the old host isn't reconciled yet.
    current = world.status("s")
    assert current["moved_in"]["proof_kind"] == "evidence"
    assert {"action": "keep", "targets": [world.gateways["h"].install_id]} in current["actions"]
    # A host that returns without having written anything since becomes a copy, silently.
    world.network.stopped.discard("h")
    world.until(lambda: not world.head("h")["authoritative"], limit=30, observe=False)
    assert world.status("h")["state"] == "moved_away"
    world.advance(20, observe=False)
    assert world.status("s")["moved_in"] is None


def test_a_cut_link_with_both_online_runs_on_two_until_the_owner_keeps_one(world):
    world.advance(10)
    world.network.split({"h"}, {"s"})  # the laptop still reaches both, but nothing bridges yet
    world.network.stopped.add("p")
    world.until(lambda: world.head("s")["authoritative"], limit=SILENCE + 60, observe=False)
    assert world.head("h")["authoritative"] and world.send("h", "user:on-h")  # online, so not isolated
    assert world.send("s", "user:on-s")
    world.network.heal()
    world.until(lambda: world.status("h")["state"] == "continued_on_two", limit=30, observe=False)
    world.until(lambda: world.status("s")["state"] == "continued_on_two", limit=30, observe=False)
    conflict = world.status("s")["conflict"]
    assert {host["name"] for host in conflict["hosts"]} == {"H", "S"} and conflict["start"] is not None
    # The group keeps running on the move (the higher epoch); the old host stops and keeps its messages,
    # and as a participant its Bots take the new host's work meanwhile.
    h, s = world.gateways["h"], world.gateways["s"]
    assert conflict["running_on"]["install_id"] == s.install_id
    assert world.head("s")["serving"] and not world.head("h")["serving"] and "user:on-h" in world.texts("h")
    from gateway import hosted_room_fence as fence
    from gateway.platforms.api_server_run_scope import room_run_scope_key
    assert fence.room_fence_state(h.runs.path, ROOM)["authority"] == {"epoch": 2, "install_id": s.install_id}
    work = {"room_id": ROOM, "home_install_id": s.install_id, "authority_gateway_id": s.install_id,
            "authority_epoch": 2, "member_id": "writer", "target_install_id": h.install_id, "target_profile": "default"}
    assert h.runs.reserve(room_run_scope_key(work), "room:task-s:1", "fp", "run-s", {"status": "queued"},
                          identity=work)[0] == "created"
    world.advance(30, observe=False)
    assert world.status("h")["state"] == world.status("s")["state"] == "continued_on_two"  # no timeout
    # The owner keeps the old host: it continues at a fresh epoch, and the other one's messages stay apart.
    with world.gateways["h"].acting():
        kept = move.keep(world.context(world.gateways["h"]), ROOM, world.gateways["h"].install_id)
    assert kept["state"] == "ok"
    world.until(lambda: not world.head("s")["authoritative"], limit=60, observe=False)
    assert "user:on-h" in world.texts("h") and "user:on-s" not in world.texts("h")


def test_an_isolated_host_pauses_itself_before_its_standby_moves(world):
    world.advance(10)
    world.network.isolate("h")
    world.network.offline.add("h")  # its own online check fails
    world.until(lambda: world.status("h")["state"] == "paused", limit=SILENCE, observe=False)
    assert world.status("h")["paused"]["reason"] == "isolated"
    assert not world.head("s")["authoritative"]
    assert not world.send("h", "user:isolated")
    world.until(lambda: world.head("s")["authoritative"], limit=SILENCE, observe=False)
    world.network.heal()
    world.network.offline.discard("h")
    world.until(lambda: not world.head("h")["authoritative"], limit=30, observe=False)
    assert world.status("h")["moved"]["separate_events"] == 0


def test_an_isolated_standby_never_moves(world):
    world.advance(10)
    world.network.isolate("s")
    world.network.offline.add("s")
    world.advance(SILENCE + 60, observe=False)
    assert not world.head("s")["authoritative"] and world.head("h")["authoritative"]
    assert world.status("h")["state"] == "ok"  # the host is online: it keeps serving


def test_no_careful_move_while_the_host_announced_a_restart(world):
    from gateway.hosted_room_succession_move import append_state
    world.advance(10)
    host = world.gateways["h"]
    with host.acting():
        append_state(host.db, ROOM, "host_restarting", until=world.wall0 + world.t + 240)
    world.advance(5)
    world.network.stopped.add("h")
    world.advance(240 + 60, observe=False)
    assert not world.head("s")["authoritative"]
    world.until(lambda: world.head("s")["authoritative"], limit=SILENCE + 120, observe=False)


def test_three_voters_never_move_carefully(tmp_path, monkeypatch):
    three = World(tmp_path, monkeypatch, ("h", "s", "t"), voters=("h", "s", "t"))
    try:
        three.advance(10)
        three.network.stopped.add("t")
        three.network.split({"h"}, {"s"})  # s alone: it can never gather a majority
        three.advance(SILENCE + 60, observe=False)
        assert not three.head("s")["authoritative"]
    finally:
        three.close()


def test_any_device_can_hand_the_old_host_the_move_and_writing_on_both_asks_the_owner(world):
    from contextlib import closing
    from gateway import hosted_room_succession_return as returning
    from gateway import hosted_rooms as rooms
    world.advance(10)
    world.network.split({"h"}, {"s"})
    world.network.stopped.add("p")
    world.until(lambda: world.head("s")["authoritative"], limit=SILENCE + 60, observe=False)
    assert world.send("h", "user:on-h")
    s = world.gateways["s"]
    with s.acting(), closing(rooms._read_connection(s.db)) as conn:
        chain = succession.chain_after_locked(conn, ROOM, 1)
    h = world.gateways["h"]
    with h.acting():
        forged = [dict(event) for event in chain]
        index = next(i for i, event in enumerate(forged) if event["kind"] == "authority.transition")
        forged[index] = {**forged[index], "payload": {**forged[index]["payload"], "successor_gateway_id": h.install_id}}
        with pytest.raises(succession.SuccessionError):
            returning.learn(world.context(h), ROOM, forged)
        # As groups.log returns them, a client may leave the epochs out; they are derived.
        bare = [{key: value for key, value in event.items() if key != "authority_epoch"} for event in chain]
        learned = returning.learn(world.context(h), ROOM, bare)
    assert learned["learned"] and learned["state"] == "continued_on_two"
    assert world.status("h")["state"] == "continued_on_two" and world.head("h")["authoritative"]


def test_an_isolated_host_stays_paused_through_a_restart_until_it_hears_its_standby(world):
    from gateway.hosted_room_succession_automatic import Automatic
    world.advance(10)
    world.network.isolate("h")
    world.network.offline.add("h")
    world.until(lambda: world.status("h")["state"] == "paused", limit=SILENCE, observe=False)
    # The host's gateway restarts: a fresh upkeep, still cut off.
    host = world.gateways["h"]
    from gateway.hosted_room_succession_automatic import install, uninstall
    with host.acting():
        uninstall(world.automatics["h"])
        fresh = Automatic(lambda: world.context(host), online_check=lambda: "h" not in world.network.offline)
        install(fresh)
    world.upkeeps["h"].automatic = world.automatics["h"] = fresh
    world.upkeeps["h"]._restored = False
    world.advance(5, observe=False)
    assert world.status("h")["state"] == "paused" and world.status("h")["paused"]["reason"] == "isolated"
    world.network.heal()
    world.network.offline.discard("h")
    world.until(lambda: world.status("h")["state"] == "ok", limit=30, observe=False)


@pytest.mark.parametrize(("address", "public"), [
    ("127.0.0.1", False), ("::1", False), ("10.0.0.5", False), ("192.168.1.2", False), ("172.16.3.4", False),
    ("169.254.1.1", False), ("fe80::1%en0", False), ("fd00::1", False), ("100.101.102.103", False),
    ("::ffff:192.168.1.2", False), ("8.8.8.8", True), ("2606:4700:4700::1111", True), ("not-an-ip", False)])
def test_only_public_addresses_count_for_the_online_check(address, public):
    from gateway.hosted_room_succession_automatic import public_address
    assert public_address(address) is public


def test_a_local_model_server_never_makes_a_cut_off_computer_look_online(monkeypatch):
    from gateway import hosted_room_succession_automatic as automatic
    tried = []

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def connect(address, timeout=None):
        tried.append(address)
        return Connection()

    def resolving(addresses):
        return lambda host, port, type=None: [(None, None, None, "", (addresses[host], port))]

    monkeypatch.setattr(automatic.socket, "create_connection", connect)
    # Ollama on this computer and on the LAN: nothing public is left, so the check fails untried.
    monkeypatch.setattr(automatic, "_known_endpoints", lambda: [("localhost", 11434), ("ollama.lan", 11434)])
    monkeypatch.setattr(automatic.socket, "getaddrinfo", resolving({"localhost": "127.0.0.1",
                                                                     "ollama.lan": "192.168.1.20"}))
    assert automatic.online_check_default() is False and tried == []
    # A public endpoint it already uses counts.
    monkeypatch.setattr(automatic, "_known_endpoints", lambda: [("localhost", 11434), ("api.example.com", 443)])
    monkeypatch.setattr(automatic.socket, "getaddrinfo", resolving({"localhost": "127.0.0.1",
                                                                     "api.example.com": "93.184.216.34"}))
    assert automatic.online_check_default() is True and tried == [("93.184.216.34", 443)]



def test_going_back_while_the_old_host_is_down_continues_there_once_it_returns(world):
    world.advance(10)
    world.network.stopped.add("h")
    world.until(lambda: world.head("s")["authoritative"], limit=SILENCE + 60, observe=False)
    s, h = world.gateways["s"], world.gateways["h"]
    assert {"action": "keep", "targets": [h.install_id]} in world.status("s")["actions"]  # "Go back to H"
    with s.acting():
        move.keep(world.context(s), ROOM, h.install_id)
    world.advance(5, observe=False)
    assert world.admitting() == set()  # the new host stepped aside; the old one is still down
    moved_to = world.head("s")["authority_epoch"]
    world.network.stopped.discard("h")
    # The old host learns of the move, honours the owner's choice it is handed and continues at a fresh
    # epoch past it; the computer it was chosen over steps down to follow it.
    world.until(lambda: world.head("h")["serving"] and world.head("h")["authority_epoch"] > moved_to,
                limit=240, observe=True)
    epoch = world.head("h")["authority_epoch"]
    world.until(lambda: world.head("s")["authority_gateway_id"] == h.install_id, limit=60, observe=True)
    assert world.admitting() == {("h", epoch)}
    # And it stays that way: the computer it was chosen over never takes the group back by itself.
    world.advance(SILENCE + 60, observe=True)
    assert world.admitting() == {("h", epoch)} and world.head("s")["authority_gateway_id"] == h.install_id


@pytest.fixture
def pair(tmp_path, monkeypatch):
    value = World(tmp_path, monkeypatch, ("h", "s"), voters=("h", "s"))
    value.network = OneWay(value.network.names)
    yield value
    value.close()


class OneWay(Network):
    """A network whose links can also fail in one direction only."""

    def __init__(self, names):
        super().__init__(names)
        self.oneway: set[tuple[str, str]] = set()

    def reachable(self, a, b):
        return super().reachable(a, b) and (a, b) not in self.oneway


@pytest.mark.parametrize("gap", [0, 1, 2, 3])
def test_going_back_works_however_soon_the_old_host_returns(pair, gap):
    """The computer the owner went back from starts following the old host as it steps aside: its silence
    counts from then, so it never takes the group back by itself while the old host continues."""
    world = pair
    world.advance(10)
    world.network.stopped.add("h")
    world.until(lambda: world.head("s")["authoritative"], limit=SILENCE + 60, observe=False)
    s, h = world.gateways["s"], world.gateways["h"]
    moved_to = world.head("s")["authority_epoch"]
    with s.acting():
        move.keep(world.context(s), ROOM, h.install_id)
    if gap:
        world.advance(gap, observe=True)
    world.network.stopped.discard("h")
    world.until(lambda: world.head("h")["serving"] and world.head("h")["authority_epoch"] > moved_to,
                limit=120, observe=True)
    epoch = world.head("h")["authority_epoch"]
    world.until(lambda: world.head("s")["authority_gateway_id"] == h.install_id, limit=60, observe=True)
    world.advance(SILENCE + 120, observe=True)
    assert world.admitting() == {("h", epoch)}
    assert world.status("h")["state"] == "ok" and world.status("s")["state"] == "moved_away"


def test_a_host_paused_for_a_step_nobody_took_continues_past_it_by_rule(pair):
    """The standby fenced the host's epoch for itself and then gave up. The host pauses for that promise;
    once the standby has left it untaken long enough, answering and hosting nothing, the host continues at
    a fresh step with the standby's own promise of it."""
    from gateway import hosted_room_fence as fence
    from gateway.hosted_room_succession_automatic import LEASE_SECONDS
    world = pair
    world.advance(10)
    h, s = world.gateways["h"], world.gateways["s"]
    epoch = world.head("h")["authority_epoch"]
    world.network.oneway.add(("h", "s"))  # the host's pushes stall, so the standby's lease to it runs out
    world.advance(LEASE_SECONDS + 2, observe=True)
    with s.acting():
        fence.fence_and_promise(s.runs.path, room_id=ROOM, fence_epoch=epoch, promise_epoch=epoch + 1,
                                candidate_install_id=s.install_id)
    world.network.oneway.clear()
    world.until(lambda: not world.head("h")["serving"], limit=90, observe=True)
    assert world.status("h")["state"] == "moving"
    world.until(lambda: world.head("h")["serving"] and world.head("h")["authority_epoch"] > epoch + 1,
                limit=move.STALLED_PROMISE_SECONDS + 120, observe=True)
    fresh = world.head("h")["authority_epoch"]
    moved = moved_by(world, "h")
    assert moved["reason"] == "automatic" and moved["proof"]["statement"] == succession.RECOVER_TEXT
    assert moved["proof"]["stalled"] == {"install_id": s.install_id, "epoch": epoch + 1}
    assert s.install_id in {receipt["custodian_install_id"] for receipt in moved["proof"]["receipts"]}
    world.until(lambda: (world.head("s")["authority_gateway_id"], world.head("s")["authority_epoch"])
                == (h.install_id, fresh), limit=60, observe=True)
    world.advance(60, observe=True)
    assert world.admitting() == {("h", fresh)}
