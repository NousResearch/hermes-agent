"""Automatic moves in majority mode across simulated gateways on a virtual clock.

A global observer checks after every simulated second that at most one epoch admits work.
"""

import pytest

from gateway import hosted_room_fence as fence
from gateway import hosted_room_succession as succession
from tests.gateway.fixtures.succession import ROOM, events
from tests.gateway.fixtures.succession_world import World


@pytest.fixture
def world(tmp_path, monkeypatch):
    value = World(tmp_path, monkeypatch, ("h", "s", "t", "p"), voters=("h", "s", "t"), others=("p",))
    yield value
    value.close()


def transition(world, name):
    return next(event for event in reversed(events(world.gateways[name], "hosted_room_events"))
                if event["kind"] == "authority.transition")


def test_the_group_is_ready_and_its_host_holds_a_majority_lease(world):
    # The host first, then the always-on successors in the owner's order; the laptop doesn't vote.
    assert world.configuration("h")["voters"] == [world.gateways[n].install_id for n in ("h", "s", "t")]
    world.advance(10)
    assert world.admitting() == {("h", 1)}
    current = world.status("h")
    assert current["state"] == "ok" and current["automatic"]["mode"] == "majority"
    assert current["automatic"]["state"] == "ready" and current["automatic"]["standby"]["name"] == "S"
    assert {action["action"] for action in current["actions"]} >= {"move", "automatic"}


def test_a_killed_host_is_replaced_within_a_minute_and_never_by_two(world):
    world.advance(10)
    assert world.send("h", "user:before")
    world.advance(5)  # copied to the voters before the host dies
    world.network.stopped.add("h")
    took = world.until(lambda: world.head("s")["authoritative"], limit=60)
    assert took <= 45
    moved = transition(world, "s")
    assert moved["payload"]["proof_kind"] == "certified" and moved["payload"]["reason"] == "automatic"
    assert moved["payload"]["at_risk"] == 0 and moved["payload"]["to_epoch"] == 2
    world.advance(10)
    assert world.admitting() == {("s", 2)}
    assert world.head("t")["authority_gateway_id"] == world.gateways["s"].install_id
    assert "user:before" in world.texts("s")
    assert world.status("s")["moved_in"]["proof_kind"] == "certified"
    # The old host comes back as a copy that follows the new one, with nothing set aside.
    world.network.stopped.discard("h")
    world.until(lambda: not world.head("h")["authoritative"], limit=30)
    assert world.head("h")["authority_gateway_id"] == world.gateways["s"].install_id
    assert world.status("h")["moved"]["separate_events"] == 0


def test_a_host_cut_off_from_its_voters_pauses_before_a_standby_admits(world):
    world.advance(10)
    world.network.split({"h"}, {"s", "t", "p"})
    world.until(lambda: world.status("h")["state"] == "paused", limit=25)
    paused = world.status("h")
    assert paused["paused"]["reason"] == "lost_majority"
    assert {item["name"] for item in paused["paused"]["waiting_for"]} == {"S", "T"}
    assert {"action": "continue_anyway"} in paused["actions"]
    assert not world.send("h", "user:cut-off")
    world.until(lambda: world.head("s")["authoritative"], limit=60)
    world.advance(10)
    assert world.admitting() == {("s", 2)}
    world.network.heal()
    world.until(lambda: not world.head("h")["authoritative"], limit=30)
    assert world.status("h")["state"] == "moved_away"


def test_a_standby_cut_off_alone_never_takes_over_or_fences_itself(world):
    world.advance(10)
    world.network.split({"s"}, {"h", "t", "p"})
    world.advance(120)
    assert world.admitting() == {("h", 1)}
    assert not world.head("s")["authoritative"]
    # It asked first and changed nothing: it can still grant the host its lease once the link is back.
    assert fence.room_fence_state(world.gateways["s"].runs.path, ROOM)["fenced_epoch"] == 0
    world.network.heal()
    world.advance(10)
    with world.gateways["h"].acting():
        assert world.automatics["h"].lease.holders(ROOM, 1) == {world.gateways[n].install_id for n in ("s", "t")}


def test_the_restart_window_holds_off_a_takeover(world):
    world.advance(10)
    from gateway.hosted_room_succession_automatic import extend_for_restart
    from gateway.hosted_room_succession_move import append_state
    host = world.gateways["h"]
    until = world.wall0 + world.t + 90
    with host.acting():
        assert extend_for_restart(world.context(host), [ROOM], until) == [ROOM]
        append_state(host.db, ROOM, "host_restarting", until=until)
    world.advance(5)  # the extension and the announcement reach the voters
    world.network.stopped.add("h")
    world.advance(80)
    assert not world.head("s")["authoritative"]
    world.until(lambda: world.head("s")["authoritative"], limit=200)


def test_competing_standbys_end_with_exactly_one_host(world, monkeypatch):
    from gateway import hosted_room_succession_automatic as automatic
    monkeypatch.setattr(automatic, "RANK_STEP_SECONDS", 0.0)  # both standbys start together
    world.advance(10)
    world.network.stopped.add("h")
    world.until(lambda: world.head("s")["authoritative"] or world.head("t")["authoritative"], limit=120)
    world.advance(30)
    hosts = [name for name in ("s", "t") if world.head(name)["authoritative"]]
    assert len(hosts) == 1 and len(world.admitting()) == 1
    other = {"s": "t", "t": "s"}[hosts[0]]
    assert world.head(other)["authority_gateway_id"] == world.gateways[hosts[0]].install_id


def test_continuing_a_paused_host_anyway_is_the_owners_choice(world):
    world.advance(10)
    world.network.split({"h", "p"}, {"s", "t"})
    world.network.stopped |= {"s", "t"}  # really offline this time
    world.until(lambda: world.status("h")["state"] == "paused", limit=25)
    from gateway.hosted_room_succession_automatic import continue_anyway
    host = world.gateways["h"]
    with host.acting():
        with pytest.raises(succession.SuccessionError) as refused:
            continue_anyway(world.context(host, subject="uid:999"), ROOM)
        assert refused.value.reason == "not_owner"
        result = continue_anyway(world.context(host), ROOM)
    assert result["state"] == "ok" and world.head("h")["authority_epoch"] == 2
    assert world.send("h", "user:anyway")
    proof = transition(world, "h")["payload"]["proof"]
    assert proof["statement"] == succession.ANYWAY_TEXT


def test_a_laptop_that_reaches_both_sides_ends_the_split_at_once(world):
    world.advance(10)
    world.network.split({"h"}, {"s", "t"})  # only the laptop p still reaches both sides
    world.until(lambda: world.head("s")["authoritative"], limit=60)
    world.until(lambda: world.head("p")["authority_gateway_id"] == world.gateways["s"].install_id, limit=30)
    # The paused old host asks the group about its epoch every few seconds: p tells it of the move.
    took = world.until(lambda: not world.head("h")["authoritative"], limit=30)
    assert took <= 12
    assert world.status("h")["state"] == "moved_away"
    assert world.head("h")["authority_gateway_id"] == world.gateways["s"].install_id


@pytest.fixture
def participant(tmp_path, monkeypatch):
    """The same group, where p also runs one of the group's Bots: a participant with a copy."""
    value = World(tmp_path, monkeypatch, ("h", "s", "t", "p"), voters=("h", "s", "t"), others=("p",), peers=("p",))
    yield value
    value.close()


def test_after_a_certified_move_the_new_host_retires_a_copy_when_it_disbands(participant):
    """End to end: the copy's retirement, enrolled with the first host, is inherited by the verified
    successor through the continuation grant the copy's computer minted for it, and delivered after
    the successor's Disband with the scope the participant's handler checks."""
    from contextlib import closing
    from gateway import hosted_room_replica_retirement as retirement
    from gateway import hosted_rooms as rooms
    from gateway.hosted_room_peer import decode_room_grant, gateway_room_grant_secret, issue_room_grant
    from gateway.hosted_rooms_common import open_sqlite
    world = participant
    h, s, p = (world.gateways[name] for name in ("h", "s", "p"))
    with h.acting():
        entry = retirement.prepare_home_enrollment(
            h.db, room_id=ROOM, target_install_id=p.install_id, endpoint=p.endpoint, local_gateway_id=h.install_id,
            secret=b"first-home-retirement-secret-of-32-bytes")
    with p.acting():
        retirement.enroll_target(p.db, enrollment=entry, target_install_id=p.install_id)
    world.advance(10)
    world.network.stopped.add("h")
    world.until(lambda: world.head("s")["authoritative"], limit=60)
    world.until(lambda: world.head("p")["authority_gateway_id"] == s.install_id, limit=70)
    # What the copy's computer answers the new host's probe, and the continuation grant it minted at
    # the fence for (successor, epoch): the route the new host's publisher asks and delivers through.
    with p.acting():
        reported = retirement.current_target_enrollment(p.db, room_id=ROOM, authority_gateway_id=s.install_id,
                                                        authority_epoch=2)
        grant = issue_room_grant(gateway_room_grant_secret(), grant_id="grant-succession-p", room_id=ROOM,
                                 home_install_id=s.install_id, authority_gateway_id=s.install_id,
                                 authority_epoch=2, member_id="bot-p", target_install_id=p.install_id,
                                 target_profile="default", permissions=("replicate", "status"))
    with s.acting():
        retirement.inherit_home_enrollment(s.db, enrollment=reported, endpoint=p.endpoint,
                                           local_gateway_id=s.install_id, proof_grant=grant)
        rooms.disband_room(s.db, room_id=ROOM, expected_gateway_id=s.install_id, expected_epoch=2)
        outgoing = retirement.materialize_notice(s.db, enrollment_id=entry["enrollment_id"],
                                                 local_gateway_id=s.install_id, secret_loader=lambda: b"")
    payload = outgoing.payload()
    with p.acting():
        # The participant's handler: the grant is its own, and names exactly the notice's scope.
        claims = decode_room_grant(gateway_room_grant_secret(), outgoing.proof_grant, permission="status")
        assert all(claims[key] == payload[key] for key in
                   ("room_id", "authority_gateway_id", "authority_epoch", "target_install_id"))
        receipt = retirement.retire_copy(p.db, payload=payload, value=outgoing.value, local_gateway_id=p.install_id)
        assert receipt["retired"] and receipt["authority_gateway_id"] == s.install_id
        with closing(open_sqlite(p.db)) as conn:
            assert retirement.copy_retired_locked(conn, ROOM)


def test_a_certified_move_wins_a_tie_with_a_host_continued_anyway(world):
    """The owner continued the cut-off host anyway while its standby took over at the same epoch: when
    they meet, the certified move keeps running and the host continued anyway keeps its messages apart."""
    from gateway import hosted_room_fence as fence
    from gateway.hosted_room_succession_automatic import continue_anyway
    world.advance(10)
    world.network.split({"h"}, {"s", "t"})
    world.network.stopped.add("p")
    world.until(lambda: world.status("h")["state"] == "paused", limit=25, observe=False)
    host = world.gateways["h"]
    with host.acting():
        continue_anyway(world.context(host), ROOM)
    assert world.send("h", "user:anyway")
    world.until(lambda: world.head("s")["authoritative"], limit=60, observe=False)
    assert world.head("h")["authority_epoch"] == world.head("s")["authority_epoch"] == 2  # a tie
    world.network.heal()
    world.until(lambda: world.status("h")["state"] == "continued_on_two", limit=40, observe=False)
    world.until(lambda: world.head("s")["authority_epoch"] == 3, limit=40, observe=False)
    s = world.gateways["s"]
    # A voter that followed the host continued anyway sets that aside and follows the kept host: never
    # back into the epoch it fenced. Its next grant lets the kept host serve.
    world.until(lambda: world.head("s")["serving"], limit=20, observe=False)
    assert world.head("t")["authority_gateway_id"] == s.install_id and world.head("t")["authority_epoch"] == 3
    assert world.status("s")["conflict"]["running_on"]["install_id"] == s.install_id
    # The host continued anyway stops, keeps its room and message, and its Bots take the kept host's work.
    assert world.head("h")["authoritative"] and not world.head("h")["serving"]
    assert "user:anyway" in world.texts("h")
    assert fence.room_fence_state(host.runs.path, ROOM)["authority"] == {"epoch": 3, "install_id": s.install_id}
    world.advance(20, observe=False)
    assert world.status("h")["state"] == "continued_on_two"  # no timeout: until the owner chooses


def test_a_stuck_automatic_move_says_so_and_offers_continue_after_a_while(world, monkeypatch):
    from gateway import hosted_room_succession_automatic as automatic
    from gateway import hosted_room_succession_status as status_module

    def stuck(ctx, room_id, **_):
        raise succession.SuccessionError("the voters did not answer in time", reason="no_majority")

    monkeypatch.setattr(automatic, "majority_move", stuck)
    world.advance(10)
    world.network.stopped.add("h")
    world.advance(status_module.UNREACHABLE_AFTER_SECONDS + 70, observe=False)
    current = world.status("s")
    assert current["state"] == "host_unreachable" and current["unavailable_reason"] == "takeover_waiting"
    assert not any(action["action"] == "continue" for action in current["actions"])
    world.advance(status_module.TAKEOVER_WAIT_SECONDS, observe=False)
    current = world.status("s")
    assert current["unavailable_reason"] == "takeover_waiting"
    offered = next(action["targets"] for action in current["actions"] if action["action"] == "continue")
    assert set(offered) == {world.gateways["s"].install_id, world.gateways["t"].install_id}


def test_every_computer_keeping_a_copy_gets_a_switch_on_the_host(world):
    world.advance(10)
    designate = next(action for action in world.status("h")["actions"] if action["action"] == "designate")
    assert sorted(designate["targets"]) == sorted(world.gateways[name].install_id for name in ("s", "t", "p"))


def test_turning_automatic_moves_off_shows_the_request_until_the_voters_settle(world):
    from gateway import hosted_room_custody as custody
    world.advance(10)
    h = world.gateways["h"]
    assert (world.status("h")["automatic"]["enabled"], world.status("h")["automatic"]["pending"]) == (True, None)
    with h.acting():
        custody.set_automatic(h.db, room_id=ROOM, enabled=False)
    current = world.status("h")["automatic"]
    assert (current["enabled"], current["pending"], current["state"]) == (True, False, "ready")
    world.settle("h")
    current = world.status("h")["automatic"]
    assert (current["enabled"], current["pending"], current["state"]) == (False, None, "off")


def test_a_host_that_promised_the_next_epoch_never_admits_again_at_its_own(tmp_path, monkeypatch):
    """Five voters, one-way cuts: the host promised the next epoch to a standby while paused. Leases coming
    back from the voters it reaches again never let it admit at its old epoch, so nothing it would
    report as protected can be set aside."""
    from gateway import hosted_room_succession_move as move
    from tests.gateway.fixtures.succession_world import Network

    class OneWay(Network):
        def __init__(self, names):
            super().__init__(names)
            self.oneway: set[tuple[str, str]] = set()

        def reachable(self, a, b):
            return super().reachable(a, b) and (a, b) not in self.oneway

    names = ["h", "s", "t", "u", "v"]
    five = World(tmp_path, monkeypatch, names, voters=names)
    try:
        five.network = OneWay(five.network.names)
        five.advance(12)
        net = five.network
        net.split({"s"}, {"t", "v"})
        net.oneway |= {("h", n) for n in ("s", "t", "u", "v")}
        original = move.record_fences

        def drop_link_after_fences(ctx, room_id, record, outcomes, to_epoch):
            result = original(ctx, room_id, record, outcomes, to_epoch)
            if five.acting_name() == "s":
                net.cut.add(frozenset(("s", "h")))
            return result

        monkeypatch.setattr(move, "record_fences", drop_link_after_fences)
        five.until(lambda: five.head("s")["authoritative"], limit=80, observe=True)
        net.oneway -= {("h", "t"), ("h", "v")}
        five.advance(6, observe=True)
        assert ("h", 1) not in five.admitting() and not five.send("h", "user:after-promise")
        net.cut.discard(frozenset(("s", "t")))
        net.oneway.clear()
        net.heal()
        five.advance(60, observe=True)
        assert five.admitting() == {("s", five.head("s")["authority_epoch"])}
    finally:
        five.close()
