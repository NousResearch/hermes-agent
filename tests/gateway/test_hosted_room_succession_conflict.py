"""Continued on two computers: the group keeps running on one by rule, the other keeps its messages apart
until the owner chooses, and nothing is merged."""

import pytest

from gateway import hosted_room_fence as fence
from gateway import hosted_room_succession as succession
from gateway import hosted_room_succession_move as move
from gateway import hosted_room_succession_return as returning
from gateway import hosted_room_succession_status as status_module
from tests.gateway.fixtures.succession import (
    ROOM, context, copy_to, events, head, home_room, make_gateways, message)


@pytest.fixture
def split(tmp_path):
    """The host is lost; s and t were both continued while each reached only its own backup."""
    gateways = make_gateways(tmp_path, "h", "s", "t", "p1", "p2")
    home_room(gateways, "h", successors=("s", "t"))
    for name in ("s", "t", "p1", "p2"):
        copy_to(gateways["h"], gateways[name])
    s, t = gateways["s"], gateways["t"]
    for gateway, cut in ((s, ("h", "t", "p2")), (t, ("h", "s", "p1"))):
        with gateway.acting():
            ctx = context(gateway, gateways, down=cut)
            preview = move.preview(ctx, ROOM, gateway.install_id)
            move.continue_here(ctx, ROOM, gateway.install_id, preview_id=preview["preview_id"], confirm=True)
    copy_to(s, gateways["p1"])
    copy_to(t, gateways["p2"])
    with s.acting():
        message(s.db, "user:on-s", gateway=s.install_id, epoch=2)
    with t.acting():
        message(t.db, "user:on-t", gateway=t.install_id, epoch=2)
    yield gateways
    for gateway in gateways.values():
        gateway.close()


def state(gateway, gateways, **options):
    with gateway.acting():
        return status_module.status(context(gateway, gateways, **options), ROOM)


def meet(gateways):
    """The partition heals: s's announcement reaches t, which already continued the group itself."""
    s = gateways["s"]
    with s.acting():
        move.announce(context(s, gateways, down=("h",)), ROOM)


def sides(gateways):
    """``(winner, loser)``: both continued at epoch 2 by hand, so the lower installation id keeps running."""
    s, t = gateways["s"], gateways["t"]
    return (s, t) if s.install_id < t.install_id else (t, s)


def message_on(gateway):
    return "user:on-" + gateway.name


def test_the_first_contact_keeps_the_group_running_on_one_and_offers_the_choice(split):
    winner, loser = sides(split)
    meet(split)
    for gateway in (winner, loser):
        current = state(gateway, split, down=("h",))
        assert current["state"] == "continued_on_two"
        assert {host["install_id"] for host in current["conflict"]["hosts"]} == {winner.install_id, loser.install_id}
        assert current["conflict"]["running_on"]["install_id"] == winner.install_id
        assert {"action": "keep", "targets": [winner.install_id, loser.install_id]} in current["actions"]
    # The winner keeps serving; the other one stops and keeps its own messages.
    with winner.acting():
        assert succession.paused_reason(winner.db, ROOM) is None and head(winner)["serving"]
    with loser.acting():
        assert succession.paused_reason(loser.db, ROOM) == "room_authority_conflict" and not head(loser)["serving"]
    assert state(loser, split, down=("h",), subject="uid:999")["actions"] == []


def test_on_a_tie_the_kept_host_moves_on_and_every_computer_follows_it(split):
    from gateway.platforms.api_server_run_scope import room_run_scope_key
    winner, loser = sides(split)
    meet(split)
    with winner.acting():
        move.maintain(context(winner, split, down=("h",)), [ROOM])
    assert head(winner)["authority_epoch"] == 3 and head(winner)["serving"]
    moved = next(event for event in reversed(events(winner)) if event["kind"] == "authority.transition")
    # Nobody chose by hand: the rule's fresh epoch is an automatic move, signed as the rule's choice.
    assert moved["payload"]["reason"] == "automatic" and moved["payload"]["proof_kind"] == "attested"
    assert moved["payload"]["proof"]["decision"]["decided_by"] == "rule"
    # The other host keeps its room and its messages, paused, while its Bots take the kept host's work.
    assert head(loser)["authoritative"] and head(loser)["authority_epoch"] == 2
    assert message_on(loser) in [event["payload"].get("text") for event in events(loser)]
    assert fence.room_fence_state(loser.runs.path, ROOM)["authority"] == {"epoch": 3, "install_id": winner.install_id}
    identity = {"room_id": ROOM, "home_install_id": winner.install_id, "authority_gateway_id": winner.install_id,
                "authority_epoch": 3, "member_id": "writer", "target_install_id": loser.install_id,
                "target_profile": "default"}
    assert loser.runs.reserve(room_run_scope_key(identity), "room:task-new:1", "fp", "run-new", {"status": "queued"},
                              identity=identity)[0] == "created"
    assert state(loser, split, down=("h",))["state"] == "continued_on_two"  # until the owner chooses
    # Every copy follows the kept host, whichever one it followed.
    for name in ("p1", "p2"):
        copy = split[name]
        with copy.acting():
            returning.follow_up(context(copy, split, down=("h",)), ROOM)
        copy_to(winner, copy)
        assert head(copy)["authority_gateway_id"] == winner.install_id and head(copy)["authority_epoch"] == 3


def test_a_copy_behind_the_rules_fresh_epoch_catches_up_across_it(split):
    """The kept host keeps its own head for the epoch it leaves, so a copy that missed that epoch's last
    events catches up across the change: to that head first, then past the change."""
    from contextlib import closing
    from gateway import hosted_room_custody as custody
    from gateway import hosted_rooms as rooms
    winner, loser = sides(split)
    copy = split["p1"] if winner is split["s"] else split["p2"]  # it followed the winner at epoch 2
    meet(split)
    with winner.acting():
        move.maintain(context(winner, split, down=("h",)), [ROOM])
    assert head(winner)["authority_epoch"] == 3
    moved = next(event for event in reversed(events(winner)) if event["kind"] == "authority.transition")
    with winner.acting(), closing(rooms._read_connection(winner.db)) as conn:
        kept = {(item["host"], item["epoch"]): item for item in custody.heads_locked(conn, ROOM)}
    left = kept[(winner.install_id, 2)]
    assert left["seq"] == moved["seq"] - 1  # the last event before the change
    assert message_on(winner) not in [event["payload"].get("text") for event in events(copy, "hosted_room_replica_events")]
    ctx = context(copy, split, down=("h",))
    with copy.acting():
        with pytest.raises(custody.CustodyError):
            move.catch_up(ctx, ROOM, winner.install_id)  # the newest head alone can't cross the tail
        move.catch_up(ctx, ROOM, winner.install_id, left)
        move.catch_up(ctx, ROOM, winner.install_id)
    assert (head(copy)["authority_gateway_id"], head(copy)["authority_epoch"]) == (winner.install_id, 3)
    assert message_on(winner) in [event["payload"].get("text") for event in events(copy, "hosted_room_replica_events")]


def test_keeping_the_running_host_steps_the_other_down_with_its_messages_apart(split):
    winner, loser = sides(split)
    meet(split)
    with winner.acting():
        move.maintain(context(winner, split, down=("h",)), [ROOM])
        kept = move.keep(context(winner, split, down=("h",)), ROOM, winner.install_id)
    assert kept["state"] == "ok"
    assert head(loser)["authoritative"] is False
    with loser.acting():
        returned = succession.load_record(loser.db, ROOM, "return")
        branch = returning.branch_log(loser.db, ROOM, returned["branch_id"])
    assert message_on(loser) in [event["payload"].get("text") for event in branch["events"]]
    assert state(loser, split, down=("h",))["state"] == "moved_away"
    texts = [event["payload"].get("text") for event in events(winner)]
    assert message_on(winner) in texts and message_on(loser) not in texts


def test_keeping_the_other_host_switches_at_a_fresh_epoch(split):
    winner, loser = sides(split)
    meet(split)
    with winner.acting():
        move.maintain(context(winner, split, down=("h",)), [ROOM])
    with loser.acting():
        switched = move.keep(context(loser, split, down=("h",)), ROOM, loser.install_id)
    # The running host stepped aside first; the kept one continues above every epoch known.
    assert switched["state"] == "ok" and head(loser)["authority_epoch"] == 4 and head(loser)["serving"]
    assert not head(winner)["serving"]
    with loser.acting():
        move.announce(context(loser, split, down=("h",)), ROOM)
    assert head(winner)["authoritative"] is False
    with winner.acting():
        returning.follow_up(context(winner, split, down=("h",)), ROOM)
    copy_to(loser, winner)
    assert head(winner)["authority_gateway_id"] == loser.install_id and head(winner)["authority_epoch"] == 4
    texts = [event["payload"].get("text") for event in events(loser)]
    assert message_on(loser) in texts and message_on(winner) not in texts


def test_a_choice_signed_by_neither_computer_changes_nothing(split):
    winner, loser, p1 = *sides(split), split["p1"]
    meet(split)
    record = succession.load_record(winner.db, ROOM, "conflict")
    with p1.acting():
        forged = move._sign_decision(context(p1, split), ROOM, {**record, "hosts": record["hosts"]}, loser.install_id)
    with loser.acting():
        with pytest.raises(succession.SuccessionError):
            from gateway import hosted_room_succession_backup as backup
            backup.answer_decision(loser.backup_context(), forged)
    assert head(loser)["authoritative"] and state(loser, split, down=("h",))["state"] == "continued_on_two"


def test_the_rule_can_only_keep_its_own_winner(split):
    winner, loser = sides(split)
    meet(split)
    record = succession.load_record(loser.db, ROOM, "conflict")
    with loser.acting():  # the other host signing that the rule keeps it is refused everywhere
        claimed = move._sign_decision(context(loser, split), ROOM, record, loser.install_id, by_rule=True)
    for gateway in (winner, split["p1"]):
        with gateway.acting(), pytest.raises(succession.SuccessionError):
            from contextlib import closing
            from gateway import hosted_rooms as rooms
            with closing(rooms._read_connection(gateway.db)) as conn:
                succession.verify_decision_locked(conn, ROOM, claimed)


def test_the_side_passed_over_checks_the_rule_against_its_own_head(split):
    """A host that misstates the other side to make the rule keep it is refused by that other side."""
    import copy
    winner, loser = sides(split)
    meet(split)
    record = copy.deepcopy(succession.load_record(loser.db, ROOM, "conflict"))
    # The loser claims the winner's side is a bare host at a lower epoch, so the rule would keep it.
    for key in ("mine", "theirs"):
        side = record[key]
        if side.get("event") and side["event"]["payload"]["successor_gateway_id"] == winner.install_id:
            record[key] = {"event": None, "head": {"install_id": winner.install_id, "epoch": 1, "since": None}}
    record["hosts"] = [h if h["install_id"] != winner.install_id else {**h, "epoch": 1, "proof_kind": None}
                       for h in record["hosts"]]
    with loser.acting():
        claimed = move._sign_decision(context(loser, split), ROOM, record, loser.install_id, by_rule=True)
    from contextlib import closing
    from gateway import hosted_rooms as rooms
    with winner.acting(), closing(rooms._read_connection(winner.db)) as conn, pytest.raises(
            succession.SuccessionError):
        succession.verify_decision_locked(conn, ROOM, claimed)
