"""Handing a group over on purpose, and learning of a move from any device, across simulated gateways."""

from contextlib import closing

import pytest

from gateway import hosted_room_fence as fence
from gateway import hosted_room_succession as succession
from gateway import hosted_room_succession_handover as handover
from gateway import hosted_room_succession_move as move
from gateway import hosted_room_succession_return as returning
from gateway import hosted_room_succession_status as status_module
from gateway import hosted_rooms as rooms
from tests.gateway.fixtures.succession import (
    ROOM, context, copy_to, events, head, home_room, make_gateways, message)


@pytest.fixture
def gateways(tmp_path):
    values = make_gateways(tmp_path, "h", "s", "p")
    home_room(values, "h", messages=3, successors=("s",))
    copy_to(values["h"], values["s"])
    copy_to(values["h"], values["p"])
    yield values
    for gateway in values.values():
        gateway.close()


def status_of(gateway, gateways, **options):
    with gateway.acting():
        return status_module.status(context(gateway, gateways, **options), ROOM)


def test_the_host_hands_its_exact_history_to_the_standby_and_becomes_a_copy(gateways):
    h, s, p = gateways["h"], gateways["s"], gateways["p"]
    with h.acting():
        message(h.db, "user:last", gateway=h.install_id)
        handover.hand_over(context(h, gateways), ROOM, s.install_id)
    assert head(s)["authoritative"] and head(s)["authority_epoch"] == 2
    transition = next(event for event in events(s) if event["kind"] == "authority.transition")
    assert transition["payload"]["proof_kind"] == "handover" and transition["payload"]["reason"] == "handover"
    assert transition["payload"]["proof"]["statement"]["last_seq"] == transition["seq"] - 1
    # The old host follows the standby and set nothing aside; the backup learned the move too.
    assert head(h)["authoritative"] is False and head(h)["authority_gateway_id"] == s.install_id
    assert status_of(h, gateways)["moved"]["separate_events"] == 0
    # The backup was one event behind: it follows once it holds the history the statement names.
    assert fence.room_fence_state(p.runs.path, ROOM)["authority"] is None
    copy_to(s, p)
    with s.acting():
        move.announce(context(s, gateways), ROOM)
    assert head(p)["authority_gateway_id"] == s.install_id
    assert fence.room_fence_state(p.runs.path, ROOM)["authority"] == {"epoch": 2, "install_id": s.install_id}
    texts = [event["payload"].get("text") for event in events(s) if event["kind"] == "message.user"]
    assert texts[-1] == "user:last"


def test_a_handover_never_completes_without_the_signed_statement_and_the_host_resumes(gateways):
    h, s, p = gateways["h"], gateways["s"], gateways["p"]
    # The statement never reached the standby: nothing moved, the host resumes at once.
    with h.acting(), pytest.raises(succession.SuccessionError):
        handover.hand_over(context(h, gateways, down=("s",)), ROOM, s.install_id)
    with h.acting():
        assert head(h)["authoritative"] and succession.paused_reason(h.db, ROOM) is None
    assert status_of(h, gateways)["last_attempt"]["to"]["install_id"] == s.install_id

    # No answer in time: the standby may have continued, so the host waits to learn the outcome.
    def slow(gateway, name, body):
        if name == "handover":
            raise TimeoutError("timed out")
    with h.acting(), pytest.raises(succession.SuccessionError):
        handover.hand_over(context(h, gateways, before=slow), ROOM, s.install_id)
    with h.acting():
        assert succession.paused_reason(h.db, ROOM) == "room_authority_promised"
        # The standby answers that it holds nothing beyond the host's epoch: the host resumes.
        assert handover.recover(context(h, gateways), ROOM) is True
        assert succession.paused_reason(h.db, ROOM) is None
    # A statement the host never signed, or for another history, moves nothing.
    with h.acting(), closing(rooms._read_connection(h.db)) as conn:
        statement = handover.statement_for(conn, ROOM, successor=s.install_id, to_epoch=2)
    with p.acting():
        forged = {"statement": statement, "signature": succession.sign(handover.HANDOVER, statement)}
    with h.acting():
        wrong = handover.sign({**statement, "last_hash": "0" * 64})
    for proof in (forged, wrong):
        with s.acting(), pytest.raises(succession.SuccessionError):
            handover.accept(context(s, gateways), ROOM, proof)
        assert head(s)["authoritative"] is False


def test_a_returning_host_learns_the_move_from_any_device_and_steps_down_quietly(gateways):
    h, s = gateways["h"], gateways["s"]
    with s.acting():
        ctx = context(s, gateways, down=("h",))
        preview = move.preview(ctx, ROOM, s.install_id)
        move.continue_here(ctx, ROOM, s.install_id, preview_id=preview["preview_id"], confirm=True)
        with closing(rooms._read_connection(s.db)) as conn:
            chain = succession.chain_after_locked(conn, ROOM, 1)
    assert [event["kind"] for event in chain][:2] == ["message.user", "authority.transition"]
    with h.acting():
        replayed = returning.learn(context(h, gateways, down=("s",)), ROOM, chain[:1])
        assert replayed["learned"] is False
        forged = [dict(event) for event in chain]
        forged[1] = {**forged[1], "payload": {**forged[1]["payload"], "successor_gateway_id": h.install_id}}
        with pytest.raises(succession.SuccessionError):
            returning.learn(context(h, gateways, down=("s",)), ROOM, forged)
        assert head(h)["authoritative"]
        learned = returning.learn(context(h, gateways, down=("s",)), ROOM, chain)
    assert learned["learned"] and head(h)["authoritative"] is False
    assert status_of(h, gateways)["state"] == "moved_away"


def test_a_handover_interrupted_before_the_signature_resumes_the_host(gateways):
    h, s = gateways["h"], gateways["s"]
    with h.acting():
        succession.save_record(h.db, ROOM, "move", {"state": "handing_over", "to": s.install_id, "signed": False,
                                                    "from_epoch": 1, "reason": "handover"})
        assert succession.paused_reason(h.db, ROOM) == "room_authority_promised"
        assert handover.recover(context(h, gateways), ROOM) is True
    assert head(h)["serving"] and status_of(h, gateways)["last_attempt"]["error"] == "target_not_ready"


def test_a_signed_handover_waits_to_learn_whether_the_standby_continued(gateways):
    h, s = gateways["h"], gateways["s"]
    signed = {"state": "handing_over", "to": s.install_id, "signed": True, "from_epoch": 1, "reason": "handover"}
    with h.acting():
        succession.save_record(h.db, ROOM, "move", signed)
        # The standby can't be asked: the host stays paused rather than risk two hosts.
        assert handover.recover(context(h, gateways, down=("s",)), ROOM) is False
        assert not head(h)["serving"]
        # It answers that it holds nothing beyond the host's epoch: nothing moved, the host resumes.
        assert handover.recover(context(h, gateways), ROOM) is True
        assert head(h)["serving"]
    # Had the standby continued, the host learns it and steps down instead of resuming.
    with h.acting(), closing(rooms._read_connection(h.db)) as conn:
        proof = handover.sign(handover.statement_for(conn, ROOM, successor=s.install_id, to_epoch=2))
    copy_to(h, s)
    with s.acting():
        handover.accept(context(s, gateways, down=("h",)), ROOM, proof)
    with h.acting():
        succession.save_record(h.db, ROOM, "move", {**signed, "from_epoch": 1})
        assert handover.recover(context(h, gateways), ROOM) is False
    assert head(h)["authoritative"] is False and head(h)["authority_gateway_id"] == s.install_id


class Turns:
    """The host's running turns as its hosted service sees them: each settles after ``polls`` looks."""

    def __init__(self, h, monkeypatch, *, polls):
        self.h, self.polls, self.running = h, polls, True
        monkeypatch.setattr(handover, "DRAIN_POLL_SECONDS", 0.0)
        monkeypatch.setattr(handover, "unsettled_turns",
                            lambda db, room_id: ["task-running"] if self.running else [])

    def publish_settled(self, room_id):
        self.polls -= 1
        if self.running and self.polls <= 0:  # the turn finished: its reply is published into the log
            message(self.h.db, "user:reply-of-running-turn", gateway=self.h.install_id)
            self.running = False

    def wakeup(self):
        pass

    def stop_room(self, room_id, cancel_id):
        return 0


def transition_payload(gateway):
    return next(event for event in events(gateway) if event["kind"] == "authority.transition")["payload"]


def test_a_turn_running_at_a_handover_settles_inside_the_signed_history(gateways, monkeypatch):
    """A turn running when the host hands over: the host lets it settle before it signs, so the turn's
    reply is part of the history the new host continues from."""
    h, s = gateways["h"], gateways["s"]
    turns = Turns(h, monkeypatch, polls=3)
    with h.acting():
        handover.hand_over(context(h, gateways, service=turns), ROOM, s.install_id)
    kept = [event["payload"].get("text") for event in events(s) if event["kind"] == "message.user"]
    assert "user:reply-of-running-turn" in kept
    assert transition_payload(s)["at_risk"] == 0 and status_of(h, gateways)["moved"]["separate_events"] == 0


def test_a_turn_still_running_after_the_drain_is_counted_at_risk(gateways, monkeypatch):
    h, s = gateways["h"], gateways["s"]
    Turns(h, monkeypatch, polls=10 ** 6)  # never settles: a stop's bounded drain gives up
    with h.acting():
        handover.hand_over(context(h, gateways, service=Turns(h, monkeypatch, polls=10 ** 6)), ROOM, s.install_id,
                           drain_seconds=0.05)
    assert transition_payload(s)["at_risk"] == 1


def test_the_owners_move_waits_for_running_turns_unless_told_to_move_now(gateways, monkeypatch):
    h, s = gateways["h"], gateways["s"]
    turns = Turns(h, monkeypatch, polls=10 ** 6)
    with h.acting():
        ctx = context(h, gateways, service=turns)
        handover.request_move(ctx, ROOM, s.install_id)
        current = status_module.status(ctx, ROOM)
        assert current["state"] == "moving" and current["moving"]["step"] == "waiting_for_turns"
        assert current["moving"]["running"] == 1 and {"action": "move_now"} in current["actions"]
        assert succession.paused_reason(h.db, ROOM) == "room_authority_promised"  # nothing new starts
        assert not handover.continue_move(ctx, ROOM)  # still running, deadline far away
        handover.move_now(ctx, ROOM)
    assert head(s)["authoritative"] and transition_payload(s)["at_risk"] == 1
    assert head(h)["authoritative"] is False


def test_the_old_hosts_bots_take_part_again_once_the_group_moves_back(gateways):
    from gateway import hosted_room_custody as custody
    h, s = gateways["h"], gateways["s"]
    with h.acting():
        handover.hand_over(context(h, gateways), ROOM, s.install_id)
    with s.acting():
        move.maintain(context(s, gateways), [ROOM])  # the new host's upkeep finishes the move
    with s.acting(), closing(rooms._read_connection(s.db)) as conn:
        entry = succession.custodians_by_id(succession.configuration_locked(conn, ROOM))[h.install_id]
    assert entry["successor"], "the old host stays a successor, so the group can move back"
    # On the new host both of the old host's Bots, the default and a named profile, wait for it.
    current = status_of(s, gateways)
    assert [(bot["member_id"], bot["on"]["install_id"], bot["on"]["name"], bot["on"]["reachable"])
            for bot in current["unavailable_bots"]] == [
        ("writer", h.install_id, "H", False), ("reviewer", h.install_id, "H", False)]
    assert not any(action["action"] == "move" and h.install_id in action["targets"] for action in current["actions"])
    # The old host answers as a copy: Move back is offered, and the Bots say where they can take part again.
    with h.acting(), closing(rooms._read_connection(h.db)) as conn:
        mark = succession.watermark_locked(conn, ROOM)
    with s.acting():
        assert custody.record_acknowledgment(s.db, room_id=ROOM, install_id=h.install_id, watermark=mark) \
            == "acknowledged"
    current = status_of(s, gateways)
    assert [bot["on"]["reachable"] for bot in current["unavailable_bots"]] == [True, True]
    assert h.install_id in next(action["targets"] for action in current["actions"] if action["action"] == "move")
    # Moving back is a planned handover: on its own computer the group has all its Bots again.
    with s.acting():
        handover.hand_over(context(s, gateways), ROOM, h.install_id)
    with h.acting():
        move.maintain(context(h, gateways), [ROOM])
    assert head(h)["authoritative"] and head(h)["authority_epoch"] == 3
    assert head(s)["authority_gateway_id"] == h.install_id
    assert status_of(h, gateways)["unavailable_bots"] == []
    with h.acting():
        assert succession.inherited_origin(h.db, ROOM) is None  # its own profiles run both Bots


def test_a_statement_kept_after_the_host_resumed_never_gives_its_lease_back(gateways):
    from gateway import hosted_room_clock as clock
    from gateway import hosted_room_fence as fence
    from gateway import hosted_room_succession_backup as backup
    h, s, p = gateways["h"], gateways["s"], gateways["p"]
    with h.acting(), closing(rooms._read_connection(h.db)) as conn:
        proof, release = handover.signed_over(context(h, gateways), conn, ROOM, successor=s.install_id, to_epoch=2)
    signed_at = release["token"]["signed_at"]

    def lease_at_p(sent_at):
        with p.acting():
            fence.release_lease(p.runs.path, room_id=ROOM, epoch=1, authority_install_id=h.install_id)
            fence.grant_lease(p.runs.path, room_id=ROOM, epoch=1, authority_install_id=h.install_id,
                              duration_s=20.0, host_sent_at=sent_at, host_boot=clock.boot_id())

    def replayed(token=release):
        with p.acting():
            backup._release_for_handover(p.backup_context(gateways), ROOM, proof, token, head=head(p),
                                         candidate=s.install_id, promise_epoch=2)
            return fence.room_lease_state(p.runs.path, ROOM) is None

    # The handover failed and the host resumed: it asked this voter again after it signed.
    lease_at_p(signed_at + 1.0)
    assert not replayed()
    # A lease asked for before the signature is the one the token gives back, and only with the token
    # the host signed for this step.
    lease_at_p(signed_at - 1.0)
    assert not replayed(None)
    forged = {"token": {**release["token"], "signed_at": signed_at + 100.0}, "signature": release["signature"]}
    assert not replayed(forged)
    assert replayed()
    # ...unless this copy already holds history the host wrote after signing.
    with h.acting():
        message(h.db, "user:after", gateway=h.install_id)
    copy_to(h, p)
    lease_at_p(signed_at - 1.0)
    assert not replayed()


def test_a_host_counts_no_grant_asked_for_before_it_signed_a_handover():
    from gateway import hosted_room_clock as clock
    from gateway.hosted_room_succession_automatic import HostLease
    lease = HostLease()
    early = lease.request(ROOM, 1)
    assert early["boot"] == clock.boot_id()
    lease.void(ROOM, clock.now())
    assert lease.acknowledged(ROOM, "voter", {"granted_until_s": 20.0, "epoch": 1}, early["sent_at"]) is False
    late = lease.request(ROOM, 1)
    assert lease.acknowledged(ROOM, "voter", {"granted_until_s": 20.0, "epoch": 1}, late["sent_at"]) is True



class _Crash(BaseException):
    """The standby's process dies between writing its transition and saving that it did."""


def test_a_standby_that_crashed_right_after_its_transition_still_finishes_the_handover(gateways, monkeypatch):
    h, s = gateways["h"], gateways["s"]
    original = move._save

    def save(ctx, room_id, kind, record):
        if kind == "move" and record.get("transition_committed") and ctx.db_path == s.db:
            raise _Crash()
        return original(ctx, room_id, kind, record)

    monkeypatch.setattr(move, "_save", save)
    with h.acting(), pytest.raises(_Crash):
        handover.hand_over(context(h, gateways), ROOM, s.install_id)
    monkeypatch.setattr(move, "_save", original)
    assert head(s)["authoritative"]  # the transition is in its log; its record never said so
    with s.acting():
        move.maintain(context(s, gateways), [ROOM])  # the restarted standby's upkeep
        assert head(s)["serving"] and succession.load_record(s.db, ROOM, "move")["state"] == "moved"
    with h.acting():
        handover.recover(context(h, gateways), ROOM)
    assert not head(h)["authoritative"] and head(h)["authority_gateway_id"] == s.install_id
