"""Continuing a group by hand across simulated gateways: real identities and stores in one process, with
each request calling the receiving gateway's real handler directly."""

import sqlite3
from contextlib import closing

import pytest

from gateway import hosted_room_custody as custody
from gateway import hosted_room_fence as fence
from gateway import hosted_room_succession as succession
from gateway import hosted_room_succession_move as move
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


def continue_on(gateway, gateways, *, down=("h",), **options):
    with gateway.acting():
        ctx = context(gateway, gateways, down=down, **options)
        preview = move.preview(ctx, ROOM, gateway.install_id)
        return preview, move.continue_here(ctx, ROOM, gateway.install_id, preview_id=preview["preview_id"],
                                           confirm=True)


def test_the_owner_continues_on_a_designated_backup_and_the_other_backups_follow(gateways):
    h, s, p = gateways["h"], gateways["s"], gateways["p"]
    preview, result = continue_on(s, gateways)
    assert preview["target"] == {"install_id": s.install_id, "name": "S", "operator_name": "S Operator"}
    assert preview["owner"] == {"name": "Dana"}
    assert preview["behind_by"] == 0 and preview["at_risk"] == {"count": 0}
    assert preview["work"] == {"completed": 0, "elsewhere": 0, "unknown": 0, "waiting_for_host": 0}
    assert [bot["member_id"] for bot in preview["unavailable_bots"]] == ["writer", "reviewer"]
    assert preview["cautions"] == [{"code": "host_may_be_running"}]
    assert result["state"] == "ok" and result["host"]["install_id"] == s.install_id
    assert result["this_install"]["role"] == "host"
    assert result["previous_host"]["install_id"] == h.install_id
    assert head(s)["authoritative"] and head(s)["authority_epoch"] == 2
    transition = next(event for event in events(s) if event["kind"] == "authority.transition")
    payload = transition["payload"]
    assert payload["proof_kind"] == "attested" and payload["successor_gateway_id"] == s.install_id
    assert (payload["reason"], payload["from_name"], payload["to_name"], payload["at_risk"]) == ("manual", "H", "S", 0)
    assert payload["text"] == "This group now continues on S."
    with s.acting(), closing(rooms._read_connection(s.db)) as conn:
        configured = custody.configuration_locked(conn, ROOM)
    roles = {entry["install_id"]: (entry["role"], entry["successor"]) for entry in configured["custodians"]}
    # The old host keeps its copy and may continue the group again, so the owner can move it back.
    assert roles[s.install_id] == ("authority", False) and roles[h.install_id] == ("custodian", True)
    # The backup that learned the move fences the old epoch and follows the new host.
    state = fence.room_fence_state(p.runs.path, ROOM)
    assert state["fenced_epoch"] == 1 and state["authority"] == {"epoch": 2, "install_id": s.install_id}
    copy_to(s, p)
    assert head(p)["authority_gateway_id"] == s.install_id and head(p)["authority_epoch"] == 2
    # Continuing again resumes nothing and changes nothing.
    with s.acting():
        again = move.continue_here(context(s, gateways, down=("h",)), ROOM, s.install_id,
                                   preview_id=preview["preview_id"], confirm=True)
    assert again["state"] == "ok" and head(s)["authority_epoch"] == 2


def test_continuing_needs_the_owner_this_computer_and_its_consent(gateways):
    s, p = gateways["s"], gateways["p"]
    with s.acting():
        for ctx, target, reason in (
                (context(s, gateways, down=("h",), subject="uid:999"), s.install_id, "not_owner"),
                (context(s, gateways, down=("h",)), p.install_id, "target_not_local")):
            with pytest.raises(succession.SuccessionError) as refused:
                move.preview(ctx, ROOM, target)
            assert refused.value.reason == reason
        assert refused.value.detail == {"target": {"install_id": p.install_id, "name": "P"}}
        # This computer's operator may act for the owner.
        operator = context(s, gateways, down=("h",), subject=None, operator=True)
        assert move.preview(operator, ROOM, s.install_id)["target"]["install_id"] == s.install_id
        custody.set_local_consent(s.db, room_id=ROOM, allowed=False)
        with pytest.raises(succession.SuccessionError) as refused:
            move.preview(context(s, gateways, down=("h",)), ROOM, s.install_id)
    assert refused.value.reason == "target_not_ready"
    # A backup the owner never designated cannot continue the group.
    with p.acting():
        with pytest.raises(succession.SuccessionError) as refused:
            move.preview(context(p, gateways, down=("h",), operator=True), ROOM, p.install_id)
    assert refused.value.reason == "target_not_ready"


def test_a_reachable_host_refuses_and_a_stale_preview_is_refused(gateways):
    h, s = gateways["h"], gateways["s"]
    with s.acting():
        with pytest.raises(succession.SuccessionError) as refused:
            move.preview(context(s, gateways, down=()), ROOM, s.install_id)
        assert refused.value.reason == "host_reachable"
        ctx = context(s, gateways, down=("h",))
        preview = move.preview(ctx, ROOM, s.install_id)
    with h.acting():
        message(h.db, "user:late", gateway=h.install_id)
    copy_to(h, gateways["p"])
    with s.acting():
        with pytest.raises(succession.SuccessionError) as refused:
            move.continue_here(ctx, ROOM, s.install_id, preview_id=preview["preview_id"], confirm=True)
        assert refused.value.reason == "preview_stale"
        assert status_module.status(ctx, ROOM)["last_attempt"]["error"] == "preview_stale"
        assert move.preview(ctx, ROOM, s.install_id)["behind_by"] == 1


def test_a_successor_adopts_the_most_complete_reachable_copy(gateways):
    h, s, p = gateways["h"], gateways["s"], gateways["p"]
    with h.acting():
        for number in range(2):
            message(h.db, f"user:extra:{number}", gateway=h.install_id)
    copy_to(h, p)
    preview, result = continue_on(s, gateways)
    assert preview["behind_by"] == 2
    assert result["state"] == "ok"
    texts = [event["payload"].get("text") for event in events(s) if event["kind"] == "message.user"]
    assert texts[-2:] == ["user:extra:0", "user:extra:1"]


def test_messages_the_host_announced_that_no_copy_holds_are_counted_at_risk(gateways):
    h, s, p = gateways["h"], gateways["s"], gateways["p"]
    held = head(s)["latest_seq"]
    with h.acting():
        for number in range(3):
            message(h.db, f"user:late:{number}", gateway=h.install_id)
    copy_to(h, s, through=held + 1)  # S holds one of them and heard of all three
    copy_to(h, p, through=held + 2)  # P holds two, with the head the host signed for them
    preview, result = continue_on(s, gateways)
    assert preview["behind_by"] == 1 and preview["at_risk"] == {"count": 1}
    transition = next(event for event in events(s) if event["kind"] == "authority.transition")
    assert transition["payload"]["at_risk"] == 1
    assert [e["payload"].get("text") for e in events(s) if e["kind"] == "message.user"][-2:] == [
        "user:late:0", "user:late:1"]


def test_a_computers_own_word_about_its_copy_changes_nothing_in_the_preview(gateways, monkeypatch):
    """Only a head the host signed counts a copy: a computer claiming more (in its answers and its fence
    receipt) neither shows messages as missing nor makes the owner's confirm stale."""
    s, p = gateways["s"], gateways["p"]
    real = succession.watermark_locked
    claims = {"n": 0}

    def watermark_locked(conn, room_id):
        mark = real(conn, room_id)
        if mark is not None and succession.local_install_id() == p.install_id:
            claims["n"] += 1
            return {**mark, "seq": mark["seq"] + 10 * claims["n"]}
        return mark

    monkeypatch.setattr(succession, "watermark_locked", watermark_locked)
    preview, result = continue_on(s, gateways)
    assert preview["behind_by"] == 0 and preview["at_risk"] == {"count": 0}
    assert result["state"] == "ok" and head(s)["authoritative"]


def admit_next(gateway, event_id, text):
    """The host's driver admits the turn its policy plans next, announcing it (``task.admitted``)."""
    import time
    from gateway import hosted_room_driver as driver
    from gateway.hosted_room_discussion import plan_next_task
    from gateway.hosted_room_policy_checkpoint import HostedRoomPolicyCheckpoint
    with gateway.acting():
        message(gateway.db, event_id, gateway=gateway.install_id, text=text,
                epoch=head(gateway)["authority_epoch"])
        room = {**rooms.room_state(gateway.db, room_id=ROOM),
                "authority_lineage": succession.authority_lineage(gateway.db, ROOM)}
        snapshot = HostedRoomPolicyCheckpoint(gateway.db).snapshot(room_id=ROOM, latest_seq=room["latest_seq"])
        task = plan_next_task(room, list(snapshot.events), local_profiles=move.policy_profiles(room),
                              initial_watermarks=snapshot.watermarks, freeze_input_context=True).task
        return driver.admit_task(gateway.db, task.identity, payload=task.payload, clock=time.time)


def test_a_successor_that_never_hosted_inherits_a_turn_the_host_left_unfinished(gateways):
    from gateway import hosted_room_driver as driver
    h, s = gateways["h"], gateways["s"]
    admitted = admit_next(h, "user:ask-writer", "@writer what do you think?")
    copy_to(h, s)
    preview, result = continue_on(s, gateways)  # S never ran a room driver
    assert preview["work"]["unknown"] == 1 and result["state"] == "ok" and result["this_install"]["role"] == "host"
    with s.acting():
        inherited = driver.list_tasks(s.db, room_id=ROOM, status="indeterminate")
    assert [task["identity"] for task in inherited] == [admitted["identity"]]
    # The new host's own driver keeps working: its work evidence has no claim lineage, and that's no error.
    assert admit_next(s, "user:ask-reviewer", "@reviewer and you?")["status"] == "queued"


def test_the_returning_host_steps_down_keeps_its_own_messages_apart_and_reports(gateways):
    from gateway import hosted_room_succession_return as returning
    h, s, p = gateways["h"], gateways["s"], gateways["p"]
    continue_on(s, gateways)
    # Unaware, the old host kept going while it was cut off.
    with h.acting():
        for number in range(2):
            message(h.db, f"user:alone:{number}", gateway=h.install_id)
        demoted = returning.check(context(h, gateways, down=()), ROOM)
    assert demoted["state"] == "stepped_down" and demoted["separate_events"] == 2
    assert head(h)["authoritative"] is False and head(h)["authority_gateway_id"] == s.install_id
    with h.acting():
        page = returning.branch_log(h.db, ROOM, demoted["branch_id"])
        status = status_module.status(context(h, gateways, down=()), ROOM)
    assert [event["payload"]["text"] for event in page["events"]] == ["user:alone:0", "user:alone:1"]
    assert status["state"] == "moved_away" and status["moved"]["separate_events"] == 2
    assert status["moved"]["to"] == {"install_id": s.install_id, "name": "S"}
    assert status["actions"] == [{"action": "open_on", "target": s.install_id}]
    assert fence.room_fence_state(h.runs.path, ROOM)["authority"] == {"epoch": 2, "install_id": s.install_id}
    with s.acting(), closing(rooms._read_connection(s.db)) as conn:
        reports = succession.load_record_locked(conn, ROOM, "move")["reports"]
    assert reports[0]["reporter_install_id"] == h.install_id and reports[0]["divergent"] == {"events": 2}
    # The old host's copy caught up with the new host and verified its transition itself.
    assert [event["kind"] for event in events(h, "hosted_room_replica_events")][-3:] == [
        "authority.transition", "custody.configured", "succession.state"]


def test_a_host_stepping_down_while_its_status_is_read_never_shows_as_serving(gateways, monkeypatch):
    """Status reads the room, then whether its host is paused. A host that steps down in between shows
    as the copy it became, never as serving its old epoch on a view that mixes before and after."""
    from gateway import hosted_room_succession_automatic as automatic
    from gateway import hosted_room_succession_return as returning
    h, s = gateways["h"], gateways["s"]
    continue_on(s, gateways)
    paused_view, stepped = automatic.paused_view, []

    def stepping_down_meanwhile(ctx, room_id):
        if not stepped:
            stepped.append(returning.check(context(h, gateways, down=()), room_id))
        return paused_view(ctx, room_id)

    monkeypatch.setattr(automatic, "paused_view", stepping_down_meanwhile)
    with h.acting():
        current = status_module.status(context(h, gateways, down=()), ROOM)
    assert stepped and stepped[0]["state"] == "stepped_down"
    assert (current["this_install"]["role"], current["state"]) == ("backup", "moved_away")


def test_a_host_stepping_down_stops_its_work_without_writing_to_the_log(gateways):
    from types import SimpleNamespace
    from gateway import hosted_room_succession_return as returning
    h, s = gateways["h"], gateways["s"]
    continue_on(s, gateways)
    stopped = []

    def stop_room(room_id, *, cancel_id):
        raise AssertionError("a Stop event would be written after the fork")

    service = SimpleNamespace(stop_room=stop_room, stop_work=lambda room_id, *, cancel_id: stopped.append(cancel_id),
                              wakeup=lambda: None)
    with h.acting():
        message(h.db, "user:alone", gateway=h.install_id)
        demoted = returning.check(context(h, gateways, down=(), service=service), ROOM)
    # Only what the host wrote while cut off is set aside: a false split would wait for the owner.
    assert stopped == ["authority-moved"] and demoted["separate_events"] == 1
    assert "room.stop_requested" not in [event["kind"] for event in events(h, "hosted_room_replica_events")]


def test_the_old_host_gives_the_new_one_a_copy_only_grant_so_it_hears_every_push(gateways):
    from gateway import hosted_room_custody as custody
    from gateway import hosted_room_succession_return as returning
    h, s = gateways["h"], gateways["s"]
    continue_on(s, gateways)
    with h.acting():
        returning.check(context(h, gateways, down=(), grants=True), ROOM)
        with closing(rooms._read_connection(h.db)) as conn:
            consents = succession.consents_locked(conn, ROOM)
        assert custody.local_consent(h.db, ROOM)  # it ran the group: it may continue it again
    assert [(item["member_id"], item["options"]["passive_only"]) for item in consents] == [
        (custody.CUSTODY_MEMBER_ID, True)]
    # Its report carried the grant: the new host pushes the history, and its lease requests, to it.
    with s.acting(), closing(rooms._read_connection(s.db)) as conn:
        routes = conn.execute(f"SELECT install_id, target_url, grant FROM {custody.ROUTES_TABLE} WHERE room_id=?",
                              (ROOM,)).fetchall()
    assert [tuple(row) for row in routes] == [
        (h.install_id, h.endpoint, f"grant:h:{custody.CUSTODY_MEMBER_ID}:{s.install_id}:2")]


def test_the_old_hosts_copy_only_grant_renews_like_any_custody_route(gateways):
    """Minted from the consent it keeps when it steps down, the old host's grant is renewed through its
    acknowledgments in its last week, like every copy-only grant, so the new host keeps copying there."""
    import time
    from gateway import hosted_room_peer as peer
    from gateway import hosted_room_succession_return as returning
    from gateway.hosted_room_replica_ingress import renewed_copy_grant
    h, s = gateways["h"], gateways["s"]
    continue_on(s, gateways)
    day, secret, issued = 24 * 3600, b"old-host-room-grant-secret-32-bytes", time.time()
    with h.acting():
        returning.check(context(h, gateways, down=()), ROOM)
        with closing(rooms._read_connection(h.db)) as conn:
            (consent,) = succession.consents_locked(conn, ROOM)
        options = consent["options"]  # as the continuation minter issues it to the new host at its epoch
        token = peer.issue_room_grant(
            secret, grant_id="grant-old-host", room_id=ROOM, home_install_id=s.install_id,
            authority_gateway_id=s.install_id, authority_epoch=2, member_id=consent["member_id"],
            target_install_id=h.install_id, target_profile=consent["target_profile"], issued_at=issued,
            permissions=peer.invitation_permissions(
                options["replication"], options["work_records"], passive_only=options["passive_only"],
                successor=options["successor"]),
            ttl_seconds=options["ttl_seconds"], status_ttl_seconds=options["status_ttl_seconds"])
        old = peer.decode_room_grant(secret, token, permission="status", now=issued)
        rooms.reserve_peer_room(h.db, claims=old, expires_at=old["status_expires_at"], now=issued)
        assert renewed_copy_grant(h.db, token=token, secret=secret, now=issued + 22 * day) is None
        renewed = renewed_copy_grant(h.db, token=token, secret=secret, now=issued + 24 * day)
    new = peer.decode_room_grant(secret, renewed, permission="replicate", now=issued + 40 * day)
    times = {"grant_id", "issued_at", "expires_at", "status_expires_at"}
    assert {k: v for k, v in new.items() if k not in times} == {k: v for k, v in old.items() if k not in times}
    assert new["status_expires_at"] - new["issued_at"] == 30 * day and "dispatch" not in new["permissions"]


def test_a_backups_copy_only_grant_becomes_the_new_hosts_custody_route(gateways):
    from types import SimpleNamespace
    from gateway import hosted_room_custody as custody
    s, p = gateways["s"], gateways["p"]
    continue_on(s, gateways)
    item = {"member_id": custody.CUSTODY_MEMBER_ID, "target_profile": "default", "grant": "grant-for-p",
            "catalog": {"installation_id": p.install_id}}
    own = {**item, "grant": "grant-for-s", "catalog": {"installation_id": s.install_id}}
    with s.acting():
        routes = move.register_routes(context(s, gateways, service=SimpleNamespace()), ROOM,
                                      {p.install_id: [item], s.install_id: [own]})
        with closing(rooms._read_connection(s.db)) as conn:
            saved = conn.execute(f"SELECT install_id, target_url, grant FROM {custody.ROUTES_TABLE} WHERE room_id=?",
                                 (ROOM,)).fetchall()
    assert routes == {f"{custody.CUSTODY_MEMBER_ID}:{p.install_id}": "registered"}
    assert [tuple(row) for row in saved] == [(p.install_id, p.endpoint, "grant-for-p")]


def test_set_aside_messages_go_with_their_group_and_the_owner_record_stays(gateways):
    from gateway import hosted_room_succession_return as returning
    h, s = gateways["h"], gateways["s"]
    continue_on(s, gateways)
    with h.acting():
        message(h.db, "user:alone", gateway=h.install_id)
        demoted = returning.check(context(h, gateways, down=()), ROOM)
        # Stepping down keeps the owner's record, so the owner can still read this copy and its branch.
        with closing(rooms._read_connection(h.db)) as conn:
            assert succession.owner_subject_locked(conn, ROOM) == "uid:501"
        assert returning.prune_branches(h.db) == 0
        assert returning.branch_log(h.db, ROOM, demoted["branch_id"])["events"]
        with pytest.raises(sqlite3.IntegrityError), rooms._transaction(h.db, immediate=True) as conn:
            conn.execute(f"DELETE FROM {returning.BRANCH_EVENTS} WHERE room_id=?", (ROOM,))
        with rooms._transaction(h.db, immediate=True) as conn:
            conn.execute("UPDATE hosted_room_replicas SET disbanded_at=? WHERE room_id=?", (1.0, ROOM))
        with pytest.raises(succession.SuccessionError):
            returning.branch_log(h.db, ROOM, demoted["branch_id"])
        assert returning.prune_branches(h.db) == 1 and returning.branches(h.db, ROOM) == []


def test_a_continuation_another_computer_holds_is_named_and_a_retry_moves_past_it(tmp_path):
    gateways = make_gateways(tmp_path, "h", "s", "t", "p")
    try:
        home_room(gateways, "h", successors=("s", "t"))
        for name in ("s", "t", "p"):
            copy_to(gateways["h"], gateways[name])
        s, t, p = gateways["s"], gateways["t"], gateways["p"]
        # t already fenced the host's epoch at p for itself.
        fence.fence_and_promise(p.runs.path, room_id=ROOM, fence_epoch=1, promise_epoch=2,
                                candidate_install_id=t.install_id)
        with s.acting():
            ctx = context(s, gateways, down=("h", "t"))
            preview = move.preview(ctx, ROOM, s.install_id)
            with pytest.raises(succession.SuccessionError) as refused:
                move.continue_here(ctx, ROOM, s.install_id, preview_id=preview["preview_id"], confirm=True)
            assert refused.value.reason == "room_authority_promised"
            assert refused.value.detail["other"] == {"install_id": t.install_id, "name": "T"}
            assert status_module.status(ctx, ROOM)["last_attempt"]["error"] == "room_authority_promised"
            preview = move.preview(ctx, ROOM, s.install_id)
            result = move.continue_here(ctx, ROOM, s.install_id, preview_id=preview["preview_id"], confirm=True)
        assert result["state"] == "ok" and head(s)["authority_epoch"] == 3
        assert fence.room_fence_state(p.runs.path, ROOM)["authority"] == {"epoch": 3, "install_id": s.install_id}
    finally:
        for gateway in gateways.values():
            gateway.close()


def test_stepping_down_is_one_transaction(gateways, monkeypatch):
    from gateway import hosted_room_succession_return as returning
    h, s = gateways["h"], gateways["s"]
    continue_on(s, gateways)
    with h.acting():
        message(h.db, "user:alone", gateway=h.install_id)
    before = events(h)

    def crash(*args, **kwargs):
        raise RuntimeError("power cut")

    monkeypatch.setattr(succession, "record_lineage_locked", crash)
    with h.acting(), pytest.raises(RuntimeError):
        transition = next(event for event in events(s) if event["kind"] == "authority.transition")
        fork = next(event for event in events(s) if event["seq"] == transition["seq"] - 1)
        returning.demote_to_custody(h.db, room_id=ROOM, transition=transition, fork_event=fork)
    assert head(h)["authoritative"] and events(h) == before
    assert returning.branches(h.db, ROOM) == []
