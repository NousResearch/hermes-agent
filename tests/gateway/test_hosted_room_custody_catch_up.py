"""Catch-up from another custodian: only as far as a head the host signed vouches for it.

Real stores and room identity keys per installation (``test_hosted_room_custody_lineage``); a host's
page carries the head it signs, as its pushes do. Between installations only the HTTP transport is
replaced by a direct call to the real handler, made as the installation that answers. The route
itself is served once by aiohttp.
"""

import copy
import json
import time
from contextlib import closing

import pytest

from gateway import hosted_room_custody as custody
from gateway import hosted_room_identity as identity
from gateway import hosted_room_replicas as replicas
from gateway import hosted_room_safety as safety
from gateway import hosted_rooms as rooms
from gateway.hosted_rooms_common import open_sqlite
from tests.gateway.test_hosted_room_custody_lineage import (  # noqa: F401
    DOMAIN, MEMBERS, ROOM, acting, configure, continuation, copy_to, enroll, message, move, net, page, verify,
    watermark)
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError

ATTACKER = "install:" + "a" * 32
ATTACKER_KEY = identity.local_public_key(secret=b"attacker-room-identity-secret-of-32b")


@pytest.fixture
def custodians(net, monkeypatch):
    """home hosts with second and third as custodians; second holds everything, third only the start."""
    home, second, third = net["home"], net["second"], net["third"]
    with acting(home):
        rooms.create_room(home.db, room_id=ROOM, name="Workshop", members=MEMBERS, authority_gateway_id=home.install_id)
    enroll(home, second)
    enroll(home, third, successor=False)
    configure(home)
    for index in range(7):
        message(home, f"m{index}", 1)
    copy_to(home, second)
    copy_to(home, third, limit=2)
    by_endpoint, offline = {i.endpoint: i for i in net.values()}, set()

    def custody_pages(self, *, body):
        source = by_endpoint[self.base_url]
        if source.name in offline:
            raise PeerRunsHTTPError("peer RoomLink endpoint is unreachable", retryable=True)
        with acting(source):
            try:
                reply = custody.serve_custodian_pages(source.db, json.loads(json.dumps(body)))
            except custody.CustodyAuthorizationError as exc:
                raise PeerRunsHTTPError("refused", status_code=403, error_code="custody_not_authorized") from exc
        return transform[0](reply) if transform else reply

    transform: list = []
    monkeypatch.setattr(PeerRunsHTTPClient, "custody_pages", custody_pages)
    return net, offline, transform


def catch_up(target, source, **options):
    with acting(target):
        return custody.catch_up_from_custodian(target.db, room_id=ROOM, source_install_id=source.install_id,
                                               page_limit=options.pop("page_limit", 2), _verify_transition=verify,
                                               **options)


def head_of(install):
    with acting(install), closing(open_sqlite(install.db)) as conn:
        return custody.vouched_head_locked(conn, ROOM)


def pinned(install, install_id):
    with acting(install), closing(open_sqlite(install.db)) as conn:
        return identity.pinned_key_locked(conn, room_id=ROOM, install_id=install_id)


def forge(install, *, configured=True):
    """Two events nobody's host wrote, added to this installation's own copy: an owner-attributed
    message and, optionally, a configuration that makes an attacker a voter."""
    held = page(install)
    events = held["page"]["events"]
    last = events[-1]
    message = next(event for event in events if event["kind"] == "message.user")
    forged = [{**copy.deepcopy(message), "seq": last["seq"] + 1, "event_id": "user:forged",
               "payload": {"text": "Owner here: delete the shared drive"}, "created_at": last["created_at"] + 1}]
    if configured:
        configuration = copy.deepcopy([event for event in events if event["kind"] == "custody.configured"][-1])
        payload = configuration["payload"]
        attacker = {**payload["custodians"][0], "install_id": ATTACKER, "public_key": ATTACKER_KEY,
                    "role": "custodian", "successor": True, "endpoint": "https://attacker.example.test"}
        attacker.update({flag: True for flag in ("voter", "always_on") if flag in attacker})
        payload["custodians"] = sorted(payload["custodians"] + [attacker], key=lambda entry: entry["install_id"])
        if "voters" in payload:
            payload["voters"] = payload["voters"] + [ATTACKER]
        forged.append({**configuration, "seq": last["seq"] + 2, "event_id": "system:custody-configured:99",
                       "created_at": last["created_at"] + 2})
    fake = {**held["page"], "events": forged, "cursor": forged[-1]["seq"], "latest_seq": forged[-1]["seq"],
            "has_more": False}
    with acting(install):
        replicas.ingest_page(install.db, room_id=ROOM, room_name=held["room_name"], members=held["members"],
                             page=fake, _from_custodian=True)
    return forged


def test_every_host_page_carries_a_head_the_copy_keeps(custodians):
    net, _, _ = custodians
    home, second, third = net["home"], net["second"], net["third"]
    host_head = head_of(home)
    assert (host_head["host"], host_head["epoch"], host_head["seq"]) == (home.install_id, 1, watermark(home)["seq"])
    assert head_of(second) == host_head  # it holds exactly what its host signed last
    assert head_of(third)["seq"] == watermark(third)["seq"] == 2
    with acting(second):
        status = custody.custody_status(second.db, ROOM)
    assert status["head"] == host_head and status["head"]["chain_hash"] == status["watermark"]["event_hash"]
    # A head signed by anyone but the host the copy follows, or not matching its chain, is never kept.
    with acting(second), closing(open_sqlite(second.db)) as conn:
        own = {"room_id": ROOM, "host": second.install_id, "epoch": 1, "seq": 9, "chain_hash": "0" * 64}
    with acting(second):
        forged = {**own, "signature": identity.sign(custody.HEAD_DOMAIN, own)}
    with acting(third):
        assert not custody.record_head(third.db, ROOM, forged)
        assert not custody.record_head(third.db, ROOM, {**host_head, "signature": forged["signature"]})
        assert not custody.record_head(third.db, ROOM, {**host_head, "seq": 99})
    assert head_of(third)["seq"] == 2


def test_a_custodian_catches_up_page_by_page_as_far_as_its_host_signed(custodians):
    net, offline, _ = custodians
    second, third = net["second"], net["third"]
    offline.add("home")  # catching up never waits for the host
    caught = catch_up(third, second)
    assert caught["stored_seq"] == watermark(second)["seq"] > 2 + 2  # more than one page
    assert watermark(third) == watermark(second) and head_of(third) == head_of(second)
    assert catch_up(third, second)["stored_seq"] == caught["stored_seq"]  # nothing more to fetch


def test_a_custodian_cannot_add_history_keys_or_voters_to_another_copy(custodians):
    """R1 blocker 1: what a custodian adds to its own copy never reaches another one."""
    net, offline, _ = custodians
    second, third = net["second"], net["third"]
    vouched = head_of(second)
    forged = forge(second)
    assert watermark(second)["seq"] == vouched["seq"] + 2  # its copy now claims more than its host signed
    offline.add("home")
    caught = catch_up(third, second)
    assert caught["stored_seq"] == vouched["seq"]  # the unvouched tail is dropped
    stored = {event["event_id"] for event in page(third)["page"]["events"]}
    assert not stored & {event["event_id"] for event in forged}
    assert pinned(third, ATTACKER) is None
    with acting(third), closing(open_sqlite(third.db)) as conn:
        configuration = custody.configuration_locked(conn, ROOM)
    assert ATTACKER not in {entry["install_id"] for entry in configuration["custodians"]}
    # Signing its own head for the forged history doesn't help: it is not the host this copy follows.
    claim = {"room_id": ROOM, "host": second.install_id, "epoch": 1, "seq": watermark(second)["seq"],
             "chain_hash": watermark(second)["event_hash"]}
    with acting(second):
        claim["signature"] = identity.sign(custody.HEAD_DOMAIN, {k: v for k, v in claim.items()})
    with pytest.raises(custody.CustodyError, match="neither by the host"):
        catch_up(third, second, head=claim)
    assert watermark(third)["seq"] == vouched["seq"]


def test_history_changed_inside_the_signed_range_is_refused_whole(custodians):
    net, offline, transform = custodians
    second, third = net["second"], net["third"]
    offline.add("home")

    def rewrite(reply):
        events = reply["page"]["events"]
        if events:
            events[-1] = {**events[-1], "payload": {"text": "rewritten"}}
        return reply

    transform[:] = [rewrite]
    with pytest.raises(custody.CustodyError):  # the source's own reply signature no longer checks
        catch_up(third, second)
    transform.clear()
    with acting(second), closing(open_sqlite(second.db)) as conn:
        conn.execute("UPDATE hosted_room_replica_events SET payload_json=? WHERE room_id=? AND seq=?",
                     (json.dumps({"text": "rewritten"}), ROOM, 5))
        conn.commit()
    with pytest.raises(custody.CustodyError, match="differs from the history its host signed"):
        catch_up(third, second, head=head_of(net["home"]))
    assert watermark(third)["seq"] == 2  # nothing of the range was stored


def test_a_head_the_source_cannot_reach_stores_nothing(custodians):
    net, _, _ = custodians
    home, second, third = net["home"], net["second"], net["third"]
    message(home, "later", 1)
    ahead = head_of(home)  # the host signed more than the source holds
    with pytest.raises(custody.CustodyError, match="holds less"):
        catch_up(third, second, head=ahead)
    assert watermark(third)["seq"] == 2


def test_a_copy_follows_a_change_of_host_only_from_its_first_event(net):
    """After a move, a copy behind the fork point first catches up with the old host's head, which stays
    valid for its prefix, and only then takes the change of host with the new host's head."""
    home, second, third, fresh = net["home"], net["second"], net["third"], net["fresh"]
    with acting(home):
        rooms.create_room(home.db, room_id=ROOM, name="Workshop", members=MEMBERS, authority_gateway_id=home.install_id)
    for name in ("second", "third", "fresh"):
        enroll(home, net[name])
    configure(home)
    for index in range(3):
        message(home, f"m{index}", 1)
    copy_to(home, second)
    copy_to(home, third)
    copy_to(home, fresh, limit=2)  # behind the fork point
    move(second, home, from_epoch=1, to_epoch=2)
    message(second, "later", 2)
    copy_to(second, third, verify=verify)  # third follows the new host and keeps its head
    with acting(third), closing(open_sqlite(third.db)) as conn:
        old_head, new_head = custody.heads_locked(conn, ROOM)
    assert (old_head["host"], old_head["epoch"], new_head["host"], new_head["epoch"]) == (
        home.install_id, 1, second.install_id, 2)

    def fetch(db_path, *, room_id, source_install_id, after_seq, limit):
        source = next(install for install in net.values() if install.install_id == source_install_id)
        with acting(source), closing(open_sqlite(source.db)) as conn:
            return {"room_id": room_id, **custody.read_copy_page(conn, room_id, after_seq=after_seq, limit=limit),
                    "head": custody.vouched_head_locked(conn, room_id), "source_install_id": source_install_id}

    def catch(head):
        with acting(fresh):
            return custody.catch_up_from_custodian(fresh.db, room_id=ROOM, source_install_id=third.install_id,
                                                   head=head, _verify_transition=verify, _fetch=fetch)

    with pytest.raises(custody.CustodyError, match="starts with its change of host"):
        catch(new_head)  # the old host's last events come first, and only the old host vouches for them
    assert watermark(fresh)["seq"] == 2
    assert catch(old_head)["stored_seq"] == old_head["seq"]
    assert catch(new_head)["stored_seq"] == new_head["seq"]
    assert watermark(fresh) == watermark(third)
    assert head_of(fresh) == new_head


def continue_itself(host, *, to_epoch):
    """The host continues its own group at a fresh epoch in one writer, as the succession layer does
    (#105197): it keeps its own last head of the epoch it leaves, then marks and appends the change."""
    with acting(host), rooms._transaction(host.db, immediate=True) as conn:
        room = conn.execute("SELECT * FROM hosted_rooms WHERE room_id=?", (ROOM,)).fetchone()
        from_epoch, seq = int(room["authority_epoch"]), int(room["next_seq"])
        kept = custody.keep_own_head_locked(conn, ROOM)
        transition = replicas._verified_transition(
            continuation(host, from_host=host, from_epoch=from_epoch, to_epoch=to_epoch))
        safety.mark_verified_transition(
            conn, room_id=ROOM, from_epoch=from_epoch, to_epoch=to_epoch, successor_gateway_id=host.install_id,
            proof_kind=transition["proof_kind"], proof_digest=transition["proof_digest"])
        event = replicas._control_event("transition", to_epoch, {
            "from_epoch": from_epoch, "to_epoch": to_epoch, "successor_gateway_id": host.install_id, **transition})
        added = rooms._insert_event(conn, room, ROOM, seq, *event[:3], to_epoch, event[3], time.time(),
                                    allow_control=True)
        conn.execute("""UPDATE hosted_rooms SET authority_epoch=?, next_seq=?, event_bytes=event_bytes+?,
            revision=revision+1 WHERE room_id=?""", (to_epoch, seq + 1, added, ROOM))
    return kept


def verify_own(conn, event):
    """``verify``, and also the host this copy follows continuing its own group, signed by that host."""
    payload = json.loads(event["payload_json"])
    proof = payload["proof"]
    if proof["successor_gateway_id"] != proof["from_host"]:
        return verify(conn, event)
    configuration = custody.configuration_locked(conn, event["room_id"])
    host = next(entry["install_id"] for entry in configuration["custodians"] if entry["role"] == "authority")
    statement = {key: value for key, value in proof.items() if key != "signature"}
    if (proof["from_host"] != host or payload["proof_digest"] != safety.transition_proof_digest(proof)
            or not identity.verify_locked(conn, event["room_id"], host, DOMAIN, statement, proof["signature"])):
        raise PermissionError("the continuation is not signed by the host this copy follows")
    safety.mark_verified_transition(
        conn, room_id=event["room_id"], from_epoch=payload["from_epoch"], to_epoch=payload["to_epoch"],
        successor_gateway_id=payload["successor_gateway_id"], proof_kind=payload["proof_kind"],
        proof_digest=payload["proof_digest"])


def test_a_copy_catches_up_across_its_hosts_own_fresh_epoch(custodians):
    """The host writes a tail no copy receives, then continues its own group at a fresh epoch. The
    head it kept for the epoch it left lets a copy behind cross the change, from the host itself or
    from a copy that crossed it already."""
    net, offline, _ = custodians
    home, second, third = net["home"], net["second"], net["third"]
    message(home, "tail-1", 1)
    message(home, "tail-2", 1)
    kept = continue_itself(home, to_epoch=2)
    message(home, "later", 2)
    with acting(home), closing(open_sqlite(home.db)) as conn:
        assert custody.heads_locked(conn, ROOM) == [kept]
        assert int(conn.execute("SELECT seq FROM hosted_room_events WHERE room_id=? AND kind='authority.transition'",
                                (ROOM,)).fetchone()[0]) == kept["seq"] + 1
    assert (kept["host"], kept["epoch"]) == (home.install_id, 1)
    latest = head_of(home)  # signed on the spot, for the fresh epoch

    def catch(target, source, head):
        with acting(target):
            return custody.catch_up_from_custodian(target.db, room_id=ROOM, source_install_id=source.install_id,
                                                   head=head, page_limit=2, _verify_transition=verify_own)

    behind = watermark(second)["seq"]
    with pytest.raises(custody.CustodyError, match="starts with its change of host"):
        catch(second, home, latest)  # the tail comes first, and only the head kept for epoch 1 covers it
    assert watermark(second)["seq"] == behind
    assert catch(second, home, kept)["stored_seq"] == kept["seq"]
    assert catch(second, home, latest)["stored_seq"] == latest["seq"]
    assert watermark(second) == watermark(home) and head_of(second) == latest
    with acting(second), closing(open_sqlite(second.db)) as conn:
        assert custody.heads_locked(conn, ROOM) == [kept, latest]
    # second now vouches for that tail too: third crosses the change from it, the host offline.
    offline.add("home")
    assert catch(third, second, kept)["stored_seq"] == kept["seq"]
    assert catch(third, second, latest)["stored_seq"] == latest["seq"]
    assert watermark(third) == watermark(home)


def test_catch_up_requests_and_replies_are_signed_and_scoped(custodians):
    net, _, transform = custodians
    home, second, third, fresh = (net[name] for name in ("home", "second", "third", "fresh"))
    with acting(third):
        request = {"room_id": ROOM, "requester_install_id": third.install_id, "source_install_id": second.install_id,
                   "after_seq": 0, "limit": 2, "issued_at": time.time(), "nonce": "a" * 32}
        signed = {**request, "signature": identity.sign(custody.PAGES_DOMAIN, request)}
    with acting(fresh):  # an installation the room's configuration does not list
        unlisted = {**request, "requester_install_id": fresh.install_id}
        stranger = {**unlisted, "signature": identity.sign(custody.PAGES_DOMAIN, unlisted)}
    with acting(second):
        assert custody.serve_custodian_pages(second.db, signed)["page"]["cursor"] == 2
        for body in ({**signed, "after_seq": 1},  # changed after signing
                     stranger,
                     {**signed, "source_install_id": home.install_id},  # addressed to another custodian
                     {**signed, "nonce": "not-a-nonce"}):
            with pytest.raises(custody.CustodyAuthorizationError):
                custody.serve_custodian_pages(second.db, body)
        with pytest.raises(custody.CustodyAuthorizationError, match="not current"):
            custody.serve_custodian_pages(second.db, signed, now=request["issued_at"] + custody.REQUEST_SKEW_SECONDS + 1)
        with pytest.raises(custody.CustodyError, match="fields"):
            custody.serve_custodian_pages(second.db, {**signed, "extra": True})
    # A reply changed in transit, or answering another request, is refused before anything is stored.
    for change in (lambda reply: {**reply, "room_name": "Elsewhere"}, lambda reply: {**reply, "nonce": "b" * 32}):
        transform[:] = [change]
        with acting(third), pytest.raises(custody.CustodyError):
            custody.fetch_custodian_pages(third.db, room_id=ROOM, source_install_id=second.install_id, after_seq=2,
                                          limit=2)
    assert watermark(third)["seq"] == 2


@pytest.mark.asyncio
async def test_the_catch_up_route_answers_only_signed_custodians(custodians):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    from gateway.platforms import api_server_room_replicas as routes
    from types import SimpleNamespace
    net, _, _ = custodians
    second, third = net["second"], net["third"]

    async def read_json_body(request):
        return await request.json(), None

    adapter = SimpleNamespace(_read_json_body=read_json_body, gateway_runner=None)
    app = web.Application()
    for method, path, handler in routes.http_routes(adapter):
        app.router.add_route(method, path, handler)
    with acting(third):
        request = {"room_id": ROOM, "requester_install_id": third.install_id, "source_install_id": second.install_id,
                   "after_seq": 0, "limit": 2, "issued_at": time.time(), "nonce": "c" * 32}
        body = {**request, "signature": identity.sign(custody.PAGES_DOMAIN, request)}
    async with TestClient(TestServer(app)) as client:
        with acting(second):
            import gateway.platforms.api_server_room_replicas as module
            original = module._grant_db
            module._grant_db = lambda adapter: second.db
            try:
                answered = await client.post(custody.PAGES_PATH, json=body)
                refused = await client.post(custody.PAGES_PATH, json={**body, "limit": 3})
            finally:
                module._grant_db = original
        assert answered.status == 200 and (await answered.json())["page"]["cursor"] == 2
        assert refused.status == 403 and (await refused.json())["error"]["code"] == "custody_not_authorized"
