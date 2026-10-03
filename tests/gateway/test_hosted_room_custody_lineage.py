"""A copy follows its Group Chat across verified changes of host, verified in page order.

Several installations in one process, each with its own root (RoomLink secret, install id and room
identity key) and room store. ``verify`` stands in for the succession verifier (#105197): it accepts
a continuation signed by a successor that the configuration held at that point names, coming from
that configuration's host.
"""

import copy
import json
import os
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest

from gateway import hosted_room_custody as custody
from gateway import hosted_room_identity as identity
from gateway import hosted_room_replicas as replicas
from gateway import hosted_room_safety as safety
from gateway import hosted_rooms as rooms
from gateway.hosted_rooms_common import open_sqlite

ROOM = "room"
MEMBERS = [{"member_id": "writer", "handle": "writer", "profile": "default",
            "target": {"kind": "local", "profile": "default"}}]
DOMAIN = b"hermes.test.continuation.v1"


@dataclass
class Install:
    name: str
    home: Path
    install_id: str
    public_key: str

    @property
    def db(self) -> Path:
        return self.home / "state.db"

    @property
    def endpoint(self) -> str:
        return f"https://{self.name}.example.test"


@contextmanager
def acting(install):
    previous = os.environ.get("HERMES_HOME")
    os.environ["HERMES_HOME"] = str(install.home)
    try:
        yield install
    finally:
        os.environ["HERMES_HOME"] = previous


@pytest.fixture
def net(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    installs = {}
    for name in ("home", "second", "third", "fresh"):
        (tmp_path / name).mkdir()
        with acting(Install(name, tmp_path / name, "", "")) as bare:
            installs[name] = Install(name, bare.home, rooms.local_authority_gateway_id(), identity.local_public_key())
    return installs


def message(install, event_id, epoch):
    with acting(install):
        rooms.append_event(install.db, room_id=ROOM, event_id=event_id, kind="message.user",
                           actor={"kind": "user", "id": "owner"}, payload={"text": event_id},
                           authority_gateway_id=install.install_id, authority_epoch=epoch)


def activity(install, event_id, epoch, *, actor=None):
    """A gateway-actor event: it names the host of its span."""
    with acting(install):
        rooms.append_event(install.db, room_id=ROOM, event_id=event_id, kind="room.activity",
                           actor={"kind": "gateway", "id": actor or install.install_id}, payload={"state": "idle"},
                           authority_gateway_id=install.install_id, authority_epoch=epoch)


def enroll(host, other, *, successor=True):
    with acting(host):
        custody.enroll_custodian(host.db, room_id=ROOM, install_id=other.install_id, public_key=other.public_key,
                                 endpoint=other.endpoint, name=other.name, role="custodian", active=True,
                                 allowed=successor, designated=successor)


def configure(host):
    with acting(host):
        return custody.maintain_configuration(host.db, room_id=ROOM, local_gateway_id=host.install_id,
                                              public_key=host.public_key, endpoint=host.endpoint, name=host.name,
                                              owner_name="Dana")


def watermark(install):
    if not install.db.exists():
        return None
    with acting(install), rooms._transaction(install.db, immediate=True) as conn:
        return custody.custody_watermark_locked(conn, ROOM, store=False)


def page(source, *, after=0, limit=rooms.MAX_LOG_LIMIT):
    with acting(source), closing(open_sqlite(source.db)) as conn:
        return custody.read_copy_page(conn, ROOM, after_seq=after, limit=limit)


def copy_to(source, target, *, after=None, limit=rooms.MAX_LOG_LIMIT, verify=None):
    """Store the source's history in the target's copy, as the source's pages carry it."""
    held = watermark(target)
    fetched = page(source, after=held["seq"] if after is None and held else after or 0, limit=limit)
    with acting(target):
        return replicas.ingest_page(target.db, room_id=ROOM, room_name=fetched["room_name"],
                                    members=fetched["members"], page=fetched["page"], _verify_transition=verify)


def continuation(successor, *, from_host, from_epoch, to_epoch):
    statement = {"room_id": ROOM, "from_epoch": from_epoch, "to_epoch": to_epoch,
                 "successor_gateway_id": successor.install_id, "from_host": from_host.install_id}
    with acting(successor):
        proof = {**statement, "signature": identity.sign(DOMAIN, statement)}
    return {"proof_kind": "attested", "proof_digest": safety.transition_proof_digest(proof), "proof": proof}


SEEN: list = []


def verify(conn, event):
    """Accept a continuation by a successor of the configuration held here, from its host."""
    payload = json.loads(event["payload_json"])
    proof = payload["proof"]
    configuration = custody.configuration_locked(conn, event["room_id"])
    host = next((c["install_id"] for c in configuration["custodians"] if c["role"] == "authority"), None)
    successors = {c["install_id"] for c in configuration["custodians"] if c["successor"]}
    statement = {key: value for key, value in proof.items() if key != "signature"}
    SEEN.append((event["seq"], configuration["configuration_seq"]))
    if (payload["proof_digest"] != safety.transition_proof_digest(proof) or proof["from_host"] != host
            or proof["successor_gateway_id"] not in successors
            or not identity.verify_locked(conn, event["room_id"], proof["successor_gateway_id"], DOMAIN, statement,
                                          proof["signature"])):
        raise PermissionError("the continuation does not verify against the configuration held here")
    safety.mark_verified_transition(
        conn, room_id=event["room_id"], from_epoch=payload["from_epoch"], to_epoch=payload["to_epoch"],
        successor_gateway_id=payload["successor_gateway_id"], proof_kind=payload["proof_kind"],
        proof_digest=payload["proof_digest"])


def move(successor, from_host, *, from_epoch, to_epoch):
    """The successor continues the group on its copy and records itself as the host."""
    with acting(successor):
        replicas.promote_replica(successor.db, room_id=ROOM, to_epoch=to_epoch, transition=continuation(
            successor, from_host=from_host, from_epoch=from_epoch, to_epoch=to_epoch))
        with rooms._transaction(successor.db, immediate=True) as conn:
            return custody.reconfigure_after_transition_locked(
                conn, ROOM, successor=successor.install_id, previous_host=from_host.install_id)


def head(install):
    with acting(install):
        state = replicas.replica_state(install.db, room_id=ROOM)
    return state["authority"]["gateway_id"], state["authority"]["epoch"], state["safety_status"]


@pytest.fixture
def two_moves(net):
    """home hosts at epoch 1; second continues at 2 and adds third; third continues at 3."""
    home, second, third = net["home"], net["second"], net["third"]
    with acting(home):
        rooms.create_room(home.db, room_id=ROOM, name="Workshop", members=MEMBERS, authority_gateway_id=home.install_id)
    message(home, "m1", 1)
    activity(home, "a1", 1)
    enroll(home, second)
    configure(home)
    message(home, "m2", 1)
    copy_to(home, second)
    move(second, home, from_epoch=1, to_epoch=2)
    enroll(second, third)
    configure(second)  # the first configuration that names third, and its key
    message(second, "m3", 2)
    activity(second, "a2", 2)
    copy_to(second, third, verify=verify)  # a new copy of a room that already moved once
    moved = move(third, second, from_epoch=2, to_epoch=3)
    message(third, "m4", 3)
    SEEN.clear()
    return net, moved


def test_a_new_copy_follows_two_moves_verified_in_page_order(two_moves):
    net, moved = two_moves
    home, second, third, fresh = (net[name] for name in ("home", "second", "third", "fresh"))
    # The second move's previous-host check passed on third: second hosted the configuration it held.
    assert [(c["install_id"], c["role"], c["successor"]) for c in moved["custodians"]] == sorted(
        [(home.install_id, "custodian", False), (second.install_id, "custodian", False),
         (third.install_id, "authority", False)])
    kinds = [event["kind"] for event in page(third)["page"]["events"]]
    first, second_move = (index for index, kind in enumerate(kinds) if kind == "authority.transition")
    assert "custody.configured" in kinds[first + 1:second_move]
    # One page holds both moves. The second verifies only against the configuration that the same
    # page recorded after the first, whose key for third is pinned only there.
    result = copy_to(third, fresh, verify=verify)
    assert result["stored_seq"] == len(kinds) and result["caught_up"]
    assert [configuration_seq for _, configuration_seq in SEEN][1] > SEEN[0][0]
    assert head(fresh) == (third.install_id, 3, "passive")
    assert watermark(fresh) == watermark(third)
    with acting(fresh), closing(open_sqlite(fresh.db)) as conn:
        uses = conn.execute("SELECT to_epoch FROM hosted_room_verified_transition_uses WHERE room_id=? ORDER BY to_epoch",
                            (ROOM,)).fetchall()
    assert [row[0] for row in uses] == [2, 3]


def test_a_copy_more_than_a_page_behind_follows_the_new_host(two_moves):
    net, _ = two_moves
    home, third, fresh = net["home"], net["third"], net["fresh"]
    copy_to(home, fresh, limit=1)  # a custodian that kept one event, then went offline
    assert head(fresh) == (home.install_id, 1, "passive")
    # The new host's pages relay the old host's history before its own move: the copy follows
    # them without moving, then moves with each verified transition.
    while not copy_to(third, fresh, limit=2, verify=verify)["caught_up"]:
        assert head(fresh)[2] == "passive"
    assert head(fresh) == (third.install_id, 3, "passive")
    assert watermark(fresh) == watermark(third)


def test_a_change_of_host_needs_a_verified_transition(two_moves):
    net, _ = two_moves
    third, fresh = net["third"], net["fresh"]
    with pytest.raises(replicas.ReplicaLineageUnverifiedError, match="first authority epoch"):
        copy_to(third, fresh)  # no verifier: a moved room's history is refused outright
    with pytest.raises(PermissionError):
        copy_to(third, fresh, verify=lambda conn, event: (_ for _ in ()).throw(PermissionError("refused")))
    with pytest.raises(replicas.ReplicaLineageUnverifiedError, match="not verified"):
        copy_to(third, fresh, verify=lambda conn, event: None)  # a verifier that marks nothing
    with acting(fresh), pytest.raises(replicas.ReplicaError, match="not found"):
        replicas.replica_state(fresh.db, room_id=ROOM)  # each refusal left nothing behind
    # A host's own events must name it: a later span written as another host is refused.
    forged = page(third)
    later = next(event for event in forged["page"]["events"] if event["event_id"] == "m4")
    forged["page"]["events"].insert(forged["page"]["events"].index(later), {
        **copy.deepcopy(later), "event_id": "forged", "kind": "room.activity",
        "actor": {"kind": "gateway", "id": net["second"].install_id}, "payload": {"state": "idle"}})
    for index, event in enumerate(forged["page"]["events"], 1):
        event["seq"] = index
    forged["page"].update(cursor=len(forged["page"]["events"]), latest_seq=len(forged["page"]["events"]))
    with acting(fresh), pytest.raises(replicas.ReplicaError, match="gateway actor"):
        replicas.ingest_page(fresh.db, room_id=ROOM, room_name=forged["room_name"], members=forged["members"],
                             page=forged["page"], _verify_transition=verify)


def test_an_enrolled_copy_follows_only_verified_successors(net, tmp_path):
    import hashlib
    from gateway import hosted_room_replica_retirement as retirement
    home, second = net["home"], net["second"]
    roster = json.dumps(MEMBERS)
    db = tmp_path / "enrolled.db"
    with rooms._transaction(db, immediate=True) as conn:
        retirement._initialize(conn)
        conn.execute(f"""INSERT INTO {retirement.ENROLLMENT_TABLE} (enrollment_id, room_id, authority_gateway_id,
            authority_epoch, target_install_id, roster_sha256, commitment, created_at) VALUES (?,?,?,?,?,?,?,?)""",
                     ("enrollment", ROOM, home.install_id, 1, second.install_id,
                      hashlib.sha256(roster.encode()).hexdigest(), "commitment", 1.0))
        scope = dict(room_id=ROOM, members_json=roster)
        matches = retirement.copy_scope_matches_locked
        assert matches(conn, **scope, authority_gateway_id=home.install_id, authority_epoch=1)
        later = dict(scope, authority_gateway_id=second.install_id, authority_epoch=2)
        assert not matches(conn, **later)  # a later host the copy has not verified
        assert matches(conn, **later, verified_head=(second.install_id, 2))  # reached through a verified move
        assert matches(conn, **later, verified_head=(home.install_id, 1))  # still relaying the enrolled host's span
        assert not matches(conn, **later, verified_head=(second.install_id, 1))
        assert not matches(conn, **dict(later, members_json="[]"), verified_head=(second.install_id, 2))
