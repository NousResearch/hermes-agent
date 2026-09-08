"""Contiguous passive lineage, not authorization to execute or proof of no lost tail."""

import copy
import hashlib
import json

import pytest

from gateway import hosted_room_replicas as replicas
from gateway import hosted_room_replica_retirement as retirement
from gateway import hosted_rooms as rooms
from tests.gateway.test_hosted_room_replica_retirement import (
    HOME, TARGET, MEMBERS, SECRET, pair, copied_prefix, enroll,  # noqa: F401
)

SUCCESSOR = "install:successor"
FINAL = "install:final"


def transfer(source, old=HOME, epoch=1, new=SUCCESSOR):
    result = rooms.claim_authority(
        source, room_id="room", expected_gateway_id=old, expected_epoch=epoch,
        new_gateway_id=new, event_id=f"claim-{epoch + 1}",
    )
    return {"gateway_id": new, "epoch": epoch + 1, "from_seq": result["claim_event"]["seq"]}


def manifest(spans):
    encoded = json.dumps(spans, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def setup_successor(source, target, spans, *, old_id=None):
    entry = retirement.prepare_home_enrollment(
        source, room_id="room", target_install_id=TARGET, endpoint="https://participant.example",
        local_gateway_id=spans[-1]["gateway_id"], secret=SECRET,
        enrollment_id=f"enroll-{len(spans)}", replace_enrollment_id=old_id,
    )
    assert entry["version"] == 2
    assert entry["lineage_sha256"] == manifest(spans)
    result = retirement.enroll_target(
        target, enrollment=entry, target_install_id=TARGET,
        authority_history=spans, expected_enrollment_id=old_id,
    )
    assert retirement.confirm_home_enrollment(source, enrollment_id=entry["enrollment_id"], proof=result)
    return entry


def ingest(target, page):
    return replicas.ingest_page(target, room_id="room", room_name="Workshop", members=MEMBERS, page=page)


def page_v2(source, spans, **kwargs):
    page = rooms.read_events(source, room_id="room", **kwargs)
    return {**page, "replica_version": 2, "lineage_sha256": manifest(spans)}


@pytest.mark.parametrize("cut", [0, 1, 3, 4])
def test_contiguous_claims_are_verified_only_when_received(pair, cut):
    source, target = pair
    spans = [{"gateway_id": HOME, "epoch": 1, "from_seq": 0}, transfer(source)]
    spans.append(transfer(source, SUCCESSOR, 2, FINAL))
    setup_successor(source, target, spans)
    if cut:
        first = page_v2(source, spans, limit=cut)
        ack = ingest(target, first)
        assert ack["lineage_status"] == "pending"
        state = replicas.replica_state(target, room_id="room")
        expected = HOME if cut < spans[1]["from_seq"] else SUCCESSOR
        assert state["authority"]["gateway_id"] == expected
        assert state["source_authority"] == {"gateway_id": FINAL, "epoch": 3}
    page = page_v2(source, spans, since_seq=cut)
    ack = ingest(target, page)
    assert ack["replica_version"] == 2 and ack["lineage_sha256"] == manifest(spans)
    assert ack["lineage_status"] == "verified"
    assert ingest(target, page)["ingested"] == 0  # lost ACK / reopened SQLite
    state = replicas.replica_state(target, room_id="room")
    assert state["authority"] == state["source_authority"] == {"gateway_id": FINAL, "epoch": 3}
    assert state["safety_status"] == "passive"
    assert state["authority_history"] == spans
    with rooms._transaction(source) as src, rooms._transaction(target) as dst:
        fields = "seq,event_id,kind,actor_json,authority_epoch,payload_json,created_at"
        expected = src.execute(f"SELECT {fields} FROM hosted_room_events ORDER BY seq").fetchall()
        actual = dst.execute(f"SELECT {fields} FROM hosted_room_replica_events ORDER BY seq").fetchall()
        assert [tuple(r) for r in actual] == [tuple(r) for r in expected]


@pytest.mark.parametrize("damage", ["actor", "missing_claim", "predecessor", "extra_claim", "epoch", "overlap", "digest", "actor_space", "actor_null"])
def test_transition_or_overlap_forgery_rolls_back_the_whole_page(pair, damage):
    source, target = pair
    spans = [{"gateway_id": HOME, "epoch": 1, "from_seq": 0}, transfer(source)]
    setup_successor(source, target, spans)
    first = page_v2(source, spans, limit=1)
    ingest(target, first)
    before = replicas.replica_state(target, room_id="room")
    page = copy.deepcopy(page_v2(source, spans))
    claim = page["events"][-1]
    if damage == "actor":
        claim["actor"] = {"kind": "system", "id": "impostor"}
    elif damage == "actor_space":
        claim["actor"]["id"] = " authority-control "
    elif damage == "actor_null":
        claim["actor"]["display_name"] = None
    elif damage == "missing_claim":
        claim["kind"] = "message.user"
        claim["actor"] = {"kind": "user", "id": "owner"}
    elif damage == "predecessor":
        claim["payload"]["previous_gateway_id"] = FINAL
    elif damage == "extra_claim":
        page["events"][1].update(kind=claim["kind"], actor=claim["actor"], payload=claim["payload"])
    elif damage == "epoch":
        claim["authority_epoch"] = 1
    elif damage == "overlap":
        page["events"][0]["payload"]["text"] = "altered"
    else:
        page["lineage_sha256"] = "0" * 64
    with pytest.raises(rooms.HostedRoomError):
        ingest(target, page)
    assert replicas.replica_state(target, room_id="room") == before
    ingest(target, page_v2(source, spans))
    assert replicas.replica_state(target, room_id="room")["lineage_status"] == "verified"


def test_v2_page_pins_head_claims_and_events_in_one_sqlite_snapshot(pair, monkeypatch):
    from gateway import hosted_room_passive_lineage as lineage
    source, _ = pair
    spans = [{"gateway_id": HOME, "epoch": 1, "from_seq": 0}, transfer(source)]
    with rooms._transaction(source) as conn:
        assert conn.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    original = lineage.source_locked
    advanced = False
    def concurrent_claim(conn, room_id, authority):
        nonlocal advanced
        if not advanced:
            advanced = True
            transfer(source, SUCCESSOR, 2, FINAL)
        return original(conn, room_id, authority)
    monkeypatch.setattr(lineage, "source_locked", concurrent_claim)
    page = rooms.read_events(source, room_id="room", replica_version=2)
    assert advanced
    assert page["authority"] == {"gateway_id": SUCCESSOR, "epoch": 2}
    assert page["lineage_sha256"] == manifest(spans)
    assert page["latest_seq"] == page["cursor"] == spans[-1]["from_seq"]
    assert page["events"][-1]["authority_epoch"] == 2
    assert rooms.room_state(source, room_id="room")["authority_epoch"] == 3
