"""Fresh scoped passive grants, with revocation/expiry racing a real SQLite lock."""

import copy
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from gateway import hosted_room_peer as peer
from gateway import hosted_room_replicas as replicas
from gateway import hosted_room_replica_retirement as retirement
from gateway import hosted_rooms as rooms
from gateway.hosted_room_replica_ingress import ingest_granted_page
from tests.gateway.test_hosted_room_replica_ingress import SECRET, HOME, TARGET, MEMBERS, grant
from tests.gateway.test_hosted_room_replica_lineage import SUCCESSOR, transfer, manifest


@pytest.fixture
def copying(tmp_path):
    source, target = tmp_path / "source.db", tmp_path / "target.db"
    members = copy.deepcopy(MEMBERS)
    observer = copy.deepcopy(members[-1])
    observer.update(member_id="observer", handle="observer", profile="spare")
    observer["target"]["profile"] = "spare"
    members.append(observer)
    rooms.create_room(source, room_id="room", name="Workshop", members=members, authority_gateway_id=HOME)
    rooms.append_event(source, room_id="room", event_id="start", kind="message.user",
        actor={"kind": "user", "id": "owner"}, payload={"text": "retained"}, authority_gateway_id=HOME, authority_epoch=1)
    entry = retirement.prepare_home_enrollment(source, room_id="room", target_install_id=TARGET,
        endpoint="https://participant.example", local_gateway_id=HOME, secret=SECRET, enrollment_id="original")
    retirement.enroll_target(target, enrollment=entry, target_install_id=TARGET)
    old, old_claims = grant(target, permissions=("status", "replicate"), member_id="observer", target_profile="spare")
    old_page = rooms.read_events(source, room_id="room")
    def send(token, page, profile="reviewer", **kwargs):
        return ingest_granted_page(target, token=token, secret=SECRET,
            target_install_id=kwargs.pop("target_install_id", TARGET), target_profile=profile,
            room_id="room", room_name="Workshop", members=members, page=page, **kwargs)
    send(old, old_page, "spare")
    spans = [{"gateway_id": HOME, "epoch": 1, "from_seq": 0}, transfer(source)]
    entry = retirement.prepare_home_enrollment(source, room_id="room", target_install_id=TARGET,
        endpoint="https://participant.example", local_gateway_id=SUCCESSOR, secret=SECRET,
        enrollment_id="successor", replace_enrollment_id="original")
    retirement.enroll_target(target, enrollment=entry, target_install_id=TARGET,
                            authority_history=spans, expected_enrollment_id="original")
    fresh, claims = grant(target, permissions=("status", "replicate"), grant_id="fresh",
                         home_install_id=SUCCESSOR, authority_gateway_id=SUCCESSOR, authority_epoch=2)
    page = rooms.read_events(source, room_id="room", replica_version=2)
    assert page["lineage_sha256"] == manifest(spans)
    return target, send, old, old_claims, old_page, fresh, claims, page


def test_replacement_fences_even_a_still_live_old_profile(copying):
    target, send, old, old_claims, old_page, fresh, _, page = copying
    assert rooms.peer_room_grant_is_current(target, claims=old_claims)
    before = replicas.replica_state(target, room_id="room")
    for token, attempt, profile, extra in [
        (old, old_page, "spare", {}), (old, page, "spare", {}),
        (fresh, page, "spare", {}), (fresh, page, "reviewer", {"target_install_id": HOME}),
        (fresh, {**page, "lineage_sha256": "0" * 64}, "reviewer", {}),
    ]:
        with pytest.raises((rooms.HostedRoomError, peer.HostedRoomGrantError)):
            send(token, attempt, profile, **extra)
        assert replicas.replica_state(target, room_id="room") == before
    assert send(fresh, page)["lineage_status"] == "verified"
    assert send(fresh, page)["ingested"] == 0


@pytest.mark.parametrize("invalidated", ["expiry", "revocation"])
def test_grant_recheck_happens_after_waiting_for_sqlite(copying, monkeypatch, invalidated):
    target, send, _, _, _, fresh, claims, page = copying
    before = replicas.replica_state(target, room_id="room")
    validated = threading.Event()
    original = replicas._validate_page
    def observed(page):
        result = original(page)
        validated.set()
        return result
    monkeypatch.setattr(replicas, "_validate_page", observed)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with rooms._transaction(target, immediate=True) as conn:
            future = pool.submit(send, fresh, page)
            assert validated.wait(5)
            if invalidated == "expiry":
                monkeypatch.setattr(time, "time", lambda: claims["status_expires_at"] + 1)
            else:
                conn.execute("""UPDATE hosted_room_peer_reservations SET revoked_at=?
                    WHERE room_id='room' AND authority_gateway_id=? AND authority_epoch=2""", (time.time(), SUCCESSOR))
        with pytest.raises(peer.HostedRoomGrantError):
            future.result(timeout=10)
    assert replicas.replica_state(target, room_id="room") == before
