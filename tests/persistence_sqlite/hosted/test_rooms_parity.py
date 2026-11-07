"""The SQLite ``HostedRoomRepository`` room and event-log verbs against main's ``gateway.hosted_rooms`` functions, read
from the pinned main checkout.

Each case seeds the placement's file through the room verbs, copies it aside, runs main's function on the copy and the
backend verb on the placement, then compares the outcomes (value, or error class name and message) and every row of
the hosted-room tables both files hold.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest

from tests.persistence_sqlite._legacy_store import legacy_import, legacy_store
from tests.persistence_sqlite.hosted.conftest import NOW, dump, outcome, same_rows, seeded_pair

__all__ = ["legacy_store"]

pytestmark = pytest.mark.usefixtures("clock")

MEMBERS = [{"member_id": "m1", "profile": "p1"}, {"member_id": "m2", "profile": "p2", "target": {"kind": "peer", "installation_id": "i2"}}]
USER = {"kind": "user", "id": "u1"}
GATEWAY = {"kind": "gateway", "id": "gw"}
DAY = 86400.0


@pytest.fixture
def m1(legacy_store):
    """Main's ``gateway.hosted_rooms`` from the pinned checkout, importable for the whole test."""
    return legacy_import("gateway.hosted_rooms")


def _seed(rooms, placement: str) -> None:
    rooms.create_room(placement=placement, room_id="room", name="Room", members=MEMBERS, authority_gateway_id="gw",
                      now=NOW - 50)
    rooms.append_event(placement=placement, room_id="room", event_id="e1", kind="message.user", actor=USER,
                       payload={"text": "hi"}, authority_gateway_id="gw", authority_epoch=1, now=NOW - 40)
    rooms.append_event(placement=placement, room_id="room", event_id="e2", kind="room.activity", actor=GATEWAY,
                       payload={"n": 1}, authority_gateway_id="gw", authority_epoch=1, now=NOW - 30)
    rooms.rename_room(placement=placement, room_id="room", event_id="r1", name="Renamed", now=NOW - 20)
    rooms.create_room(placement=placement, room_id="legacy-room", name="Old",
                      members=[{"member_id": "m1", "profile": "p1", "target": {}}], authority_gateway_id="legacy",
                      now=NOW - 45)
    rooms.create_room(placement=placement, room_id="recent", name="Recent", members=[], authority_gateway_id="gw",
                      now=NOW - 10)
    rooms.disband_room(placement=placement, room_id="recent", expected_gateway_id="gw", expected_epoch=1, now=NOW - 5)
    # Disbanded last, so the seed's own retention passes keep it; a prune at NOW is past its retention.
    rooms.create_room(placement=placement, room_id="gone", name="Gone", members=[], authority_gateway_id="gw",
                      now=NOW - 100 * DAY)
    rooms.disband_room(placement=placement, room_id="gone", expected_gateway_id="gw", expected_epoch=1,
                       now=NOW - 91 * DAY)


def _seeded(backend, scopes, placement: str, tmp_path: Path) -> tuple[Path, Path]:
    return seeded_pair(backend, scopes["a"], placement, tmp_path,
                       lambda path: _seed(backend.open(scopes["a"]).hosted_rooms, placement))


Call = Callable[[Any, Any], Any]
# name -> (main's call on its module and a path, backend call on the repository with the placement)
CASES: dict[str, tuple[Call, Callable[[Any, str], Any]]] = {
    "create_new": (
        lambda m1, p: m1.create_room(p, room_id="new", name="New", members=MEMBERS, authority_gateway_id="gw", now=NOW),
        lambda r, pl: r.create_room(placement=pl, room_id="new", name="New", members=MEMBERS, authority_gateway_id="gw", now=NOW)),
    "create_replay": (
        lambda m1, p: m1.create_room(p, room_id="room", name="Renamed", members=MEMBERS, authority_gateway_id="gw", now=NOW),
        lambda r, pl: r.create_room(placement=pl, room_id="room", name="Renamed", members=MEMBERS, authority_gateway_id="gw", now=NOW)),
    "create_conflict": (
        lambda m1, p: m1.create_room(p, room_id="room", name="Other", members=MEMBERS, authority_gateway_id="gw", now=NOW),
        lambda r, pl: r.create_room(placement=pl, room_id="room", name="Other", members=MEMBERS, authority_gateway_id="gw", now=NOW)),
    "create_other_authority": (
        lambda m1, p: m1.create_room(p, room_id="room", name="Renamed", members=MEMBERS, authority_gateway_id="gw-2", now=NOW),
        lambda r, pl: r.create_room(placement=pl, room_id="room", name="Renamed", members=MEMBERS, authority_gateway_id="gw-2", now=NOW)),
    "create_adopts_legacy": (
        lambda m1, p: m1.create_room(p, room_id="legacy-room", name="Old", members=[{"member_id": "m1", "profile": "p1", "target": {"kind": "local"}}], authority_gateway_id="gw", now=NOW),
        lambda r, pl: r.create_room(placement=pl, room_id="legacy-room", name="Old", members=[{"member_id": "m1", "profile": "p1", "target": {"kind": "local"}}], authority_gateway_id="gw", now=NOW)),
    "create_disbanded_id": (
        lambda m1, p: m1.create_room(p, room_id="recent", name="Recent", members=[], authority_gateway_id="gw", now=NOW),
        lambda r, pl: r.create_room(placement=pl, room_id="recent", name="Recent", members=[], authority_gateway_id="gw", now=NOW)),
    "create_invalid_payload": (
        lambda m1, p: m1.create_room(p, room_id="bad room", name="X", members=[], authority_gateway_id="gw", now=NOW),
        lambda r, pl: r.create_room(placement=pl, room_id="bad room", name="X", members=[], authority_gateway_id="gw", now=NOW)),
    "rename_new": (
        lambda m1, p: m1.rename_room(p, room_id="room", event_id="r2", name="Again", now=NOW),
        lambda r, pl: r.rename_room(placement=pl, room_id="room", event_id="r2", name="Again", now=NOW)),
    "rename_replay": (
        lambda m1, p: m1.rename_room(p, room_id="room", event_id="r1", name="Renamed", now=NOW),
        lambda r, pl: r.rename_room(placement=pl, room_id="room", event_id="r1", name="Renamed", now=NOW)),
    "rename_conflict": (
        lambda m1, p: m1.rename_room(p, room_id="room", event_id="r1", name="Different", now=NOW),
        lambda r, pl: r.rename_room(placement=pl, room_id="room", event_id="r1", name="Different", now=NOW)),
    "rename_disbanded": (
        lambda m1, p: m1.rename_room(p, room_id="recent", event_id="r9", name="X", now=NOW),
        lambda r, pl: r.rename_room(placement=pl, room_id="recent", event_id="r9", name="X", now=NOW)),
    "rename_missing": (
        lambda m1, p: m1.rename_room(p, room_id="nope", event_id="r9", name="X", now=NOW),
        lambda r, pl: r.rename_room(placement=pl, room_id="nope", event_id="r9", name="X", now=NOW)),
    "append_new": (
        lambda m1, p: m1.append_event(p, room_id="room", event_id="e3", kind="message.user", actor=USER, payload={"text": "x"}, authority_gateway_id="gw", authority_epoch=1, expected_latest_seq=3, now=NOW),
        lambda r, pl: r.append_event(placement=pl, room_id="room", event_id="e3", kind="message.user", actor=USER, payload={"text": "x"}, authority_gateway_id="gw", authority_epoch=1, expected_latest_seq=3, now=NOW)),
    "append_replay": (
        lambda m1, p: m1.append_event(p, room_id="room", event_id="e1", kind="message.user", actor=USER, payload={"text": "hi"}, authority_gateway_id="gw", authority_epoch=1, now=NOW),
        lambda r, pl: r.append_event(placement=pl, room_id="room", event_id="e1", kind="message.user", actor=USER, payload={"text": "hi"}, authority_gateway_id="gw", authority_epoch=1, now=NOW)),
    "append_conflict": (
        lambda m1, p: m1.append_event(p, room_id="room", event_id="e1", kind="message.user", actor=USER, payload={"text": "other"}, authority_gateway_id="gw", authority_epoch=1, now=NOW),
        lambda r, pl: r.append_event(placement=pl, room_id="room", event_id="e1", kind="message.user", actor=USER, payload={"text": "other"}, authority_gateway_id="gw", authority_epoch=1, now=NOW)),
    "append_stale_authority": (
        lambda m1, p: m1.append_event(p, room_id="room", event_id="e4", kind="message.user", actor=USER, payload={}, authority_gateway_id="gw", authority_epoch=2, now=NOW),
        lambda r, pl: r.append_event(placement=pl, room_id="room", event_id="e4", kind="message.user", actor=USER, payload={}, authority_gateway_id="gw", authority_epoch=2, now=NOW)),
    "append_cursor_conflict": (
        lambda m1, p: m1.append_event(p, room_id="room", event_id="e4", kind="message.user", actor=USER, payload={}, authority_gateway_id="gw", authority_epoch=1, expected_latest_seq=1, now=NOW),
        lambda r, pl: r.append_event(placement=pl, room_id="room", event_id="e4", kind="message.user", actor=USER, payload={}, authority_gateway_id="gw", authority_epoch=1, expected_latest_seq=1, now=NOW)),
    "append_control": (
        lambda m1, p: m1.append_event(p, room_id="room", event_id="stop-1", kind="room.stop_requested", actor=GATEWAY, payload={"cancel_id": "c"}, authority_gateway_id="gw", authority_epoch=1, now=NOW),
        lambda r, pl: r.append_event(placement=pl, room_id="room", event_id="stop-1", kind="room.stop_requested", actor=GATEWAY, payload={"cancel_id": "c"}, authority_gateway_id="gw", authority_epoch=1, now=NOW)),
    "append_expired_room": (
        lambda m1, p: m1.append_event(p, room_id="nope", event_id="e9", kind="message.user", actor=USER, payload={}, authority_gateway_id="gw", authority_epoch=1, now=NOW),
        lambda r, pl: r.append_event(placement=pl, room_id="nope", event_id="e9", kind="message.user", actor=USER, payload={}, authority_gateway_id="gw", authority_epoch=1, now=NOW)),
    "disband_live": (
        lambda m1, p: m1.disband_room(p, room_id="room", expected_gateway_id="gw", expected_epoch=1, now=NOW),
        lambda r, pl: r.disband_room(placement=pl, room_id="room", expected_gateway_id="gw", expected_epoch=1, now=NOW)),
    "disband_replay": (
        lambda m1, p: m1.disband_room(p, room_id="recent", expected_gateway_id="other", expected_epoch=7, now=NOW),
        lambda r, pl: r.disband_room(placement=pl, room_id="recent", expected_gateway_id="other", expected_epoch=7, now=NOW)),
    "disband_stale": (
        lambda m1, p: m1.disband_room(p, room_id="room", expected_gateway_id="gw", expected_epoch=2, now=NOW),
        lambda r, pl: r.disband_room(placement=pl, room_id="room", expected_gateway_id="gw", expected_epoch=2, now=NOW)),
    "disband_missing": (
        lambda m1, p: m1.disband_room(p, room_id="nope", expected_gateway_id="gw", expected_epoch=1, now=NOW),
        lambda r, pl: r.disband_room(placement=pl, room_id="nope", expected_gateway_id="gw", expected_epoch=1, now=NOW)),
    "prune": (
        lambda m1, p: m1.prune_disbanded_rooms(p, now=NOW),
        lambda r, pl: r.prune_disbanded_rooms(placement=pl, now=NOW)),
    "list_live": (
        lambda m1, p: m1.list_rooms(p),
        lambda r, pl: r.list_rooms(placement=pl)),
    "list_all_paged": (
        lambda m1, p: m1.list_rooms(p, include_disbanded=True, limit=2, offset=1),
        lambda r, pl: r.list_rooms(placement=pl, include_disbanded=True, limit=2, offset=1)),
    "state_live": (
        lambda m1, p: m1.room_state(p, room_id="room"),
        lambda r, pl: r.room_state(placement=pl, room_id="room")),
    "state_disbanded": (
        lambda m1, p: m1.room_state(p, room_id="recent", include_disbanded=True),
        lambda r, pl: r.room_state(placement=pl, room_id="recent", include_disbanded=True)),
    "state_disbanded_hidden": (
        lambda m1, p: m1.room_state(p, room_id="recent"),
        lambda r, pl: r.room_state(placement=pl, room_id="recent")),
    "events_all": (
        lambda m1, p: m1.read_events(p, room_id="room"),
        lambda r, pl: r.read_events(placement=pl, room_id="room")),
    "events_page": (
        lambda m1, p: m1.read_events(p, room_id="room", since_seq=1, limit=1),
        lambda r, pl: r.read_events(placement=pl, room_id="room", since_seq=1, limit=1)),
    "events_ahead": (
        lambda m1, p: m1.read_events(p, room_id="room", since_seq=9),
        lambda r, pl: r.read_events(placement=pl, room_id="room", since_seq=9)),
    "events_disbanded": (
        lambda m1, p: m1.read_events(p, room_id="recent", include_disbanded=True),
        lambda r, pl: r.read_events(placement=pl, room_id="recent", include_disbanded=True)),
    "probe_live": (
        lambda m1, p: m1.probe_hosted_room(p, room_id="room"),
        lambda r, pl: r.probe_hosted_room(placement=pl, room_id="room")),
    "probe_disbanded": (
        lambda m1, p: m1.probe_hosted_room(p, room_id="recent"),
        lambda r, pl: r.probe_hosted_room(placement=pl, room_id="recent")),
}


@pytest.mark.parametrize("placement", ["shared", "profile"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_room_verb_matches_m1(case, placement, backend, scopes, tmp_path, m1):
    legacy, target = _seeded(backend, scopes, placement, tmp_path)
    legacy_call, verb = CASES[case]
    expected = outcome(lambda: legacy_call(m1, legacy))
    actual = outcome(lambda: verb(backend.open(scopes["a"]).hosted_rooms, placement))
    assert actual == expected
    backend_rows, legacy_rows = same_rows(target, legacy)
    assert backend_rows == legacy_rows


def test_prune_disbanded_rooms_purges_old_tombstones_like_m1(backend, scopes, tmp_path, m1):
    legacy, target = _seeded(backend, scopes, "shared", tmp_path)
    assert m1.prune_disbanded_rooms(legacy, now=NOW) == 1
    assert backend.open(scopes["a"]).hosted_rooms.prune_disbanded_rooms(placement="shared", now=NOW) == 1
    backend_rows, legacy_rows = same_rows(target, legacy)
    assert backend_rows == legacy_rows
    rows = dump(target)
    assert "gone" not in {row[0] for row in rows["hosted_rooms"]}
    assert "gone" in {row[0] for row in rows["hosted_room_retired_ids"]}
