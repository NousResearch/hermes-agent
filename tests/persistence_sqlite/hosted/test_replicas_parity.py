"""The SQLite ``HostedRoomReplicaRepository`` against main's ``gateway.hosted_room_replicas`` functions, read from the
pinned main checkout: the placement's file is seeded through the storage verbs and copied, main's function runs on the
copy, the backend verb on the placement, then the outcomes and the hosted-room rows of both files are compared. Main's
stable install identity is pinned to the gateway id the backend verbs are given."""

from __future__ import annotations

from typing import Any

import pytest

from persistence.gateway.hosted_rooms import ReplicaPage
from persistence.sqlite.gateway.hosted_room_replicas import SQLiteHostedRoomReplicaRepository
from session_rules.hosted_rooms import _validate_members
from tests.persistence_sqlite._legacy_store import legacy_import, legacy_store
from tests.persistence_sqlite.hosted.conftest import NOW, outcome, same_rows, seeded_pair

__all__ = ["legacy_store"]

pytestmark = pytest.mark.usefixtures("clock")

LOCAL = "install:local"
REMOTE = "gw-remote"
MEMBERS = [{"member_id": "m1", "profile": "p1"}]
USER = {"kind": "user", "id": "u1"}


@pytest.fixture
def m1_replicas(legacy_store, monkeypatch):
    """Main's ``gateway.hosted_room_replicas`` from the pinned checkout, with its install identity pinned."""
    module = legacy_import("gateway.hosted_room_replicas")
    monkeypatch.setattr(module, "local_authority_gateway_id", lambda: LOCAL)
    return module


def _event(seq: int, *, epoch: int = 1, text: str = "x") -> dict[str, Any]:
    return {"seq": seq, "event_id": f"e{seq}", "kind": "message.user", "actor": USER, "authority_epoch": epoch,
            "payload": {"text": text}, "created_at": NOW - 100 + seq}


def _page(events: list[dict[str, Any]], *, gateway: str = REMOTE, epoch: int = 1, latest: Any = None) -> dict[str, Any]:
    page: dict[str, Any] = {"events": events, "authority": {"gateway_id": gateway, "epoch": epoch}}
    if latest is not None:
        page["latest_seq"] = latest
    return page


def _typed(page: dict[str, Any]) -> ReplicaPage:
    latest = page.get("latest_seq")
    return ReplicaPage(
        authority_gateway_id=page["authority"]["gateway_id"], authority_epoch=page["authority"]["epoch"],
        events=tuple(page["events"]), latest_seq=latest if isinstance(latest, int) and not isinstance(latest, bool) else None)


def _seed(store, placement: str) -> None:
    members_json = _validate_members(MEMBERS)[1]

    def ingest(room_id: str, name: str, page: dict[str, Any], now: float) -> None:
        store.hosted_room_replicas.ingest_page(
            placement=placement, room_id=room_id, room_name=name, members_json=members_json, page=_typed(page), now=now)
    ingest("rep", "Rep", _page([_event(1), _event(2)], epoch=2, latest=5), NOW - 50)
    ingest("mine", "Mine", _page([_event(1)], gateway=LOCAL), NOW - 45)
    for room_id, name, gateway in (("local", "Local", LOCAL), ("foreign", "Foreign", "gw-other")):
        store.hosted_rooms.create_room(
            placement=placement, room_id=room_id, name=name, members=MEMBERS, authority_gateway_id=gateway, now=NOW - 40)
    ingest("local", "Local", _page([_event(1)]), NOW - 35)


def _ingest(room_id: str, page: dict[str, Any], members=MEMBERS):
    return (
        lambda m1, p: m1.ingest_page(p, room_id=room_id, room_name="Rep", members=members, page=page, now=NOW),
        lambda r, pl: r.ingest_page(placement=pl, room_id=room_id, room_name="Rep", members_json=_validate_members(members)[1],
                                    page=_typed(page), now=NOW))


CASES = {
    "ingest_next": _ingest("rep", _page([_event(3), _event(4)], epoch=2)),
    "ingest_overlap": _ingest("rep", _page([_event(2), _event(3)], epoch=3, latest=3)),
    "ingest_replay": _ingest("rep", _page([_event(1), _event(2)], epoch=2)),
    "ingest_gap": _ingest("rep", _page([_event(4)], epoch=2)),
    "ingest_regression": _ingest("rep", _page([_event(3)], epoch=1)),
    "ingest_new_room": _ingest("new", _page([_event(1, text="é")], latest=True)),
    "ingest_unserializable": _ingest("new", _page([{**_event(1), "payload": {"bad": {1, 2}}}])),
    "promote": (
        lambda m1, p: m1.promote_replica(p, room_id="rep", reason="authority-unreachable", now=NOW),
        lambda r, pl: r.promote_replica(placement=pl, room_id="rep", local_gateway_id=LOCAL, reason="authority-unreachable", now=NOW)),
    "promote_missing": (
        lambda m1, p: m1.promote_replica(p, room_id="nope", now=NOW),
        lambda r, pl: r.promote_replica(placement=pl, room_id="nope", local_gateway_id=LOCAL, now=NOW)),
    "promote_already_ours": (
        lambda m1, p: m1.promote_replica(p, room_id="mine", now=NOW),
        lambda r, pl: r.promote_replica(placement=pl, room_id="mine", local_gateway_id=LOCAL, now=NOW)),
    "promote_room_exists": (
        lambda m1, p: m1.promote_replica(p, room_id="local", now=NOW),
        lambda r, pl: r.promote_replica(placement=pl, room_id="local", local_gateway_id=LOCAL, now=NOW)),
    "demote": (
        lambda m1, p: m1.demote_room(p, room_id="local", observed_gateway_id=REMOTE, observed_epoch=3, now=NOW),
        lambda r, pl: r.demote_room(placement=pl, room_id="local", observed_gateway_id=REMOTE, observed_epoch=3, local_gateway_id=LOCAL, now=NOW)),
    "demote_same_lineage": (
        lambda m1, p: m1.demote_room(p, room_id="local", observed_gateway_id=LOCAL, observed_epoch=1, now=NOW),
        lambda r, pl: r.demote_room(placement=pl, room_id="local", observed_gateway_id=LOCAL, observed_epoch=1, local_gateway_id=LOCAL, now=NOW)),
    "demote_regression": (
        lambda m1, p: m1.demote_room(p, room_id="local", observed_gateway_id=REMOTE, observed_epoch=1, now=NOW),
        lambda r, pl: r.demote_room(placement=pl, room_id="local", observed_gateway_id=REMOTE, observed_epoch=1, local_gateway_id=LOCAL, now=NOW)),
    "demote_not_local": (
        lambda m1, p: m1.demote_room(p, room_id="foreign", observed_gateway_id=REMOTE, observed_epoch=4, now=NOW),
        lambda r, pl: r.demote_room(placement=pl, room_id="foreign", observed_gateway_id=REMOTE, observed_epoch=4, local_gateway_id=LOCAL, now=NOW)),
    "demote_missing": (
        lambda m1, p: m1.demote_room(p, room_id="nope", observed_gateway_id=REMOTE, observed_epoch=4, now=NOW),
        lambda r, pl: r.demote_room(placement=pl, room_id="nope", observed_gateway_id=REMOTE, observed_epoch=4, local_gateway_id=LOCAL, now=NOW)),
    "state": (
        lambda m1, p: m1.replica_state(p, room_id="rep"),
        lambda r, pl: r.replica_state(placement=pl, room_id="rep")),
    "state_missing": (
        lambda m1, p: m1.replica_state(p, room_id="nope"),
        lambda r, pl: r.replica_state(placement=pl, room_id="nope")),
}


@pytest.mark.parametrize("placement", ["shared", "profile"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_replica_verb_matches_m1(case, placement, backend, scopes, tmp_path, m1_replicas):
    legacy, target = seeded_pair(
        backend, scopes["a"], placement, tmp_path, lambda path: _seed(backend.open(scopes["a"]), placement))
    legacy_call, verb = CASES[case]
    expected = outcome(lambda: legacy_call(m1_replicas, legacy))
    actual = outcome(lambda: verb(SQLiteHostedRoomReplicaRepository(backend.open(scopes["a"])), placement))
    assert actual == expected
    backend_rows, legacy_rows = same_rows(target, legacy)
    assert backend_rows == legacy_rows
