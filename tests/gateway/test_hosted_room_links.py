"""Private negotiated RoomLink storage tests."""

from __future__ import annotations

import stat
from concurrent.futures import ThreadPoolExecutor

from gateway.hosted_room_links import (
    load_room_links,
    make_stored_link,
    save_room_link,
)
from gateway.hosted_room_peer import GatewayRoomCatalog, catalog_mapping


def _catalog(installation="install-peer"):
    return GatewayRoomCatalog.from_mapping(
        catalog_mapping(
            installation_id=installation,
            persistent_process=True,
        )
    )


def test_room_link_store_is_private_transactional_and_upserted(tmp_path):
    path = tmp_path / "state.db"
    first = make_stored_link(
        room_id="room-1",
        member_id="member-1",
        target_url="https://peer.example.test",
        target_profile="reviewer",
        grant="grant.one",
        catalog=_catalog(),
        cancellation_scope_id="cancel-1",
        trace_id="trace-1",
    )
    save_room_link(path, first)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert load_room_links(path) == (first,)

    replacement = make_stored_link(
        room_id="room-1",
        member_id="member-1",
        target_url="https://relay.example.test",
        target_profile="reviewer",
        grant="grant.two",
        catalog=_catalog(),
        cancellation_scope_id="cancel-1",
        trace_id="trace-2",
    )
    save_room_link(path, replacement)
    assert load_room_links(path) == (replacement,)


def test_room_link_store_keeps_distinct_room_member_routes(tmp_path):
    path = tmp_path / "state.db"
    for member in ("member-a", "member-b"):
        save_room_link(
            path,
            make_stored_link(
                room_id="room-1",
                member_id=member,
                target_url=f"https://{member}.example.test",
                target_profile="default",
                grant=f"grant.{member}",
                catalog=_catalog(f"install-{member}"),
                cancellation_scope_id="cancel-room-1",
                trace_id=f"trace-{member}",
            ),
        )
    assert {row.member_id for row in load_room_links(path)} == {
        "member-a",
        "member-b",
    }


def test_concurrent_room_link_registrations_do_not_lose_updates(tmp_path):
    path = tmp_path / "state.db"

    def store(index):
        save_room_link(
            path,
            make_stored_link(
                room_id="room-1",
                member_id=f"member-{index}",
                target_url=f"https://member-{index}.example.test",
                target_profile=f"profile-{index}",
                grant=f"grant.{index}",
                catalog=_catalog(f"install-{index}"),
                cancellation_scope_id="cancel-room-1",
                trace_id=f"trace-{index}",
            ),
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(store, range(24)))
    assert len(load_room_links(path)) == 24


def test_room_link_repr_never_contains_grant():
    link = make_stored_link(
        room_id="room-1",
        member_id="member-1",
        target_url="https://peer.example.test",
        target_profile="reviewer",
        grant="top.secret.grant",
        catalog=_catalog(),
        cancellation_scope_id="cancel-1",
        trace_id="trace-1",
    )
    assert "top.secret.grant" not in repr(link)
