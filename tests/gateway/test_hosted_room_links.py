"""Private negotiated RoomLink storage tests."""

from __future__ import annotations

import stat
import pytest
from concurrent.futures import ThreadPoolExecutor

from gateway.hosted_room_links import (
    load_room_links,
    make_stored_link,
    save_room_link,
)
from gateway.hosted_room_peer import GatewayRoomCatalog, catalog_mapping
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle

PLACEMENT = "profile"


def _catalog(installation="install-peer"):
    return GatewayRoomCatalog.from_mapping(
        catalog_mapping(
            target_profile="default",
            installation_id=installation,
            persistent_process=True,
        )
    )


def test_room_link_store_is_private_transactional_and_upserted(tmp_path):
    store = storage_handle(tmp_path)
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
    save_room_link(store, first, placement=PLACEMENT)
    assert load_room_links(store, placement=PLACEMENT) == (first,)

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
    save_room_link(store, replacement, placement=PLACEMENT)
    assert load_room_links(store, placement=PLACEMENT) == (replacement,)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("placement", ["profile", "shared"])
def test_room_link_store_has_owner_only_mode(tmp_path, placement):
    home = tmp_path / "profiles" / "default"
    home.mkdir(parents=True)
    path = home / "state.db" if placement == "profile" else tmp_path / "shared-state.db"
    save_room_link(storage_handle(home), make_stored_link(
        room_id="room", member_id="member", target_url="https://peer.example.test",
        target_profile="default", grant="grant", catalog=_catalog(),
        cancellation_scope_id="cancel", trace_id="trace",
    ), placement=placement)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_room_link_store_keeps_distinct_room_member_routes(tmp_path):
    store = storage_handle(tmp_path)
    for member in ("member-a", "member-b"):
        save_room_link(
            store,
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
            placement=PLACEMENT,
        )
    assert {row.member_id for row in load_room_links(store, placement=PLACEMENT)} == {
        "member-a",
        "member-b",
    }


def test_concurrent_room_link_registrations_do_not_lose_updates(tmp_path):
    handle = storage_handle(tmp_path)

    def store(index):
        save_room_link(
            handle,
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
            placement=PLACEMENT,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(store, range(24)))
    assert len(load_room_links(handle, placement=PLACEMENT)) == 24


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
