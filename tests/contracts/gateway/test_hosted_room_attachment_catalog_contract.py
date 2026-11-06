"""An exact published version has the same identity in the catalog and the viewer read."""

from tests.contracts.gateway.test_hosted_room_attachments_reads_contract import (
    DATA,
    _commit,
    _publish,
    _put,
    _room,
)


def test_catalog_version_matches_the_room_viewers_download(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    meta = _put(store)
    _commit(store, [meta], viewer=True, hold=True)
    _publish(store, [meta])
    files = store.hosted_room_attachments
    scope = dict(
        placement="profile",
        room_id="room",
        authority_gateway_id="gw",
        authority_epoch=1,
    )
    (item,) = files.list_published(**scope)["items"]
    assert item.get("available", True)
    assert (item["event_id"], item["attachment_id"]) == ("e1", meta["attachment_id"])
    exact = dict(event_id=item["event_id"], attachment_id=item["attachment_id"])
    assert files.published_reference(**scope, **exact)
    assert not files.published_reference(
        **scope, event_id="other", attachment_id=meta["attachment_id"]
    )
    assert files.read_viewer(**scope, **exact).data == DATA
