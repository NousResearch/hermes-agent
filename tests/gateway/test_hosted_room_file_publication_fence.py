"""Bridge integration with the already reviewed viewer fence; no fence rewrite."""

import sqlite3

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway.hosted_room_attachments import AttachmentError
from gateway.hosted_room_file_contract import FileAccessError
from tests.gateway.test_hosted_room_file_access import publish
from tests.gateway.test_hosted_room_file_access import file_state as file_state
from tests.gateway.platforms.test_api_server_room_control_files import file_api as file_api


def remove_publication_after_verified_read(state, monkeypatch):
    original = state.store._read_blob
    changes = []

    def read_then_unpublish(**kwargs):
        data = original(**kwargs)
        with sqlite3.connect(state.db, timeout=2) as conn:
            conn.execute(
                "UPDATE hosted_room_events SET payload_json='{}' WHERE room_id='room-1' AND event_id='event-1'"
            )
        changes.append(True)
        return data

    monkeypatch.setattr(state.store, "_read_blob", read_then_unpublish)
    return changes


@pytest.mark.parametrize("profile", ["default", "ops"])
def test_bridge_selected_read_honors_post_byte_publication_removal(
    file_state, monkeypatch, profile
):
    state = file_state
    item = publish(state, "private.txt", b"Previously published exact bytes")
    changes = remove_publication_after_verified_read(state, monkeypatch)
    denied = False
    try:
        result = state.backend.read_file(
            room=state.room,
            profile=profile,
            event_id=item["event_id"],
            attachment_id=item["attachment_id"],
        )
    except FileAccessError:
        denied = True
    else:
        assert result.data == b"Previously published exact bytes"
    assert changes
    # The existing fenced store correctly rejects this same selected row now.
    with pytest.raises(AttachmentError):
        state.store.read_viewer(
            room_id="room-1",
            event_id=item["event_id"],
            attachment_id=item["attachment_id"],
            authority_gateway_id=state.authority,
            authority_epoch=1,
        )
    assert denied, (
        "Messaging bridge returned bytes after the publication was removed during I/O"
    )


@pytest.mark.asyncio
async def test_real_authorized_http_retrieval_does_not_bypass_publication_fence(
    file_api, monkeypatch
):
    state = file_api
    item = publish(state, "private.txt", b"Previously published exact bytes")
    changes = remove_publication_after_verified_read(state, monkeypatch)
    async with TestClient(TestServer(state.app)) as client:
        response = await client.get(
            f"/v1/room-controls/room-1/files/{item['attachment_id']}?event_id={item['event_id']}",
            headers=state.headers,
        )
        body = await response.read()
        assert changes
        with pytest.raises(AttachmentError):
            state.store.read_viewer(
                room_id="room-1",
                event_id=item["event_id"],
                attachment_id=item["attachment_id"],
                authority_gateway_id=state.authority,
                authority_epoch=1,
            )
        assert response.status != 200 and body != b"Previously published exact bytes", (
            response.status,
            body,
        )


@pytest.mark.parametrize("profile", ["default", "ops"])
def test_empty_upload_is_rejected_before_it_can_affect_a_files_page(
    file_state, profile
):
    state = file_state
    ordinary = publish(state, "ordinary.txt", b"ordinary readable bytes")
    with pytest.raises(AttachmentError, match="must not be empty"):
        publish(state, "empty.txt", b"")
    page = state.backend.list_files(room=state.room, profile=profile)
    assert {item["attachment_id"] for item in page["items"]} == {
        ordinary["attachment_id"]
    }
