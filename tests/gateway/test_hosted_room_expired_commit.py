"""Expired staged bytes cannot be revived by a delayed share."""
import pytest
from gateway.hosted_room_attachments import HostedRoomAttachmentStore, AttachmentNotFoundError, UNCOMMITTED_TTL_SECONDS


def test_expired_upload_cannot_be_committed_without_pruning(tmp_path):
    now = [100.0]
    store = HostedRoomAttachmentStore(tmp_path / 'state.db', clock=lambda: now[0])
    uploaded = store.put(room_id='room', upload_id='upload', kind='file', name='a.txt', mime='text/plain', data=b'bytes')
    manifest = [{key: uploaded[key] for key in ('attachment_id', 'kind', 'name', 'size', 'mime')}]
    now[0] += UNCOMMITTED_TTL_SECONDS
    with pytest.raises(AttachmentNotFoundError, match='expired'):
        store.commit_message(room_id='room', event_id='event', manifest=manifest, recipient_member_ids=['member'], viewer_access=True, hold_until_event=True)
