"""Published log references survive missing local bytes without widening viewer authority."""
import base64
import json
import sqlite3

import pytest

from gateway.hosted_room_attachments import HostedRoomAttachmentStore
from gateway.session_controls import AuthorityConnection
from tests.gateway.test_session_group_files import call, gateway as gateway_fixture, reason, share

# Explicit fixture alias keeps its pytest ownership separate from test parameters.
gateway = gateway_fixture


def _forget_bytes(gateway):
    with sqlite3.connect(gateway.authority.db.db_path) as conn:
        conn.execute('DELETE FROM hosted_room_attachments')
        conn.execute('DELETE FROM hosted_room_attachment_blobs')


def test_three_retained_versions_and_one_new_file_keep_order_identity_and_downloads(gateway):
    prior = [share(gateway, n, f'old version {n}'.encode()) for n in range(3)]
    _forget_bytes(gateway)
    first = call(gateway.owner, 'groups.attachment.list', room_id='room', limit=2)['result']
    assert [(r['event_id'], r['attachment_id']) for r in first['items']] == list(reversed(prior))[:2]
    assert all(r['available'] is False for r in first['items'])
    fresh = share(gateway, 4, b'fresh bytes')
    second = call(gateway.owner, 'groups.attachment.list', room_id='room', limit=2, cursor=first['next_cursor'])['result']
    assert [(r['event_id'], r['attachment_id']) for r in second['items']] == [prior[0]]
    assert second['snapshot_seq'] == first['snapshot_seq'] and not second['has_more']
    all_rows = call(gateway.owner, 'groups.attachment.list', room_id='room')['result']['items']
    assert [(r['event_id'], r['attachment_id']) for r in all_rows] == [fresh, *reversed(prior)]
    assert all_rows[0].get('available', True) is True
    assert all(r['available'] is False for r in all_rows[1:])
    for event, attachment in prior:
        assert reason(call(gateway.owner, 'groups.attachment.download', room_id='room',
                           event_id=event, attachment_id=attachment)) == 'attachment_unavailable'
    result = call(gateway.owner, 'groups.attachment.download', room_id='room',
                  event_id=fresh[0], attachment_id=fresh[1])['result']
    assert base64.b64decode(result['data_base64']) == b'fresh bytes'
    filtered = call(gateway.owner, 'groups.attachment.list', room_id='room', query='report', producer_member_id='desktop')['result']
    assert len(filtered['items']) == 4


def test_missing_reference_does_not_reveal_unrelated_private_or_unauthorized_metadata(gateway):
    event, attachment = share(gateway, 1, b'shared bytes')
    _forget_bytes(gateway)
    stranger = AuthorityConnection(gateway.authority, object(), {'user_id':'bob'})
    assert reason(call(stranger, 'groups.attachment.list', room_id='room')) == 'permission_denied'
    assert reason(call(stranger, 'groups.attachment.download', room_id='room',
                       event_id=event, attachment_id=attachment)) == 'permission_denied'
    for owner_event, owner_attachment in [('unknown-event', attachment), (event, 'att_' + 'a'*32)]:
        assert reason(call(gateway.owner, 'groups.attachment.download', room_id='room',
                           event_id=owner_event, attachment_id=owner_attachment)) == 'invalid_params'
    private_event, private_attachment = share(gateway, 2, b'private bytes')
    with sqlite3.connect(gateway.authority.db.db_path) as conn:
        conn.execute('UPDATE hosted_room_attachments SET viewer_access=0 WHERE attachment_id=?', (private_attachment,))
    rows = call(gateway.owner, 'groups.attachment.list', room_id='room')['result']['items']
    assert [(r['event_id'],r['attachment_id']) for r in rows] == [(event,attachment)]
    assert reason(call(gateway.owner, 'groups.attachment.download', room_id='room',
                       event_id=private_event, attachment_id=private_attachment)) == 'invalid_params'


@pytest.mark.parametrize('fault', ['missing-file','hash-mismatch','revocation-during-read'])
def test_byte_failure_is_typed_only_while_its_exact_published_reference_stays_authorized(gateway, monkeypatch, fault):
    event, attachment = share(gateway, 1, b'canonical bytes')
    store = HostedRoomAttachmentStore(gateway.authority.db.db_path)
    with sqlite3.connect(gateway.authority.db.db_path) as conn:
        blob = conn.execute('SELECT blob_id FROM hosted_room_attachments WHERE attachment_id=?',(attachment,)).fetchone()[0]
    path = store._blob_path(blob)
    if fault == 'missing-file':
        path.unlink()
        row, = call(gateway.owner, 'groups.attachment.list', room_id='room')['result']['items']
        assert row['available'] is False
    elif fault == 'hash-mismatch':
        path.write_bytes(b'x'*len(b'canonical bytes'))
    else:
        original = HostedRoomAttachmentStore._read_blob
        def revoked(self, **kwargs):
            value = original(self, **kwargs)
            with sqlite3.connect(gateway.authority.db.db_path) as conn:
                conn.execute('UPDATE hosted_room_attachments SET viewer_access=0 WHERE attachment_id=?',(attachment,))
            return value
        monkeypatch.setattr(HostedRoomAttachmentStore, '_read_blob', revoked)
    expected = 'invalid_params' if fault == 'revocation-during-read' else 'attachment_unavailable'
    assert reason(call(gateway.owner, 'groups.attachment.download', room_id='room', event_id=event,
                       attachment_id=attachment)) == expected


@pytest.mark.parametrize('fault', ['no-store', 'actor', 'kind', 'manifest', 'event'])
def test_historical_references_require_the_exact_valid_public_log(gateway, fault):
    event, attachment = share(gateway, 1, b'known public version')
    _forget_bytes(gateway)
    with sqlite3.connect(gateway.authority.db.db_path) as conn:
        if fault == 'no-store':
            conn.execute('DROP TABLE hosted_room_attachments')
            conn.execute('DROP TABLE hosted_room_attachment_blobs')
        else:
            payload = json.loads(conn.execute('SELECT payload_json FROM hosted_room_events WHERE event_id=?',(event,)).fetchone()[0])
            payload['attachments'][0]['extra'] = 'unverified'
            statements = {
                'actor': ("UPDATE hosted_room_events SET actor_json='{}' WHERE event_id=?", (event,)),
                'kind': ("UPDATE hosted_room_events SET kind='private.fixture' WHERE event_id=?", (event,)),
                'manifest': ('UPDATE hosted_room_events SET payload_json=? WHERE event_id=?', (json.dumps(payload),event)),
                'event': ('DELETE FROM hosted_room_events WHERE event_id=?', (event,)),
            }
            conn.execute(*statements[fault])
    rows = call(gateway.owner, 'groups.attachment.list', room_id='room')['result']['items']
    if fault == 'no-store':
        assert [(r['event_id'],r['attachment_id'],r['available']) for r in rows] == [(event,attachment,False)]
    else:
        assert rows == []
    assert reason(call(gateway.owner, 'groups.attachment.download', room_id='room',event_id=event,
                       attachment_id=attachment)) == ('storage_unavailable' if fault == 'no-store' else 'invalid_params')


def test_peer_history_reference_is_visible_but_never_downloadable(gateway):
    event, attachment = share(gateway, 1, b"remote bytes", name="remote.txt")
    _forget_bytes(gateway)
    # A historical log imported without its originating gateway's byte store.
    with sqlite3.connect(gateway.authority.db.db_path) as conn:
        conn.execute("UPDATE hosted_room_events SET kind='message.member', actor_json=? WHERE event_id=?",
                     (json.dumps({"kind": "member", "id": "peer", "connection_id": "elsewhere"}), event))
    row, = call(gateway.owner, "groups.attachment.list", room_id="room")["result"]["items"]
    assert row["name"] == "remote.txt" and row["available"] is False
    assert reason(call(gateway.owner, "groups.attachment.download", room_id="room", event_id=event,
                       attachment_id=attachment)) == "attachment_unavailable"
