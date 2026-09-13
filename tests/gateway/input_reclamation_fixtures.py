"""Real profile ownership and SQLite/Files fixtures, never a worker or model."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from gateway import hosted_rooms
from gateway.hosted_room_attachments import HostedRoomAttachmentStore
from gateway.hosted_room_input_custody import initialize_input_custody
from gateway.hosted_room_input_reclamation import initialize_working_copies
from gateway.runtime_ownership import process_ownership
from gateway.session_authority import LiveSession
from gateway.session_contract import Principal
from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
from hermes_state_terminal import ADMISSION_PREFIX, identity_key
from tests.gateway.test_native_media_budget import _authority


def owned(tmp_path, monkeypatch, *, initialize=True):
    process_ownership.reserve([tmp_path])
    db, owner = _authority(tmp_path, monkeypatch)
    initialize_input_custody(db)
    if initialize:
        initialize_working_copies(db, epoch=owner.epoch)
    return db, owner


def close(db, home):
    db.close()
    process_ownership.release(home)


def rpc_files(home, owner, *, named=False, count=1, image=False):
    hosted_rooms.create_room(owner.db.db_path, room_id='room', name='Room', authority_gateway_id='home',
        members=[dict(member_id='member', profile='default', handle='member')])
    store = HostedRoomAttachmentStore(owner.db.db_path)
    entries = []
    for index in range(count):
        data = bytes([65 + index]) * 32
        is_image = image and index == count - 1
        if is_image:
            from tests.gateway.test_api_media_retention import PNG
            data = PNG
        saved = store.put(room_id='room', upload_id=f'upload-{index}', kind='image' if is_image else 'file',
            name=f'{index}.png' if is_image else f'{index}.txt', mime='image/png' if is_image else 'text/plain', data=data)
        entries.append(({key: saved[key] for key in ('attachment_id', 'kind', 'name', 'size', 'mime')}, data))
    store.commit_message(room_id='room', event_id='event', manifest=[item for item, _ in entries], recipient_member_ids=['member'])
    bound = [({**item, 'event_id': 'event'}, data) for item, data in entries]
    actor = Principal('human', owner.profile_id, frozenset({'session:read', 'session:submit', 'session:control'}), 'fixture')
    rpc = HostedRoomAuthorityRPC(owner, asyncio.get_running_loop(), room_id='room', member_id='member', profile='default',
        principal=actor, authorize=lambda *args: True)
    owner.db.create_session(rpc.ref.session_id, source='cli')
    owner.sessions[rpc.ref.session_id] = LiveSession(SimpleNamespace(platform=None, user_id='human'), 'fixture')
    if named:
        rpc.hosted_attachment_data = bound
    return rpc, bound


def retire_metadata(db, admission_id):
    """Seed canonical terminal-only evidence, not a session-close operation."""
    def write(conn):
        row = dict(conn.execute('SELECT * FROM session_admissions WHERE admission_id=?', (admission_id,)).fetchone())
        row.update(status='terminal', outcome='completed', payload_json='{}', lineage_json='[]')
        conn.execute('DELETE FROM session_admissions WHERE admission_id=?', (admission_id,))
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', (ADMISSION_PREFIX + admission_id, json.dumps(row)))
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
            (identity_key(row['principal_id'], row['target_session_id'], row['request_id']), json.dumps(admission_id)))
    db._execute_write(write)


def copy_records(db):
    with db._read_ctx() as conn:
        return [dict(row) for row in conn.execute('SELECT * FROM input_custody_copies ORDER BY copy_id')]


def expire(db, handle):
    db._execute_write(lambda conn: conn.execute('UPDATE input_custody_preparations SET expires_at=0 WHERE preparation_id=?',
                                              (handle.preparation_id,)))


def v3_path(prepared):
    # Fixture uses the fixed payload formatter, not production prompt parsing.
    return Path(prepared.payload['text'].split('file: ', 1)[1].strip())
