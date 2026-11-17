"""Real process reopen and SQL failures preserve the accepted file boundary."""
import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing

import pytest
from gateway import hosted_rooms
from gateway.hosted_room_attachments import HostedRoomAttachmentStore, UNCOMMITTED_TTL_SECONDS


def test_accepted_file_survives_fresh_process_after_staging_expiry(tmp_path):
    db = tmp_path / 'state.db'
    hosted_rooms.create_room(db, room_id='room', name='Room', members=[], authority_gateway_id='home')
    store = HostedRoomAttachmentStore(db, clock=lambda: 100.0)
    item = store.put(room_id='room', upload_id='upload', kind='file', name='proof.txt', mime='text/plain', data=b'accepted bytes')
    manifest = [{k: item[k] for k in ('attachment_id', 'kind', 'name', 'size', 'mime')}]
    store.commit_message(room_id='room', event_id='share', manifest=manifest, recipient_member_ids=['member'], viewer_access=True, hold_until_event=True)
    args = dict(room_id='room', event_id='share', kind='message.user', actor={'kind':'user', 'id':'owner'}, payload={'attachments':manifest}, authority_gateway_id='home', authority_epoch=1, now=101.0, expected_latest_seq=0)
    with closing(sqlite3.connect(db)) as conn:
        conn.execute("CREATE TRIGGER reject_share BEFORE INSERT ON hosted_room_events BEGIN SELECT RAISE(ABORT, 'disk rejected'); END")
        conn.commit()
    with pytest.raises(sqlite3.IntegrityError, match='disk rejected'):
        hosted_rooms.append_event(db, **args)
    with closing(sqlite3.connect(db)) as conn:
        assert conn.execute('SELECT expires_at FROM hosted_room_attachments').fetchone()[0] is not None
        assert conn.execute('SELECT count(*) FROM hosted_room_events').fetchone()[0] == 0
        conn.execute('DROP TRIGGER reject_share')
        conn.commit()
    with pytest.raises(hosted_rooms.EventCursorConflictError):
        hosted_rooms.append_event(db, **{**args, 'expected_latest_seq':1})
    hosted_rooms.append_event(db, **args)
    script = '''import json,sys
from gateway.hosted_room_attachments import HostedRoomAttachmentStore
from gateway import hosted_rooms
store=HostedRoomAttachmentStore(sys.argv[1],clock=lambda:float(sys.argv[3]))
data=store.read(room_id='room',attachment_id=sys.argv[2],recipient_member_id='member').data
print(json.dumps({'data':data.decode(),'events':len(hosted_rooms.read_events(sys.argv[1],room_id='room')['events']),'pruned':store.prune()}))
'''
    result = subprocess.run([sys.executable, '-c', script, str(db), item['attachment_id'], str(100 + UNCOMMITTED_TTL_SECONDS + 1)], env={**os.environ, 'HERMES_HOME':str(tmp_path)}, stdin=subprocess.DEVNULL, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == {'data':'accepted bytes', 'events':1, 'pruned':0}
