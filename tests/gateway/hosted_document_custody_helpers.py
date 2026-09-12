"""Document fixtures using only the canonical runtime's existing Files contract."""
from types import SimpleNamespace

from gateway import hosted_rooms
from gateway.hosted_room_attachments import HostedRoomAttachmentStore


def local_documents(tmp_path, *, transferred):
    db_path = tmp_path / 'state.db'
    hosted_rooms.create_room(db_path, room_id='room', name='Room', authority_gateway_id='home',
        members=[dict(member_id='member', profile='default', handle='member')])
    store = HostedRoomAttachmentStore(db_path)
    files = []
    for index in range(2):
        data = bytes([65 + index]) * 2048
        saved = store.put(room_id='room', upload_id=f'upload-{index}', kind='file', name=f'{index}.txt',
            mime='text/plain', data=data)
        files.append(({key: saved[key] for key in ('attachment_id', 'kind', 'name', 'size', 'mime')}, data))
    store.commit_message(room_id='room', event_id='source', manifest=[item for item, _ in files],
        recipient_member_ids=['member'])
    bound = [{**item, 'event_id': 'source'} for item, _ in files]
    rpc = SimpleNamespace(authority=SimpleNamespace(db=SimpleNamespace(db_path=db_path)), room_id='room', member_id='member')
    if transferred:
        rpc.hosted_attachment_data = [(item, data) for item, (_, data) in zip(bound, files)]
    return rpc, bound
