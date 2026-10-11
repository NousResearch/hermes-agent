"""A hosted-room submission holds the documents it captured until its admission commits.

Documents ride in the prompt text, so the authority's own submit lease does not cover them: an
older row's release that names the same content-addressed capture, landing between the capture
(submission_payload) and the admission commit, unlinked the file the new turn's prompt points at.
"""
from pathlib import Path


def test_hosted_document_captured_for_a_new_turn_survives_a_concurrent_release(hosted_owner, monkeypatch):
    from gateway import session_authority
    from gateway.hosted_room_attachments import HostedRoomAttachmentStore
    from gateway.hosted_room_driver import TaskIdentity
    from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
    from gateway.session_ingress_media import release_unheld_media
    from hermes_state_runtime import list_session_admissions

    authority, loop, principal, _ = hosted_owner
    rpc = HostedRoomAuthorityRPC(authority, loop, room_id='room', member_id='member', profile='default',
                                 principal=principal, authorize=lambda *args: True)
    coords = dict(profile='default', source='bot_room')
    sid = rpc.create(**coords, title='Group: room')['session_id']
    store = HostedRoomAttachmentStore(authority.db.db_path)
    saved = store.put(room_id='room', upload_id='upload', kind='file', name='note.txt', mime='text/plain',
                      data=b'exact bytes')
    manifest = [{k: saved[k] for k in ('attachment_id', 'kind', 'name', 'size', 'mime')}]
    store.commit_message(room_id='room', event_id='event', manifest=manifest, recipient_member_ids=['member'])

    # An older turn's cleanup releases the same capture after submission_payload captured it and
    # before the admission write commits (no committed row holds it yet).
    real = session_authority.admit_session_input

    def admit_after_a_release(db, **kwargs):
        path = kwargs['payload']['text'].split('file: ')[1].split('\n')[0]
        release_unheld_media(db, [{'path': path, 'sha256': Path(path).parent.name}], retain_history=False)
        return real(db, **kwargs)
    monkeypatch.setattr(session_authority, 'admit_session_input', admit_after_a_release)

    rpc.submit(**coords, session_id=sid, prompt='read', task=TaskIdentity('room', 'task', 'thread', 'turn'),
               execution_generation=1, on_terminal=lambda receipt: None,
               attachments=[{**manifest[0], 'event_id': 'event'}])
    (row,) = list_session_admissions(authority.db, session_id=sid, pending_only=False)
    path = Path(row['payload']['text'].split('file: ')[1].split('\n')[0])
    assert path.exists(), 'a concurrent release unlinked the document the admitted prompt names'
    assert path.read_bytes() == b'exact bytes'
