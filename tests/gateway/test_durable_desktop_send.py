"""Real AuthorityConnection + SQLite admission/upload ledger. No production sockets or model calls."""
import base64
import hashlib
import uuid

import pytest

from tests.fixtures.gateway_owner import GatewayOwner


def test_durable_upload_reconnect_and_exact_admission(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    # A linked worktree's .git lives under the real profile. This fixture must never run updater recovery.
    from hermes_cli import _early_recovery
    monkeypatch.setattr(_early_recovery, 'restore_interrupted_pull', lambda root: False)
    owner = GatewayOwner(tmp_path, monkeypatch)
    # Only the model scheduling boundary is held: all route, ACL, upload, create and admission writes are real.
    owner.authority._schedule_admitted = lambda *args, **kwargs: None
    life = owner._run(owner._open())
    def rpc(method, **params):
        result = owner._run(life.connection.dispatch({'id': 1, 'method': method, 'params': params}))
        assert 'error' not in result, result
        return result['result']
    try:
        description = rpc('durable.describe')
        assert 'durable-desktop-send-v1' in description['capabilities']
        fence = {key: description[key] for key in ('store_id', 'profile_id', 'principal_id')}
        create_id = str(uuid.uuid4())
        created = rpc('session.create', request_id=create_id, source='gui')
        sid = created['session_id']
        assert rpc('session.create', request_id=create_id, source='gui')['session_id'] == sid
        content = b'attachment preserved across disconnection\n'
        file = {'id': str(uuid.uuid4()), 'name': 'original.txt', 'mediaType': 'text/plain',
                'size': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
        submission = str(uuid.uuid4())
        args = {'owner': fence, 'session_id': sid, 'submission_id': submission, 'file': file}
        first = rpc('durable.upload', **args, offset=0, data=base64.b64encode(content[:9]).decode())
        assert first['offset'] == 9 and not first['complete']
        # Lost upload reply + socket replacement: same chunk is reconciled, not appended twice.
        owner._run(life.connection.close())
        life = owner._run(owner._open())
        rpc('session.resume', session_id=sid, omit_messages=True)
        assert rpc('durable.upload', **args, offset=0, data=base64.b64encode(content[:9]).decode()) == first
        final = rpc('durable.upload', **args, offset=9, data=base64.b64encode(content[9:]).decode())
        assert final['complete'] and final['sha256'] == file['sha256']
        payload = {'owner': fence, 'session_id': sid, 'submission_id': submission, 'text': 'Read this file',
                   'attachments': [{**file, 'reference': final['reference']}]}
        accepted = rpc('durable.submit', **payload)
        # Lost admission ACK: a new connection repeats the frozen payload against the real ledger.
        owner._run(life.connection.close())
        life = owner._run(owner._open())
        rpc('session.resume', session_id=sid, omit_messages=True)
        replay = rpc('durable.submit', **payload)
        assert replay['admission_id'] == accepted['admission_id']
        row = owner.persistence.admissions.find(principal_id=life.connection.actor.subject,
                                                session_id=sid, request_id=submission)
        assert row is not None
        conflict = owner._run(life.connection.dispatch({'id': 1, 'method': 'durable.submit',
                                                       'params': {**payload, 'text': 'changed'}}))
        assert conflict['error']['data']['reason'] == 'admission_conflict'
        forged = {**payload, 'attachments': [{**file, 'reference': '0' * 64}]}
        refused = owner._run(life.connection.dispatch({'id': 1, 'method': 'durable.submit', 'params': forged}))
        assert refused['error']['data']['reason'] == 'permission_denied'
        # Neither a valid reference nor a negotiated identity authorizes a different destination.
        other = rpc('session.create', request_id=str(uuid.uuid4()), source='gui')['session_id']
        for changed in ({**payload, 'session_id': other},
                        {**payload, 'owner': {**fence, 'principal_id': 'another-human'}},
                        {**payload, 'owner': {**fence, 'store_id': 'another-store'}}):
            answer = owner._run(life.connection.dispatch({'id': 1, 'method': 'durable.submit', 'params': changed}))
            assert answer['error']['data']['reason'] == 'permission_denied'
        # An original image is materialized with its real MIME, never from a thumbnail.
        image = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aPZkAAAAASUVORK5CYII=')
        image_file = {'id': str(uuid.uuid4()), 'name': 'original.png', 'mediaType': 'image/png',
                      'size': len(image), 'sha256': hashlib.sha256(image).hexdigest()}
        image_id = str(uuid.uuid4())
        uploaded = rpc('durable.upload', owner=fence, session_id=sid, submission_id=image_id,
                       file=image_file, offset=0, data=base64.b64encode(image).decode())
        image_payload = {'owner': fence, 'session_id': sid, 'submission_id': image_id, 'text': 'Original image',
                         'attachments': [{**image_file, 'reference': uploaded['reference']}]}
        image_ack = rpc('durable.submit', **image_payload)
        assert rpc('durable.submit', **image_payload)['admission_id'] == image_ack['admission_id']
        from pathlib import Path
        materialized = Path(fence['profile_id']) / 'cache' / 'images' / (
            f'durable_{uploaded["reference"]}_{image_file["sha256"]}.png')
        assert materialized.read_bytes() == image
        # Text-only sends still require a stable valid identity.
        bad = {**payload, 'submission_id': None, 'attachments': []}
        invalid = owner._run(life.connection.dispatch({'id': 1, 'method': 'durable.submit', 'params': bad}))
        assert 'error' in invalid
    finally:
        owner.close()
