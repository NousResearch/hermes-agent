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
        # The actual model context preparation must expand the admitted document outside cwd.
        from agent.context_references import preprocess_context_references_async
        workspace = tmp_path / 'workspace'
        workspace.mkdir()
        prepared = owner._run(preprocess_context_references_async(row.payload['text'], cwd=workspace,
                             allowed_root=workspace, context_length=100000, plugins=False))
        assert content.decode().strip() in prepared.message
        assert prepared.injected_tokens > 0
        # Successful admission releases staging quota, not the document used by live/history context.
        from gateway.session_durable_send import _database
        db = _database(fence['profile_id'])
        staged = db.execute('SELECT size, length(body), retired, path FROM uploads WHERE id=?', (final['reference'],)).fetchone()
        db.close()
        assert staged[:3] == (0, 0, 1)
        from pathlib import Path
        assert Path(staged[3]).parent == Path(fence['profile_id']) / 'attachments'
        assert Path(staged[3]).read_bytes() == content
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
        assert not materialized.exists()  # disposable image staging released after immutable admission
        image_row = owner.persistence.admissions.find(principal_id=life.connection.actor.subject,
                                                     session_id=sid, request_id=image_id)
        assert Path(image_row.payload['attachments_v1']['media'][0]['path']).read_bytes() == image
        # Text-only sends still require a stable valid identity.
        bad = {**payload, 'submission_id': None, 'attachments': []}
        invalid = owner._run(life.connection.dispatch({'id': 1, 'method': 'durable.submit', 'params': bad}))
        assert 'error' in invalid
    finally:
        owner.close()


def test_submission_serialization_is_shared_by_distinct_authorities(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    from types import SimpleNamespace
    from gateway import session_durable_send as durable

    async def scenario():
        active = 0
        overlapped = False
        entered = asyncio.Event()
        release = asyncio.Event()

        @durable._serialized
        async def critical(connection, ref, params):
            nonlocal active, overlapped
            active += 1
            if active > 1:
                overlapped = True
            entered.set()
            await release.wait()
            active -= 1

        identity = {'submission_id': str(uuid.uuid4())}
        ref = SimpleNamespace(session_id='same-session')
        connections = [
            SimpleNamespace(authority=SimpleNamespace(profile_id=str(tmp_path)),
                            actor=SimpleNamespace(subject='same-principal'))
            for _ in range(2)
        ]
        first = asyncio.create_task(critical(connections[0], ref, identity))
        await entered.wait()
        second = asyncio.create_task(critical(connections[1], ref, identity))
        await asyncio.sleep(0.1)
        release.set()
        await asyncio.gather(first, second)
        assert not overlapped

    import asyncio
    asyncio.run(scenario())


def test_cancel_removes_document_materialized_before_path_record(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    from hermes_cli import _early_recovery
    monkeypatch.setattr(_early_recovery, 'restore_interrupted_pull', lambda root: False)
    from pathlib import Path
    from gateway import session_durable_send as durable
    owner = GatewayOwner(tmp_path, monkeypatch)
    life = owner._run(owner._open())

    def rpc(method, **params):
        result = owner._run(life.connection.dispatch({'id': 1, 'method': method, 'params': params}))
        assert 'error' not in result, result
        return result['result']

    try:
        description = rpc('durable.describe')
        fence = {key: description[key] for key in ('store_id', 'profile_id', 'principal_id')}
        sid = rpc('session.create', request_id=str(uuid.uuid4()), source='gui')['session_id']
        data = b'crash between materialization and path commit'
        file = {'id': str(uuid.uuid4()), 'name': 'crash.txt', 'mediaType': 'text/plain',
                'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        submission = str(uuid.uuid4())
        uploaded = rpc('durable.upload', owner=fence, session_id=sid, submission_id=submission,
                       file=file, offset=0, data=base64.b64encode(data).decode())
        from types import SimpleNamespace
        ref = SimpleNamespace(session_id=sid)
        key, metadata = durable._identity(life.connection, ref, submission, file)
        from gateway.session_policy import policy_for_source
        policy = policy_for_source(owner.authority.runner, owner.authority.sessions[sid].source)
        assert policy is not None
        session = {'cwd': policy.cwd, 'profile_home': fence['profile_id']}
        path, image = durable._materialize(fence['profile_id'], key, metadata, file, session)
        assert not image
        assert Path(path).read_bytes() == data
        db = durable._database(fence['profile_id'])
        db.execute('UPDATE uploads SET path=NULL WHERE id=?', (key,))
        db.commit()
        db.close()

        assert rpc('durable.cancel', owner=fence, session_id=sid, submission_id=submission,
                   files=[file]) == {'cancelled': True}
        assert not Path(path).exists()
        assert uploaded['reference'] == key
    finally:
        owner.close()


def test_staging_quota_reuse_and_cancellation_reconciliation(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    from hermes_cli import _early_recovery
    monkeypatch.setattr(_early_recovery, 'restore_interrupted_pull', lambda root: False)
    from gateway import session_durable_send as durable
    monkeypatch.setattr(durable, '_MAX_TOTAL', 32)
    owner = GatewayOwner(tmp_path, monkeypatch)
    owner.authority._schedule_admitted = lambda *args, **kwargs: None
    life = owner._run(owner._open())
    def rpc(method, **params):
        result = owner._run(life.connection.dispatch({'id': 1, 'method': method, 'params': params}))
        assert 'error' not in result, result
        return result['result']
    try:
        d = rpc('durable.describe')
        fence = {key: d[key] for key in ('store_id', 'profile_id', 'principal_id')}
        sid = rpc('session.create', request_id=str(uuid.uuid4()), source='gui')['session_id']
        content = b'payload uses the entire quota!!!'
        assert len(content) == 32
        retained = []
        for _ in range(5):
            file = {'id': str(uuid.uuid4()), 'name': 'retained.txt', 'mediaType': 'text/plain',
                    'size': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
            mid = str(uuid.uuid4())
            upload = rpc('durable.upload', owner=fence, session_id=sid, submission_id=mid,
                         file=file, offset=0, data=base64.b64encode(content).decode())
            params = dict(owner=fence, session_id=sid, submission_id=mid, text='read',
                          attachments=[{**file, 'reference': upload['reference']}])
            ack = rpc('durable.submit', **params)
            assert rpc('durable.submit', **params)['admission_id'] == ack['admission_id']
            assert rpc('durable.cancel', owner=fence, session_id=sid, submission_id=mid, files=[file]) == {'cancelled': False}
            retained.append(params)
        db = durable._database(d['profile_id'])
        assert db.execute('SELECT SUM(size), SUM(length(body)) FROM uploads').fetchone() == (0, 0)
        from pathlib import Path
        for (path,) in db.execute('SELECT path FROM uploads'):
            assert Path(path).read_bytes() == content
        db.close()
        file = {**file, 'id': str(uuid.uuid4())}
        mid = str(uuid.uuid4())
        upload_args = dict(owner=fence, session_id=sid, submission_id=mid, file=file, offset=0,
                           data=base64.b64encode(content[:8]).decode())
        rpc('durable.upload', **upload_args)
        cancel_args = dict(owner=fence, session_id=sid, submission_id=mid, files=[file])
        assert rpc('durable.cancel', **cancel_args) == {'cancelled': True}
        # Lost cancellation ACK and new socket: tombstone survives, the first identity cannot revive.
        owner._run(life.connection.close())
        life = owner._run(owner._open())
        rpc('session.resume', session_id=sid, omit_messages=True)
        assert rpc('durable.cancel', **cancel_args) == {'cancelled': True}
        refused = owner._run(life.connection.dispatch({'id': 1, 'method': 'durable.upload', 'params': upload_args}))
        assert refused['error']['data']['reason'] == 'admission_conflict'
        db = durable._database(d['profile_id'])
        assert db.execute('SELECT COUNT(*) FROM uploads WHERE retired=0').fetchone()[0] == 0
        db.close()
        assert rpc('durable.submit', **retained[0])['admission_id']
    finally:
        owner.close()
