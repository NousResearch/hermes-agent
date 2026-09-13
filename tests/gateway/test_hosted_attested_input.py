"""Digest-only preclaim reconciliation over preconstructed SQLite metadata.

No owner startup, admission, task execution, lease, settlement or retirement.
"""
from contextlib import contextmanager, nullcontext
import hashlib
import json
import sqlite3
from types import SimpleNamespace

import pytest


@contextmanager
def _accepted(tmp_path, monkeypatch, namespace):
    from gateway import session_hosted_transport as transport
    from gateway.hosted_room_input_reclamation import copy_path
    from gateway.session_admission import admission_fingerprint
    from gateway.session_contract import SessionRef
    from gateway.session_ingress_media import _media_root
    from hermes_state_input_custody import create_schema

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    db = SimpleNamespace(db_path=tmp_path / 'state.db', _read_ctx=lambda: nullcontext(conn))
    authority = SimpleNamespace(db=db, profile_id=str(tmp_path))
    binding = {'source_home': str(tmp_path / 'source'), 'target_home': str(tmp_path),
               'selector': {'room_id': 'room', 'member_id': 'member', 'profile': 'default'},
               'owner': 'owner'}
    principal = transport._principal(authority, binding).subject
    identity = {'room_id': 'room', 'task_id': 'task', 'thread_id': 'thread', 'turn_id': 'turn'}
    request_id = 'hosted:' + json.dumps([identity, 1])
    create_schema(conn)
    conn.execute('CREATE TABLE state_meta(key TEXT PRIMARY KEY,value TEXT)')
    conn.execute('''CREATE TABLE session_admissions(admission_id TEXT,principal_id TEXT,
        target_session_id TEXT,request_id TEXT,payload_json TEXT,payload_digest TEXT,intent TEXT)''')
    paths, manifest, digests = [], [], []
    for ordinal, name in enumerate(('first.txt', 'second.txt')):
        data = ('private-' + name).encode()
        digest = hashlib.sha256(data).hexdigest()
        copy = dict(namespace=namespace, name=name, digest=digest)
        path = copy_path(db, copy) if namespace == 'v3' else _media_root() / digest / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        paths.append(path)
        manifest.append(dict(attachment_id='att_' + format(ordinal, '032x'), event_id='event',
                             kind='file', name=name, mime='text/plain', size=len(data)))
        digests.append(digest)
        if namespace == 'v3':
            st = path.stat()
            conn.execute('INSERT INTO input_custody_copies VALUES(?,?,?,?,?,?,?,?,?)',
                         (str(ordinal), 'v3', name, digest, len(data), 1, 'ready', str(st.st_dev), str(st.st_ino)))
    payload = {'text': 'original' + ''.join('\n[Shared attachment] file: ' + str(path) + '\n' for path in paths),
               'local_operator_v1': {'fixture': 'immutable-authority-data'}}
    digest = admission_fingerprint(canonical_target='s', payload={'input': payload, 'intent': 'queue'})
    conn.execute('INSERT INTO session_admissions VALUES(?,?,?,?,?,?,?)',
                 ('admitted', principal, 's', request_id, json.dumps(payload), digest, 'queue'))
    if namespace == 'v3':
        for ordinal in range(len(paths)):
            conn.execute('INSERT INTO input_custody_refs VALUES(?,?,?,?,?,?,?,?,?)',
                         ('admitted', ordinal, str(ordinal), 1, principal, 's', request_id, digest, 'queue'))
    conn.execute('INSERT INTO state_meta VALUES(?,?)', (transport._BINDING + 's', json.dumps(binding)))
    attested = dict(owner='owner', prompt='original', attachments=manifest, attachment_digests=digests)
    operations = []
    def attest(binding, operation, params):
        operations.append(operation)
        assert operation == 'execute', 'preclaim may not transfer another byte batch'
        return attested
    monkeypatch.setattr(transport, '_attest', attest)
    row = dict(admission_id='admitted', principal_id=principal, target_session_id='s',
               request_id=request_id, payload=payload)
    try:
        yield transport, authority, SessionRef(str(tmp_path), 's'), row, attested, paths, conn, operations
    finally:
        conn.close()


@pytest.mark.parametrize('namespace', ['v3', 'legacy'])
def test_preclaim_preserves_exact_accepted_paths_and_identity_without_transfer(tmp_path, monkeypatch, namespace):
    with _accepted(tmp_path, monkeypatch, namespace) as (transport, authority, ref, row, attested, paths, conn, operations):
        before = list(conn.iterdump())
        assert transport._check_remote_hosted_admission(authority, ref, row)
        assert operations == ['execute']
        assert list(conn.iterdump()) == before
        assert all(path.exists() for path in paths)


@pytest.mark.parametrize('mutation', ['digest', 'prompt', 'missing-copy', 'reference-identity'])
def test_preclaim_refuses_changed_attestation_or_custody_without_repair(tmp_path, monkeypatch, mutation):
    from hermes_state_runtime import RuntimeStoreError
    with _accepted(tmp_path, monkeypatch, 'v3') as (transport, authority, ref, row, attested, paths, conn, operations):
        if mutation == 'digest':
            attested['attachment_digests'][1] = '0' * 64
        elif mutation == 'prompt':
            attested['prompt'] = 'substituted'
        elif mutation == 'missing-copy':
            paths[1].unlink()
        else:
            conn.execute("UPDATE input_custody_refs SET request_id='wrong' WHERE ordinal=1")
        before = list(conn.iterdump())
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            transport._check_remote_hosted_admission(authority, ref, row)
        assert operations == ['execute'] and list(conn.iterdump()) == before
        assert paths[0].exists()
        if mutation == 'missing-copy':
            assert not paths[1].exists()
