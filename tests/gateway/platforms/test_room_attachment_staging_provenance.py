"""Private staging evidence from registered HTTP writes, not cleanup authority."""
import asyncio
import base64
import hashlib
import hmac
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.hosted_room_peer import attachment_manifest_digest, decode_room_grant, issue_room_grant
from gateway.platforms import api_server_room_attachments as attachments_api
from tests.gateway.test_canonical_peer_files_target import (
    files_state,
    files_target,  # noqa: F401
    inprocess_http,
    source_files,
)
from tests.gateway.test_canonical_peer_target_setup import invite, target  # noqa: F401
from tests.gateway.test_canonical_peer_text_admission import dispatch
from tui_gateway.hosted_room_peer_attachments import bound_attachment_payloads
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError


_ORIGIN_CLAIMS = (
    '_token_sha256', 'grant_id', 'issued_at', 'expires_at', 'status_expires_at',
    'room_id', 'home_install_id', 'authority_gateway_id', 'authority_epoch',
    'member_id', 'target_install_id', 'target_profile',
)


def expected_origin(target, claims):
    return dict(version=1, **{key: claims[key] for key in _ORIGIN_CLAIMS if key != 'status_expires_at'},
                status_expires_at=claims.get('status_expires_at', claims['expires_at']),
                receiver_home=target.authority.profile_id,
                receiver_epoch=target.authority.epoch,
                receiver_instance_id=target.authority.instance_id)


def saved_origin(case):
    with sqlite3.connect(case.spool.db_path) as conn:
        conn.row_factory = sqlite3.Row
        found = conn.execute('SELECT * FROM roomlink_attachment_batches WHERE task_id=?',
                             (case.value.task_id,)).fetchone()
        if found is None:
            return None, 0
        row = dict(found)
    encoded = row.get('staging_origin_json')
    return (json.loads(encoded) if encoded is not None else None,
            row.get('staging_origin_ambiguous', 0))


def history_schema(case):
    """Pre-existing metadata schema makes writer RED independent of migration."""
    with sqlite3.connect(case.spool.db_path) as conn:
        columns = {row[1] for row in conn.execute('PRAGMA table_info(roomlink_attachment_batches)')}
        if 'staging_origin_json' not in columns:
            conn.execute('ALTER TABLE roomlink_attachment_batches ADD COLUMN staging_origin_json TEXT')
        if 'staging_origin_ambiguous' not in columns:
            conn.execute('''ALTER TABLE roomlink_attachment_batches ADD COLUMN
                staging_origin_ambiguous INTEGER NOT NULL DEFAULT 0 CHECK (staging_origin_ambiguous IN (0,1))''')


async def case_for(target, monkeypatch):
    issued = await invite(target)
    store, _, refs, raw = source_files(target.home, mixed=True)
    pending = bound_attachment_payloads(store, 'room-one', 'member-one', refs)
    manifest = [{k: v for k, v in item.items() if k != 'data'} for item in pending]
    value = replace(dispatch(issued), attachment_manifest_digest=attachment_manifest_digest(manifest))
    claims = decode_room_grant(target.adapter._room_grant_secret(), issued['grant'], permission='attachment.stage')
    spool = attachments_api._default_spool()
    # Stable time makes rows/fences comparable without changing grant verification.
    monkeypatch.setattr(spool, 'clock', lambda: float(claims['issued_at']))
    responses = []
    calls = inprocess_http(target, monkeypatch, responses=responses)
    return SimpleNamespace(issued=issued, claims=claims, value=value, spool=spool, manifest=manifest,
        raw=raw, responses=responses, calls=calls,
        client=PeerRunsHTTPClient(base_url='http://127.0.0.1:8642', api_key=''))


async def manifest_request(case, *, grant=None, value=None, manifest=None, extra=None):
    body = {'hosted_room_dispatch': (value or case.value).as_mapping(),
            'attachments': case.manifest if manifest is None else manifest}
    body.update(extra or {})
    return await asyncio.to_thread(case.client._request, '/v1/room-members/attachments',
        method='POST', body=body, room_grant=grant or case.issued['grant'], reject_redirects=True)


async def put_request(case, *, grant=None, index=0, data=None):
    return await asyncio.to_thread(case.client._put_attachment,
        f'/v1/room-members/attachments/{case.value.task_id}/1/{case.manifest[index]["attachment_id"]}',
        data=case.raw[index] if data is None else data, grant=grant or case.issued['grant'])


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['prepare', 'put', 'discard'])
@pytest.mark.parametrize('expired', [False, True])
async def test_registered_handlers_preserve_legacy_grant_horizon(files_target, monkeypatch, operation, expired):
    case = await case_for(files_target, monkeypatch)
    # The real decoder supports this older signed shape. Do not mock its result.
    payload = {key: value for key, value in case.claims.items()
               if key not in {'_token_sha256', 'status_expires_at'}}
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('ascii')
    encode = lambda data: base64.urlsafe_b64encode(data).rstrip(b'=').decode('ascii')
    legacy = encode(encoded) + '.' + encode(hmac.new(
        files_target.adapter._room_grant_secret(), encoded, hashlib.sha256).digest())
    claims = decode_room_grant(files_target.adapter._room_grant_secret(), legacy, permission='attachment.stage')
    assert 'status_expires_at' not in claims
    assert claims['_token_sha256'] == hashlib.sha256(legacy.encode('ascii')).hexdigest()
    if operation != 'prepare':
        await manifest_request(case)
    if operation == 'discard':
        await put_request(case)
        await put_request(case, index=1)
    before = files_state(files_target, case.spool)

    async def attempt():
        if operation == 'prepare':
            return await manifest_request(case, grant=legacy)
        if operation == 'put':
            return await put_request(case, grant=legacy)
        return await asyncio.to_thread(case.client.discard_attachments, task_id=case.value.task_id,
                                       execution_generation=1, grant=legacy)

    if expired:
        monkeypatch.setattr('gateway.hosted_room_peer.time.time', lambda: claims['expires_at'] + 1)
        with pytest.raises(PeerRunsHTTPError) as denied:
            await attempt()
        assert denied.value.status_code == 401
        assert files_state(files_target, case.spool) == before
        return

    result = await attempt()
    if operation == 'prepare':
        first = expected_origin(files_target, claims)
        assert first['status_expires_at'] == claims['expires_at']
        assert saved_origin(case) == (first, 0)
        assert result['idempotent'] is False
        assert (await manifest_request(case, grant=legacy))['idempotent'] is True
        await put_request(case, grant=legacy)
        assert (await put_request(case, grant=legacy, index=1))['complete'] is True
        assert (await put_request(case, grant=legacy))['idempotent'] is True
        assert saved_origin(case) == (first, 0)
    elif operation == 'put':
        assert saved_origin(case) == (expected_origin(files_target, case.claims), 1)
        assert (await put_request(case, grant=legacy))['idempotent'] is True
        assert saved_origin(case) == (expected_origin(files_target, case.claims), 1)
    else:
        assert result['removed'] == 1
        after = files_state(files_target, case.spool)
        after_staging, before_staging = after['staging'], before['staging']
        assert isinstance(after_staging, dict) and isinstance(before_staging, dict)
        assert after_staging['roomlink_attachment_batches'] == [] and after['bytes'] == {}
        assert after_staging['roomlink_attachment_attempt_fences'] == before_staging['roomlink_attachment_attempt_fences']
    assert 'status_expires_at' not in claims


def renewed_token(target, case):
    signer = {key: case.claims[key] for key in (
        'grant_id', 'room_id', 'home_install_id', 'authority_gateway_id', 'authority_epoch',
        'member_id', 'target_install_id', 'target_profile', 'execution_policy_digest', 'permissions')}
    token = issue_room_grant(target.adapter._room_grant_secret(), **signer,
        issued_at=case.claims['issued_at'] + 1,
        ttl_seconds=case.claims['expires_at'] - case.claims['issued_at'] - 1,
        status_expires_at=case.claims['status_expires_at'])
    claims = decode_room_grant(target.adapter._room_grant_secret(), token, permission='attachment.stage')
    assert claims['grant_id'] == case.claims['grant_id'] and claims['_token_sha256'] != case.claims['_token_sha256']
    return token


@contextmanager
def observe_commit(case, target, monkeypatch, *, abort=False):
    """Inspect real transaction visibility/locks, never supply grant evidence."""
    transaction = case.spool._transaction
    observed = []
    @contextmanager
    def checked(*, immediate=False):
        with transaction(immediate=immediate) as conn:
            yield conn
            if immediate:
                # Prune uses its own transaction without the profile fence. Only
                # inspect an operation after its actual current-grant checks.
                if not armed:
                    return
                inside = conn.execute('SELECT staging_origin_json, staging_origin_ambiguous '
                                      'FROM roomlink_attachment_batches WHERE task_id=?',
                                      (case.value.task_id,)).fetchone()
                for name in ('shared-state.db', 'state.db'):
                    with sqlite3.connect(target.home / name, timeout=0) as other:
                        with pytest.raises(sqlite3.OperationalError, match='locked'):
                            other.execute('BEGIN IMMEDIATE')
                observed.append(tuple(inside) if inside else None)
                assert saved_origin(case) == before
                if abort:
                    raise sqlite3.IntegrityError('synthetic precommit rollback')
    from gateway import session_peer_target
    require_current = session_peer_target.require_current_grant
    armed = False
    def current(conn, claims):
        nonlocal armed
        require_current(conn, claims)
        if Path(conn.execute('PRAGMA database_list').fetchone()[2]).name == 'state.db':
            armed = True
    before = saved_origin(case)
    with monkeypatch.context() as patch:
        patch.setattr(case.spool, '_transaction', checked)
        patch.setattr(session_peer_target, 'require_current_grant', current)
        yield observed


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['prepare', 'put', 'idempotent-put', 'repair-put'])
@pytest.mark.parametrize('change', ['token', 'receiver', 'unknown'])
async def test_http_origin_is_immutable_and_ambiguity_monotonic(files_target, monkeypatch, operation, change):
    case = await case_for(files_target, monkeypatch)
    history_schema(case)
    first = expected_origin(files_target, case.claims)
    with observe_commit(case, files_target, monkeypatch) as observed:
        prepared = await manifest_request(case)
    assert len(observed) == 1 and json.loads(observed[0][0]) == first and observed[0][1] == 0
    assert prepared['idempotent'] is False
    assert saved_origin(case) == (first, 0)  # Semantic writer RED with schema already present.
    await put_request(case)
    if operation != 'put':
        await put_request(case, index=1)
    before = files_state(files_target, case.spool)
    assert (await manifest_request(case))['idempotent'] is True
    assert (await put_request(case))['idempotent'] is True
    assert files_state(files_target, case.spool) == before

    token = case.issued['grant']
    if change == 'token':
        token = renewed_token(files_target, case)
    elif change == 'receiver':
        from gateway.session_authorities import SessionAuthorities
        from gateway.session_authority import SessionAuthority
        from hermes_state_runtime import begin_runtime_epoch
        # Data-only retained-owner replacement; no runtime startup or executor.
        owner = SessionAuthority(files_target.runner, profile_id=str(files_target.home),
            instance_id='successor', db=files_target.db,
            epoch=begin_runtime_epoch(files_target.db, instance_id='successor'))
        registry = SessionAuthorities(files_target.home)
        registry.add(files_target.home, owner)
        files_target.runner.session_authorities = registry
        files_target.runner.session_authority = files_target.authority = owner
        assert expected_origin(files_target, case.claims) != first
    index = 1 if operation == 'put' else 0
    if operation == 'repair-put':
        case.spool._file_path(prepared['batch_key'], case.manifest[index]['attachment_id']).unlink()
    if change == 'unknown':
        if operation == 'prepare':
            case.spool.prepare(case.value, case.manifest)
        else:
            case.spool.put(claims=case.claims, task_id=case.value.task_id, execution_generation=1,
                attachment_id=case.manifest[index]['attachment_id'], data=case.raw[index])
    else:
        with observe_commit(case, files_target, monkeypatch) as observed:
            result = (await manifest_request(case, grant=token) if operation == 'prepare'
                      else await put_request(case, grant=token, index=index))
        assert len(observed) == 1 and observed[0][1] == 1
        if operation == 'repair-put':
            assert result['repaired'] is True
    assert saved_origin(case) == (first, 1)
    await manifest_request(case)
    await put_request(case, index=index)
    assert saved_origin(case) == (first, 1)
    reopened = attachments_api.RoomAttachmentSpool(case.spool.db_path, root=case.spool.root, clock=case.spool.clock)
    assert saved_origin(SimpleNamespace(spool=reopened, value=case.value)) == (first, 1)
    assert len(files_state(files_target, case.spool)['staging']['roomlink_attachment_batches']) == 1
    if operation != 'put':
        assert [Path(item['path']).read_bytes() for item in reopened.materialize(case.value)] == case.raw
    for _, _, _, body in case.responses:
        encoded = json.dumps(body)
        assert not any(secret in encoded for secret in (
            token, case.issued['grant'], case.claims['_token_sha256'], files_target.authority.profile_id,
            'synthetic-target-api-key', case.value.prompt))
    assert set(first) == {'version', *_ORIGIN_CLAIMS, 'receiver_home', 'receiver_epoch', 'receiver_instance_id'}
    assert ('POST', '/v1/room-members/attachments') in case.calls
    assert any(method == 'PUT' for method, _ in case.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize('scenario', [
    'legacy', 'direct', 'home-history', 'instance-history', 'epoch-history',
    'manifest-conflict', 'bytes-conflict', 'forged-origin',
    'prepare-rollback', 'prepare-new-rollback', 'put-rollback', 'publish-failure', 'repair-failure',
    'prepare-quota', 'put-quota', 'prepare-lock', 'put-lock',
    'prepare-stale', 'put-stale', 'prepare-owner', 'put-owner', 'prepare-policy', 'put-policy',
    'prepare-epoch', 'put-epoch', 'http-exact', 'http-scope',
    pytest.param('native-exact', marks=pytest.mark.linux_only),
    pytest.param('native-scope', marks=pytest.mark.linux_only),
    'accepted-disposal',
])
async def test_provenance_preserves_unknown_history_and_transaction_boundaries(files_target, monkeypatch, scenario):
    case = await case_for(files_target, monkeypatch)
    history_schema(case)
    first = expected_origin(files_target, case.claims)
    if scenario in {'legacy', 'direct'}:
        case.spool.prepare(case.value, case.manifest)
        await put_request(case)
        if scenario == 'legacy':
            # An actual old-shape spool with rows/bytes/fences, not inferred history.
            with sqlite3.connect(case.spool.db_path) as conn:
                conn.execute('ALTER TABLE roomlink_attachment_batches DROP COLUMN staging_origin_json')
                conn.execute('ALTER TABLE roomlink_attachment_batches DROP COLUMN staging_origin_ambiguous')
            attachments_api.RoomAttachmentSpool(case.spool.db_path, root=case.spool.root, clock=case.spool.clock)
            with sqlite3.connect(case.spool.db_path) as conn:
                row = conn.execute('SELECT staging_origin_json, staging_origin_ambiguous FROM roomlink_attachment_batches').fetchone()
                assert row == (None, 0)
                with pytest.raises(sqlite3.IntegrityError):
                    conn.execute('UPDATE roomlink_attachment_batches SET staging_origin_ambiguous=2')
        await manifest_request(case)
        await put_request(case, index=1)
        await manifest_request(case, grant=renewed_token(files_target, case))
        assert saved_origin(case)[0] is None
        assert [Path(item['path']).read_bytes() for item in case.spool.materialize(case.value)] == case.raw
        return

    await manifest_request(case)
    # Seed durable known history independently of the missing writer. These
    # cases also run RED on base for monotonic/isolation semantics, not DDL.
    with sqlite3.connect(case.spool.db_path) as conn:
        conn.execute('UPDATE roomlink_attachment_batches SET staging_origin_json=?',
                     (json.dumps(first, sort_keys=True, separators=(',', ':')),))
    if scenario.endswith('-history'):
        key = {'home-history': 'receiver_home', 'instance-history': 'receiver_instance_id',
               'epoch-history': 'receiver_epoch'}[scenario]
        prior = first | {key: first[key] + 1 if key == 'receiver_epoch' else first[key] + '-prior'}
        with sqlite3.connect(case.spool.db_path) as conn:
            conn.execute('UPDATE roomlink_attachment_batches SET staging_origin_json=?',
                         (json.dumps(prior, sort_keys=True, separators=(',', ':')),))
        await put_request(case)
        assert saved_origin(case) == (prior, 1)
        await manifest_request(case)
        assert saved_origin(case) == (prior, 1)
        return

    if scenario.startswith(('http-', 'native-')) or scenario == 'accepted-disposal':
        await put_request(case)
        await put_request(case, index=1)
        if scenario == 'accepted-disposal':
            from gateway.platforms import api_server_runs
            launched = []
            async def inert(adapter, launch, **kwargs):
                launched.append(launch.admission)
            monkeypatch.setattr(api_server_runs, '_execute_run', inert)
            accepted = await asyncio.to_thread(case.client.dispatch, dispatch=case.value.as_mapping(), grant=case.issued['grant'])
            await asyncio.sleep(0)
            assert len(launched) == 1
            payload = launched[0][2]['payload']
            assert payload['text'] == case.value.prompt
            before = files_state(files_target, case.spool)
            saved = [tuple(row) for row in files_target.db._conn.execute('SELECT * FROM session_admissions')]
            result = await asyncio.to_thread(case.client.discard_attachments, task_id=case.value.task_id,
                execution_generation=1, grant=case.issued['grant'])
            assert result['removed'] == 1
            after = files_state(files_target, case.spool)
            assert after['staging']['roomlink_attachment_batches'] == [] and after['bytes'] == {}
            assert after['staging']['roomlink_attachment_attempt_fences'] == before['staging']['roomlink_attachment_attempt_fences']
            assert after['accepted_bytes'] == before['accepted_bytes'] and list(after['accepted_bytes'].values()) == case.raw
            assert after['custody'] == before['custody'] and len(after['custody']) == 1
            replay = await asyncio.to_thread(case.client.dispatch, dispatch=case.value.as_mapping(), grant=case.issued['grant'])
            assert replay['run_id'] == accepted['run_id'] and len(launched) == 1
            assert [tuple(row) for row in files_target.db._conn.execute('SELECT * FROM session_admissions')] == saved
        else:
            replacement = renewed_token(files_target, case)
            before = files_state(files_target, case.spool)
            exact = scenario.endswith('exact')
            if scenario.startswith('native'):
                result = await files_target.connection.dispatch(dict(id=2,
                    method='groups.peer.revoke_exact' if exact else 'groups.peer.revoke',
                    params={'grant': case.issued['grant']}))
                assert result['result']['revoked'] is True
            else:
                result = await asyncio.to_thread(case.client.revoke_grant_exact if exact else case.client.revoke_grant,
                                                 grant=case.issued['grant'])
                assert result['revoked'] is True
            after = files_state(files_target, case.spool)
            if scenario.startswith('native'):
                assert after['bytes'] == {}
                with sqlite3.connect(case.spool.db_path) as conn:
                    assert conn.execute('SELECT complete FROM roomlink_attachment_batches').fetchall() == [(0,)]
                    assert conn.execute('SELECT stored FROM roomlink_attachment_files').fetchall() == [(0,), (0,)]
                assert saved_origin(case) == (first, 0)
                assert after['staging']['roomlink_attachment_attempt_fences'] == before['staging']['roomlink_attachment_attempt_fences']
                assert after['custody'] == before['custody'] and after['copies'] == before['copies']
                assert after['accepted_bytes'] == before['accepted_bytes']
                if exact:
                    assert (await manifest_request(case, grant=replacement))['complete'] is False
                    await put_request(case, grant=replacement)
                    await put_request(case, grant=replacement, index=1)
                    assert saved_origin(case) == (first, 1)
            else:
                assert {k: v for k, v in after.items() if k != 'reservations'} == {k: v for k, v in before.items() if k != 'reservations'}
                if exact:
                    await put_request(case, grant=replacement)
                    assert saved_origin(case) == (first, 1)
        return

    token = renewed_token(files_target, case)
    is_prepare = scenario.startswith('prepare') or scenario in {'manifest-conflict', 'forged-origin'}
    if scenario == 'prepare-new-rollback':
        case.value = replace(case.value, task_id='rolled-back-new-attempt')
    if scenario == 'repair-failure':
        await put_request(case)
        key = attachments_api._batch_key(case.value)
        case.spool._file_path(key, case.manifest[0]['attachment_id']).unlink()
    before = files_state(files_target, case.spool)
    async def attempt():
        if scenario == 'manifest-conflict':
            manifest = [case.manifest[0] | {'name': 'different.txt'}, case.manifest[1]]
            return await manifest_request(case, grant=token, manifest=manifest,
                value=replace(case.value, attachment_manifest_digest=attachment_manifest_digest(manifest)))
        if scenario == 'forged-origin':
            return await manifest_request(case, grant=token, extra={'staging_origin_json': first})
        if is_prepare:
            return await manifest_request(case, grant=token,
                value=replace(case.value, task_id='new-attempt') if scenario == 'prepare-quota' else None)
        return await put_request(case, grant=token, data=b'wrong bytes' if scenario == 'bytes-conflict' else None)

    if scenario.endswith('-rollback'):
        with observe_commit(case, files_target, monkeypatch, abort=True) as observed:
            with pytest.raises(PeerRunsHTTPError) as denied:
                await attempt()
        assert len(observed) == 1 and observed[0][0] is not None
        assert observed[0][1] == (0 if scenario == 'prepare-new-rollback' else 1)
        assert denied.value.status_code == 500
    elif scenario.endswith('-lock'):
        from gateway import hosted_rooms
        original = hosted_rooms._connect
        def impatient(path):
            conn = original(path)
            conn.execute('PRAGMA busy_timeout=0')
            return conn
        monkeypatch.setattr(hosted_rooms, '_connect', impatient)
        with sqlite3.connect(files_target.home / 'state.db') as lock:
            lock.execute('BEGIN IMMEDIATE')
            with pytest.raises(PeerRunsHTTPError) as denied:
                await attempt()
            lock.rollback()
        assert denied.value.status_code == 500
    else:
        if scenario.endswith('-failure'):
            def failed_publication(*args, **kwargs):
                raise OSError('synthetic publication failure')
            monkeypatch.setattr(case.spool, '_write_atomic', failed_publication)
        elif scenario.endswith('-quota'):
            monkeypatch.setattr(attachments_api, 'MAX_SPOOL_BATCHES' if is_prepare else 'MAX_SPOOL_BYTES', 0)
        elif scenario.endswith(('-stale', '-owner', '-policy', '-epoch')):
            initial = files_target.adapter._room_grant_claims
            def interleave(request, *, permission):
                claims = initial(request, permission=permission)
                if scenario.endswith('-stale'):
                    from gateway import hosted_rooms
                    hosted_rooms.revoke_room_grant_id(files_target.home / 'state.db', claims=claims,
                                                      expires_at=claims['status_expires_at'])
                elif scenario.endswith('-owner'):
                    files_target.runner.session_authorities._by_key.clear()
                elif scenario.endswith('-epoch'):
                    from hermes_state_runtime import begin_runtime_epoch
                    begin_runtime_epoch(files_target.db, instance_id='stale-owner')
                else:
                    (files_target.home / 'config.yaml').write_text('agent:\n  max_turns: 3\napprovals:\n  mode: manual\n')
                return claims
            monkeypatch.setattr(files_target.adapter, '_room_grant_claims', interleave)
        with pytest.raises(PeerRunsHTTPError) as denied:
            await attempt()
        expected = (500 if scenario.endswith('-failure') else 401 if scenario.endswith(('-stale', '-owner', '-policy', '-epoch'))
                    else 409 if scenario.endswith('-conflict') else 400)
        assert denied.value.status_code == expected
    after = files_state(files_target, case.spool)
    if scenario == 'put-rollback':
        # Existing spool publication precedes SQL commit. A failed commit can
        # leave verified private bytes, but neither stored=1 nor provenance.
        assert {k: v for k, v in after.items() if k != 'bytes'} == {k: v for k, v in before.items() if k != 'bytes'}
        assert list(after['bytes'].values()) == [case.raw[0]]
        assert saved_origin(case) == (first, 0)
        await put_request(case)
        assert saved_origin(case) == (first, 0)
    else:
        assert after == before
        assert saved_origin(case) == (None if scenario == 'prepare-new-rollback' else first, 0)
