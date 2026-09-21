"""Retry obligations survive owner SQL faults and refuse changed authority."""
import asyncio
from dataclasses import replace
import json
import os
import sqlite3

import pytest
from tests.gateway.test_canonical_peer_target_setup import target  # noqa: F401
from tests.gateway.test_canonical_peer_files_target import files_target  # noqa: F401
from tests.gateway.peer_output_fixtures import peer_case
from tests.gateway.test_peer_output_retry import tick, pending, clock, supersede
from tests.gateway.test_peer_output_retry_recovery import renew
from gateway.run import _profile_runtime_scope


@pytest.mark.asyncio
async def test_completion_commit_fault_keeps_retryable_exact_ack_obligation(files_target, monkeypatch):
    async with peer_case(files_target, monkeypatch) as c:
        now = clock(c)
        c.db._execute_write(lambda conn: conn.execute("CREATE TRIGGER reject_completion BEFORE INSERT ON hosted_room_artifact_completions BEGIN SELECT RAISE(ABORT,'inert completion fault'); END"))
        with pytest.raises(sqlite3.Error):
            await tick(c)
        retry, = pending(c)
        assert retry['blocked'] == 0, 'owner SQL failure must not be misclassified as permanent remote denial'
        assert c.db._conn.execute('SELECT count(*) FROM hosted_room_artifact_completions').fetchone()[0] == 0
        before = list(c.wire.calls)
        await tick(c)
        assert c.wire.calls == before
        c.db._execute_write(lambda conn: conn.execute('DROP TRIGGER reject_completion'))
        now[0] = retry['next_attempt_at']
        await tick(c)
        assert pending(c) == []
        assert len([e for e in c.service._events('room-one') if e['kind'] == 'message.member']) == 1
        assert len(c.executions) == len(c.launched) == 1


@pytest.mark.asyncio
async def test_discard_disk_fault_is_backed_off_before_exact_cleanup(files_target, monkeypatch):
    async with peer_case(files_target, monkeypatch) as c:
        now = clock(c)
        await supersede(c)
        original = os.unlink
        with monkeypatch.context() as disk:
            def fail(path, *args, **kwargs):
                if str(path).startswith('blob_') and kwargs.get('dir_fd') is not None:
                    raise OSError('inert disk failure')
                return original(path, *args, **kwargs)
            disk.setattr(os, 'unlink', fail)
            await tick(c)
        retry, = pending(c)
        assert retry['blocked'] == 0 and retry['operation'] == 'discard'
        before = list(c.wire.calls)
        await tick(c)
        assert c.wire.calls == before
        now[0] = retry['next_attempt_at']
        await tick(c)
        assert pending(c) == []
        assert not [e for e in c.service._events('room-one') if e['kind'] == 'message.member']


@pytest.mark.asyncio
@pytest.mark.parametrize('drift', ['cancel', 'generation', 'result', 'recipients', 'owner', 'route', 'expired', 'unknown', 'missing-event'])
async def test_completed_work_never_confers_authority_on_changed_binding(files_target, monkeypatch, drift):
    async with peer_case(files_target, monkeypatch) as c:
        now = clock(c)
        await tick(c)
        assert pending(c) == []
        if drift == 'cancel':
            sql, params = 'UPDATE hosted_room_driver_tasks SET cancel_generation=cancel_generation+1', ()
        elif drift == 'generation':
            sql, params = 'UPDATE hosted_room_driver_tasks SET execution_generation=execution_generation+1', ()
        elif drift == 'result':
            sql, params = 'UPDATE hosted_room_driver_tasks SET result_json=?', (json.dumps(dict(c.stored['result'], peer_result_digest='0'*64)),)
        elif drift == 'recipients':
            sql, params = 'UPDATE hosted_room_driver_tasks SET payload_json=?', (json.dumps(dict(c.task['payload'], recipient_member_ids=['writer'])),)
        elif drift == 'owner':
            sql, params = "UPDATE state_meta SET value='foreign' WHERE key='gateway.hosted.owner.v1:room-one'", ()
        elif drift == 'route':
            sql, params = "UPDATE hosted_room_links SET target_url='https://foreign.invalid'", ()
        elif drift == 'missing-event':
            sql, params = "DELETE FROM hosted_room_events WHERE kind='message.member'", ()
        elif drift == 'unknown':
            sql, params = "UPDATE hosted_room_driver_tasks SET status='indeterminate'", ()
        else:
            now[0] = c.claims['expires_at'] + 1
            sql, params = 'SELECT 1', ()
        c.db._execute_write(lambda conn: conn.execute(sql, params))
        before = list(c.wire.calls)
        try:
            await tick(c)
        except Exception:
            pass  # an explicit refusal is also valid; never a transmission/NEW
        assert c.wire.calls == before
        assert len(c.launched) == len(c.executions) == 1


@pytest.mark.asyncio
async def test_output_retry_does_not_require_new_dispatch_or_input_rights(files_target, monkeypatch):
    from gateway import hosted_room_links
    from gateway.hosted_room_peer import issue_room_grant
    async with peer_case(files_target, monkeypatch) as c:
        clock(c)
        claims = c.claims
        signer = {k: claims[k] for k in ('grant_id', 'room_id', 'home_install_id', 'authority_gateway_id',
            'authority_epoch', 'member_id', 'target_install_id', 'target_profile', 'execution_policy_digest', 'issued_at')}
        grant = issue_room_grant(c.target.adapter._room_grant_secret(), **signer,
            permissions=('status', 'artifact.read', 'artifact.ack'),
            ttl_seconds=claims['expires_at']-claims['issued_at'], status_expires_at=claims['status_expires_at'])
        with _profile_runtime_scope(c.home, hydrate_secrets=False):
            link = hosted_room_links.load_room_link(c.db.db_path, room_id='room-one', member_id='writer')
            hosted_room_links.save_room_link(c.db.db_path, replace(link, grant=grant))
            route = c.service.peer_routes[('room-one','writer')]
            c.service.peer_routes[('room-one','writer')] = replace(route, grant=grant)
        await tick(c)
        assert pending(c) == []
        assert len([e for e in c.service._events('room-one') if e['kind'] == 'message.member']) == 1
        assert len(c.launched) == len(c.executions) == 1


@pytest.mark.asyncio
async def test_authenticated_recovery_does_not_unblock_stale_cancel_generation(files_target, monkeypatch):
    from gateway.session_peer_output_custody import PeerOutputCustody
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
    async with peer_case(files_target, monkeypatch) as c:
        clock(c)
        with monkeypatch.context() as denial:
            def denied(self, *args):
                raise PeerRunsHTTPError('inert denial', status_code=403)
            denial.setattr(PeerOutputCustody, 'read', denied)
            await tick(c)
        c.db._execute_write(lambda conn: conn.execute('UPDATE hosted_room_driver_tasks SET cancel_generation=cancel_generation+1'))
        await renew(c)
        assert pending(c)[0]['blocked'] == 1
        before = list(c.wire.calls)
        await tick(c)
        assert c.wire.calls == before
