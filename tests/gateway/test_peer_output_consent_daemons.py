"""Positive legacy/output consent recovery over actual signed gateway HTTP."""
import asyncio
import copy
import json
import sqlite3
from types import SimpleNamespace

import pytest

from tests.gateway.test_peer_output_faults import pair
from tests.gateway.fixtures.local_recovery_probe import daemon, rpc, websocket
from tests.gateway.test_session_group_peer_daemons import _join_pair
from tui_gateway.hosted_room_driver import HostedRoomBinding
from tui_gateway.hosted_room_peer_transport import PeerMemberRoute, build_member_dispatch
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError
from tui_gateway.hosted_room_peer_output import consent_key, stored_consent, task_output
from gateway.hosted_room_peer_output import OUTPUT_CAPABILITY


@pytest.mark.parametrize('output', [False, True])
@pytest.mark.parametrize('historical_null', [False, True])
def test_lost_consent_requires_exact_signed_positive_legacy_or_output_evidence(tmp_path, output, historical_null):
    with pair(tmp_path, '') as p:
        p.pm.actor = 'plain'
        async def exercise(hd, pd):
            async with websocket(p.home, hd) as hw, websocket(p.peer, pd) as pw:
                room, invite = await _join_pair(hw, pw)
                binding = HostedRoomBinding('linked', room['authority_gateway_id'], room['authority_epoch'])
                route = PeerMemberRoute(home_install_id=binding.gateway_id, member_id='reviewer',
                    target_install_id=invite['catalog']['installation_id'], target_profile='default',
                    capability_digest=invite['catalog']['catalog_digest'], cancellation_scope_id='cancel-consent',
                    trace_id='trace-consent', grant=invite['grant'],
                    execution_policy_digest=invite['catalog']['execution_policy']['policy_digest'])
                client = PeerRunsHTTPClient(base_url=p.url, api_key='', receipt_db_path=p.home / 'state.db',
                                            proof_install_id=route.target_install_id)
                task = dict(identity=SimpleNamespace(task_id='dtask:consent'), execution_generation=0, status='queued',
                    payload={'target_member_id': 'reviewer', 'target_profile': 'default',
                             'source_event_seq': 1, 'prompt': 'Say hello.'})
                if output:
                    assert await asyncio.to_thread(task_output, p.home / 'state.db', binding, task, route, client) == OUTPUT_CAPABILITY
                dispatch = build_member_dispatch(binding=binding, route=route, room_id='linked', task_id='dtask:consent',
                    target_profile='default', execution_generation=1, source_event_seq=1, prompt='Say hello.',
                    trace_id=route.trace_id, document_output=OUTPUT_CAPABILITY if output else None)
                accepted = await asyncio.to_thread(client.dispatch, dispatch=dispatch.as_mapping(), grant=route.grant)
                task.update(status='indeterminate', execution_generation=1)
                async with asyncio.timeout(20):
                    while True:
                        status = await asyncio.to_thread(client._request, '/v1/runs/' + accepted['run_id'], room_grant=route.grant)
                        if status['status'] == 'completed':
                            break
                        await asyncio.sleep(.1)
                assert 'peer_dispatch_evidence' not in status  # unchanged default text wire
                with sqlite3.connect(p.home / 'state.db') as db:
                    db.execute('DELETE FROM state_meta WHERE key=?', (consent_key(dispatch.as_mapping()),))
                historical = None
                if historical_null:
                    historical = {'target': {key: getattr(route, key) for key in ('target_install_id', 'target_profile',
                        'home_install_id', 'capability_digest', 'execution_policy_digest', 'cancellation_scope_id', 'trace_id')},
                        'contract': None, 'dispatched': False}
                    with sqlite3.connect(p.home / 'state.db') as db:
                        db.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
                            (consent_key(dispatch.as_mapping()), json.dumps(historical)))
                assert stored_consent(p.home / 'state.db', dispatch.as_mapping()) == historical
                # Missing evidence stays unknown, with no text-only row or execution POST.
                changed = copy.deepcopy(task)
                changed['payload']['prompt'] = 'Changed instruction'
                with pytest.raises(PeerRunsHTTPError) as refused:
                    await asyncio.to_thread(task_output, p.home / 'state.db', binding, changed, route, client)
                assert refused.value.ambiguous and not refused.value.not_admitted
                assert stored_consent(p.home / 'state.db', dispatch.as_mapping()) == historical
                assert p.proxy.run_posts == 1
                # Corrupted accepted dispatch fails closed at the actual HTTP projection.
                with sqlite3.connect(p.peer / 'state.db') as db:
                    admission_id, encoded = db.execute("SELECT admission_id,payload_json FROM session_admissions WHERE principal_id='api'").fetchone()
                    malformed = json.loads(encoded)
                    malformed['api_turn_v1']['settings']['room_dispatch']['prompt'] += ' corrupt'
                    db.execute('UPDATE session_admissions SET payload_json=? WHERE admission_id=?', (json.dumps(malformed), admission_id))
                with pytest.raises(PeerRunsHTTPError) as corrupt:
                    await asyncio.to_thread(client._request, '/v1/runs/' + accepted['run_id'], room_grant=route.grant)
                assert corrupt.value.status_code == 503
                with sqlite3.connect(p.peer / 'state.db') as db:
                    db.execute('UPDATE session_admissions SET payload_json=? WHERE admission_id=?', (encoded, admission_id))
                if output:
                    # Removing the entire dispatch must not look like a stopped ordinary API Run.
                    with sqlite3.connect(p.peer / 'state.db') as db:
                        absent = json.loads(encoded)
                        absent['api_turn_v1']['settings']['room_dispatch'] = None
                        db.execute("UPDATE session_admissions SET payload_json=?,status='unknown' WHERE admission_id=?",
                                   (json.dumps(absent), admission_id))
                    fresh = PeerRunsHTTPClient(base_url=p.url, api_key='', receipt_db_path=p.home / 'state.db',
                                               proof_install_id=route.target_install_id)
                    await asyncio.to_thread(fresh.recover_dispatch, dispatch=dispatch.as_mapping(), grant=route.grant)
                    with pytest.raises(PeerRunsHTTPError) as unknown:
                        await asyncio.to_thread(fresh.status, room_id='linked', profile='default',
                            session_id=accepted['session_id'], grant=route.grant)
                    assert unknown.value.ambiguous and not unknown.value.not_admitted
                    with pytest.raises(PeerRunsHTTPError):
                        await asyncio.to_thread(fresh.history, room_id='linked', profile='default',
                            session_id=accepted['session_id'], grant=route.grant)
                    with sqlite3.connect(p.peer / 'state.db') as db:
                        db.execute("UPDATE session_admissions SET payload_json=?,status='terminal' WHERE admission_id=?",
                                   (encoded, admission_id))
                recovered = await asyncio.to_thread(task_output, p.home / 'state.db', binding, task, route, client)
                assert recovered == (OUTPUT_CAPABILITY if output else None)
                saved = stored_consent(p.home / 'state.db', dispatch.as_mapping())
                assert saved['dispatched'] is True and saved['dispatch'] == dispatch.as_mapping()
                assert saved['contract'] == recovered and saved['provenance'] == 'canonical-dispatch-v1'
                assert p.proxy.run_posts == 1 and len(p.pm.requests) == 1
                assert (await rpc(hw, 'groups.disband', room_id='linked'))['result']['tombstone']
        with daemon(p.root, p.peer, p.pe, barrier=False) as (_, pd), daemon(p.root, p.home, p.he, barrier=False) as (_, hd):
            asyncio.run(exercise(hd, pd))


def test_home_restart_restores_lost_output_consent_without_resubmitting(tmp_path):
    from tests.gateway.test_peer_output_faults import end
    from tests.gateway.test_session_group_peer_daemons import _events
    import base64
    from tests.gateway.test_peer_output_daemons import CONTENT
    with pair(tmp_path, '') as p:
        with daemon(p.root, p.peer, p.pe, barrier=False) as (_, pd):
            async def accepted_then_detach(hd, home_process):
                async with websocket(p.home, hd) as hw, websocket(p.peer, pd) as pw:
                    _, invite = await _join_pair(hw, pw, peer_first=True)
                    assert (await rpc(hw, 'groups.peer.register', room_id='linked', member_id='reviewer',
                        target_url=p.url, target_profile='default', grant=invite['grant'], catalog=invite['catalog']))['result']['registered']
                    assert (await rpc(hw, 'groups.send', room_id='linked', event_id='lost-consent',
                        payload={'text': '@reviewer Create and share a checklist.', 'thread_id': 'thread'}))['result']['accepted']
                    assert await asyncio.to_thread(p.pm.shared_event.wait, 20)
                    assert p.pm.shared['ok']
                    home_process.kill(); home_process.wait(timeout=10)
            with daemon(p.root, p.home, p.he, barrier=False) as (hp, hd):
                asyncio.run(accepted_then_detach(hd, hp))
            with sqlite3.connect(p.home / 'state.db') as db:
                (saved_raw,) = db.execute("SELECT value FROM state_meta WHERE key LIKE 'group.peer-output.v1.%'").fetchone()
                saved = json.loads(saved_raw)
                db.execute("DELETE FROM state_meta WHERE key LIKE 'group.peer-output.v1.%'")
                db.execute('DELETE FROM hosted_room_remote_runs')
            p.pm.release.set()
            async def reopened(hd):
                async with websocket(p.home, hd) as hw:
                    replies = await _events(hw, 'message.member', count=2, timeout=110)
                    reply = next(e for e in replies if e['actor']['id'] == 'reviewer')
                    attachment, = reply['payload']['attachments']
                    download = await rpc(hw, 'groups.attachment.download', room_id='linked',
                        event_id=reply['event_id'], attachment_id=attachment['attachment_id'])
                    assert base64.b64decode(download['result']['data_base64']) == CONTENT.encode()
                    restored = stored_consent(p.home / 'state.db', saved['dispatch'])
                    assert restored['contract'] == saved['contract'] and restored['dispatch'] == saved['dispatch']
                    assert restored['dispatched'] is True and p.proxy.run_posts == 1
                    await end(p, hw)
                    with sqlite3.connect(p.peer / 'state.db') as db:
                        assert db.execute("SELECT COUNT(*) FROM session_admissions WHERE principal_id='api'").fetchone()[0] == 1
                        assert db.execute('SELECT COUNT(*) FROM hosted_room_output_artifacts WHERE acknowledged_at IS NULL OR blob_reclaimed_at IS NULL').fetchone()[0] == 0
            with daemon(p.root, p.home, p.he, barrier=False) as (_, hd):
                asyncio.run(reopened(hd))
