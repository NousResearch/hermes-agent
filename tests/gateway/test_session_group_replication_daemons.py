"""Two real canonical gateways: a participant keeps a copy and its task evidence, then retires it after Disband."""
import asyncio
from pathlib import Path
import socket
import sqlite3

from tests.gateway.fixtures.local_recovery_probe import daemon, rpc, websocket
from tests.gateway.test_session_group_peer_daemons import _events, _gateway, _last_user_text, _model, _send


async def _until(read, accept, timeout=90):
    value = None
    try:
        async with asyncio.timeout(timeout):
            while not accept(value := await read()):
                await asyncio.sleep(.25)
    except TimeoutError as exc:
        raise AssertionError(f'copy did not reach expected state: {value!r}') from exc
    return value


def _home_obligations(home):
    with sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True) as db:
        return db.execute('SELECT state FROM hosted_room_replica_retirement_home').fetchall()


def test_a_participant_copy_keeps_history_and_evidence_across_a_home_restart_then_retires(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home_model = _model('HOME_REPLY')
    target_model = _model('PEER_REPLY', 'HOLD_ACCEPTED')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        api_port = sock.getsockname()[1]
    home, home_env = _gateway(tmp_path, 'home', home_model, root)
    target, target_env = _gateway(tmp_path, 'target', target_model, root, api_port=api_port)
    seen = {}

    async def copy_of(target_ws):
        reply = await rpc(target_ws, 'groups.replica_state', room_id='linked')
        return reply.get('result') or {'error': reply.get('error')}

    async def latest_seq(home_ws):
        return (await rpc(home_ws, 'groups.state', room_id='linked'))['result']['room']['latest_seq']

    async def synced(home_ws, target_ws):
        """The copy and the home's current head, read together so a growing room is compared fairly."""
        latest = await latest_seq(home_ws)
        copy = await copy_of(target_ws)
        return {**copy, '_caught_up': copy.get('last_seq') == latest, '_home_latest_seq': latest,
                '_home_replication': (await rpc(home_ws, 'groups.state', room_id='linked'))['result']['driver_status'].get('replication')}

    async def before_restart(home_desc, target_desc, home_proc):
        async with websocket(home, home_desc) as home_ws, websocket(target, target_desc) as target_ws:
            async with asyncio.timeout(30):
                while not (link := (await rpc(target_ws, 'groups.capabilities'))['result']['room_link'])['enabled']:
                    await asyncio.sleep(.1)
            catalog, url = link['catalog'], link['endpoint']['url']
            methods = (await rpc(home_ws, 'groups.capabilities'))['result']['methods']
            assert {'groups.replica_state', 'groups.replication.prepare', 'groups.replication.enroll',
                    'groups.replication.revoke'} <= set(methods)
            pinned = {'kind': 'peer', 'peer_id': 'target-gateway', 'installation_id': catalog['installation_id'],
                      'profile': 'default', 'capability_digest': catalog['catalog_digest']}
            room = (await rpc(home_ws, 'groups.create', room_id='linked', name='Linked', members=[
                {'member_id': 'host', 'profile': 'default', 'handle': 'host'},
                {'member_id': 'reviewer', 'profile': 'default', 'handle': 'reviewer', 'target': pinned}]))['result']['room']
            # The participant's operator opts in to the copy and its task evidence on the member grant.
            invited = (await rpc(target_ws, 'groups.peer.invite', room_id='linked', member_id='reviewer',
                                 home_install_id=room['authority_gateway_id'],
                                 authority_gateway_id=room['authority_gateway_id'],
                                 authority_epoch=room['authority_epoch'], replication=True, work_records=True))['result']
            registered = await rpc(home_ws, 'groups.peer.register', room_id='linked', member_id='reviewer',
                                   target_url=url, target_profile='default', grant=invited['grant'], catalog=catalog)
            assert registered['result']['registered'], registered
            # Retirement takes both owners: the room owner prepares it, the participant's operator enrolls it.
            enrollment = (await rpc(home_ws, 'groups.replication.prepare', room_id='linked',
                                    target_install_id=catalog['installation_id'], endpoint=url))['result']['enrollment']
            enrolled = (await rpc(target_ws, 'groups.replication.enroll', enrollment=enrollment))['result']
            assert enrolled == {**enrollment, 'state': 'active'}
            seen['enrollment'] = enrollment
            seen['grant'] = invited['grant']
            seen['target'] = catalog['installation_id']

            await _send(home_ws, 'ask', '@reviewer FIRST')
            assert (await _events(home_ws, 'message.member'))[0]['payload']['text'] == 'PEER_REPLY'
            # Let the first discussion settle before the next message: a message that lands between
            # a turn's settlement and its room.activity is dropped by the base room policy.
            await _events(home_ws, 'room.activity')
            held, _ = target_model.gates['HOLD_ACCEPTED']
            await _send(home_ws, 'held', '@reviewer HOLD_ACCEPTED')
            assert await asyncio.to_thread(held.wait, 60)
            # While the participant runs the accepted turn, its copy holds the history so far and the
            # home's evidence that this turn was accepted, with the exact run receipt.
            copy = await _until(lambda: synced(home_ws, target_ws),
                                lambda c: c.get('_caught_up') and c.get('work_records', {}).get('phases') == {'running': 1})
            evidence = copy['work_records']
            assert copy['safety_status'] == 'passive' and evidence['source_loss_safe'] is False
            assert [r['member_id'] for r in evidence['receipts']] == ['reviewer'], evidence
            assert 'HOLD_ACCEPTED' not in str(evidence)
            # The participant shows the room only as a read-only copy, never as a room of its own.
            listed = (await rpc(target_ws, 'groups.list'))['result']['rooms']
            assert [(room['room_id'], room['copy'], room['revision']) for room in listed] == [('linked', True, 0)]
            home_proc.kill()
            await asyncio.to_thread(home_proc.wait, 10)

    async def after_restart(home_desc, target_desc, home_proc, target_proc):
        async with websocket(home, home_desc) as home_ws, websocket(target, target_desc) as target_ws:
            # The restarted home finishes the accepted turn once and its publisher resumes from the stored route.
            replies = await _events(home_ws, 'message.member', count=2)
            assert [r['payload']['text'] for r in replies] == ['PEER_REPLY', 'PEER_REPLY'], replies
            assert sum('HOLD_ACCEPTED' in _last_user_text(r) for r in target_model.requests) == 1
            copy = await _until(lambda: synced(home_ws, target_ws),
                                lambda c: c.get('_caught_up') and c['work_records'].get('phases', {}).get('running') is None)
            assert copy['work_records']['availability'] == 'available'
            assert copy['name'] == 'Linked'
            latest = copy['last_seq']

            async def prefixes():
                home_custody = (await rpc(home_ws, 'groups.custody.status', room_id='linked'))['result']
                copy_custody = (await rpc(target_ws, 'groups.custody.status', room_id='linked'))['result']
                held = {c['install_id']: c['watermark'] for c in home_custody['custodians']}.get(seen['target'])
                return held, copy_custody['watermark']

            # The participant holds the exact full prefix: the home verified the very watermark the
            # copy computes over its own stored history.
            held, own = await _until(prefixes, lambda pair: pair is not None and pair[0] is not None
                                     and pair[0] == pair[1] and pair[0]['seq'] >= latest)
            assert own['seq'] >= latest and len(own['event_hash']) == 64

            seen['last_seq'] = latest
            # The retirement authorization is independent of this now-revoked member grant.
            assert (await rpc(target_ws, 'groups.peer.revoke', grant=seen['grant']))['result']['revoked']
            target_proc.kill()
            await asyncio.to_thread(target_proc.wait, 10)
            disbanded = await rpc(home_ws, 'groups.disband', room_id='linked')
            assert 'tombstone' in disbanded['result'], disbanded
            with sqlite3.connect(home / 'state.db') as db:
                assert db.execute('SELECT COUNT(*) FROM hosted_room_links').fetchone()[0] == 0
            assert _home_obligations(home)[0][0] in {'closed', 'ready'}
            home_proc.kill()
            await asyncio.to_thread(home_proc.wait, 10)

    async def retired_after_both_restarts(home_desc, target_desc):
        async with websocket(home, home_desc) as home_ws, websocket(target, target_desc) as target_ws:
            retired = await _until(lambda: copy_of(target_ws), lambda c: c.get('safety_status') == 'retired')
            assert retired['disbanded_at'] is None and retired['copy_retired_at'] > 0
            assert retired['last_seq'] == seen['last_seq']  # no invented Disband event
            await _until(lambda: asyncio.to_thread(_home_obligations, home), lambda rows: rows == [('acknowledged',)])
            again = await rpc(target_ws, 'groups.replication.enroll', enrollment=seen['enrollment'])
            assert again['error']['message'] == 'replica_retirement_conflict', again

    try:
        with daemon(root, target, target_env, barrier=False) as (target_proc, target_desc):
            with daemon(root, home, home_env, barrier=False) as (home_proc, home_desc):
                asyncio.run(before_restart(home_desc, target_desc, home_proc))
            target_model.gates['HOLD_ACCEPTED'][1].set()
            with daemon(root, home, home_env, barrier=False) as (home_proc, home_desc):
                asyncio.run(after_restart(home_desc, target_desc, home_proc, target_proc))
        with daemon(root, target, target_env, barrier=False) as (_, target_desc):
            with daemon(root, home, home_env, barrier=False) as (_, home_desc):
                asyncio.run(retired_after_both_restarts(home_desc, target_desc))
    finally:
        for model in (home_model, target_model):
            for _, release in model.gates.values():
                release.set()
            model.shutdown()
            model.server_close()
