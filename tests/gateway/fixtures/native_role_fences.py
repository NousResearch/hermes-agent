"""Owned await-race and fresh-interpreter retention controls."""
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace


async def fences(runner, authority, adapter, guild, event, peer, mode):
    from gateway.config import Platform
    from gateway.session_ingress_context import native_callback
    from hermes_state_runtime import RuntimeStoreError, cancel_session_input, list_session_admissions

    state = Path(os.environ['HERMES_HOME'])

    def rows(sid):
        return list_session_admissions(authority.db, session_id=sid, pending_only=False)

    async def accept(ev):
        with native_callback(runner, ev, state):
            receipt = await authority.admit_native(ev)
        # Own the committed-before-execution scheduling boundary; no claim mocked.
        task = authority.sessions[receipt.ref.session_id].task
        task.cancel()
        return receipt

    if mode == 'fences':
        ev = event('frozen-request')
        guild.hook = lambda: setattr(ev, 'message_id', 'changed-during-fetch')
        receipt = await accept(ev)
        guild.hook = None
        failures = []
        if rows(receipt.ref.session_id)[0]['request_id'] != 'frozen-request':
            failures.append('request identity changed during authorization')
        second = await accept(event('second'))
        # The checked oldest row can be canceled during remote I/O. Its successor
        # must receive its own current membership check, not inherit the old verdict.
        def cancel_first():
            guild.hook = None
            cancel_session_input(authority.db, epoch=authority.epoch, admission_id=receipt.admission_id)
            guild.roles = []
        guild.hook = cancel_first
        await authority._drain(receipt.ref)
        if rows(second.ref.session_id)[1]['status'] != 'queued':
            failures.append('unchecked successor consumed after cancel during preflight')
        if peer.requests:
            failures.append('unchecked successor reached the model')
        guild.roles = [SimpleNamespace(id=700)]
        # A same-credential replacement is still a different live connector.
        other = type(adapter)(adapter.config)
        other.gateway_runner = runner
        other._client = adapter._client
        guild.hook = lambda: runner.adapters.__setitem__(Platform.DISCORD, other)
        try:
            await accept(event('wrong-connector'))
        except RuntimeStoreError as exc:
            assert exc.reason == 'not_found', exc.reason
        else:
            raise AssertionError('connector replaced during fetch was accepted')
        finally:
            guild.hook = None
            runner.adapters[Platform.DISCORD] = adapter
        before_drain = rows(second.ref.session_id)
        guild.hook = lambda: setattr(runner, '_draining', True)
        try:
            try:
                await accept(event('shutdown-during-auth'))
            except RuntimeStoreError as exc:
                assert exc.reason == 'runtime_draining', exc.reason
            else:
                failures.append('new input accepted after shutdown began during lookup')
            runner._draining = False
            await authority._drain(second.ref)
            if rows(second.ref.session_id) != before_drain:
                failures.append('shutdown consumed queued input after authorization await')
        finally:
            guild.hook = None
            runner._draining = False
        assert not failures, failures
        assert len(rows(second.ref.session_id)) == 2
        print(json.dumps({'mode': mode, 'rows': rows(second.ref.session_id), 'model_calls': 0}), flush=True)
        return
    if mode == 'capture':
        receipt = await accept(event('restart-role'))
        (state / 'role-restart.json').write_text(json.dumps({'sid': receipt.ref.session_id}))
        assert rows(receipt.ref.session_id)[0]['status'] == 'queued'
        os._exit(0)
    sid = json.loads((state / 'role-restart.json').read_text())['sid']
    source = event('binding').source
    before = rows(sid)
    if mode == 'recover':
        rejected = {}
        for case in ('revoked', 'wrong-credential'):
            token = adapter.config.token
            if case == 'revoked':
                guild.roles = []
            else:
                adapter.config.token = 'different-connector'
            result = await authority.recover_native_sessions([(sid, source, adapter)])
            assert result[sid] in {'permission_denied', 'profile_mismatch'}, result
            assert rows(sid) == before
            rejected[case] = result[sid]
            adapter.config.token = token
            guild.roles = [SimpleNamespace(id=700)]
    result = await authority.recover_native_sessions([(sid, source, adapter)])
    assert result[sid] == 'ready', result
    await authority.sessions[sid].task
    assert all(row['outcome'] == 'completed' for row in rows(sid)), rows(sid)
    assert len(peer.requests) == (1 if mode == 'recover' else 0)
    print(json.dumps({'mode': mode, 'rows': rows(sid), 'model_calls': len(peer.requests)}), flush=True)
