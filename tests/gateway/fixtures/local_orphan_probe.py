"""Credential-free creator for the bounded #142592 crash/restart investigation."""
import asyncio
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch


def _pending_work(authority, sid, scenario):
    from hermes_state_local import local_receipt
    from hermes_state_runtime import admit_session_input, claim_session_input, register_worker_execution
    if scenario in ('queued', 'started'):
        admit_session_input(authority.db, epoch=authority.epoch,
                            principal_id=local_receipt(authority.db, sid)['principal_id'],
                            session_id=sid, request_id='pending-fixture', payload={'text': 'never execute'})
        if scenario == 'started':
            claim_session_input(authority.db, epoch=authority.epoch, session_id=sid)
    elif scenario == 'worker':
        register_worker_execution(authority.db, epoch=authority.epoch, execution_id='fixture-worker',
                                  session_id=sid, generation=authority.db.get_session(sid)['runtime_generation'],
                                  kind='compute', adoption_secret='fixture-only')


async def create(home, scenario='idle'):
    attempts = []

    def audit(event, args):
        if event in {'socket.connect', 'socket.sendto', 'socket.getaddrinfo'}:
            attempts.append(event)
            raise RuntimeError('Network forbidden in isolated local-session probe')

    sys.addaudithook(audit)
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session

    with patch.object(run, '_load_gateway_config', return_value={'platform_toolsets': {'cli': []}}):
        store = SessionStore(home / 'sessions', GatewayConfig())
        runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={},
                                 _draining=False, _cached_agent_for=lambda route: None,
                                 _adapter_for_source=lambda source: None)
        authority = await initialize_session_authority(
            runner, profile_id=str(home), instance_id=f'creator-{os.getpid()}')
        connection = AuthorityConnection(authority, object(), {'user_id': 'fixture-owner'})
        records = {}
        for name, surface, refresh in (
            ('tui_local', 'tui', True), ('desktop_local', 'gui', True),
            ('unrefreshed_tui', 'tui', False), ('tool_control', 'tool', True),
            ('cli_control', 'cli', True), ('acp_control', 'acp', True),
        ):
            ref = create_local_session(authority, connection.actor, {
                'request_id': name, 'source': surface, 'cwd': str(home),
                'model': 'm', 'toolsets': [],
            })
            authority.db.append_message(ref.session_id, 'user', f'fixture history for {name}')
            before = authority.db.get_session(ref.session_id)['source']
            if refresh:
                # The ordinary public metadata update invokes the production peer refresh.
                store.update_session(authority.sessions[ref.session_id].route, touch_activity=False)
            after = authority.db.get_session(ref.session_id)['source']
            records[name] = {'id': ref.session_id, 'before': before, 'after': after}
        # Age only fixture timestamps; no six-hour sleep or fabricated session/receipt rows.
        old = time.time() - 100 * 86400
        for row in records.values():
            authority.db._write_sql('UPDATE sessions SET started_at=? WHERE id=?', (old, row['id']))
            authority.db._write_sql('UPDATE messages SET timestamp=? WHERE session_id=?', (old, row['id']))
        _pending_work(authority, records['tui_local']['id'], scenario)
        result = {'creator_pid': os.getpid(), 'records': records, 'network_attempts': attempts}
        (home / 'creator-evidence.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        # Deliberately skip every close, finalizer and atexit handler of this private child.
        os._exit(23)


if __name__ == '__main__':
    asyncio.run(create(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else 'idle'))
