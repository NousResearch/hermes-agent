"""worker.register requires the worker-adoption capability a launched worker's ticket carries
(B1-14): an interactive client (session:control) must not park a 'registered' compute row that
nothing reaps, pausing the session's FIFO until a restart turns it unknown. Real SQLite, real
processes."""
import subprocess
import sys
from types import SimpleNamespace

import psutil
import pytest

from hermes_cli.process_identity import worker_birth
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_seed import seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


@pytest.mark.asyncio
async def test_worker_register_requires_the_owners_launched_worker(tmp_path, monkeypatch):
    from gateway.config import Platform
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import SessionActor, SessionRef
    from gateway.session_worker import worker_request
    from persistence.core.errors import AuthorityRefusal

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    db = storage_handle(tmp_path)
    seed_session(db, 'owned', 'telegram')
    runner = SimpleNamespace(_draining=False, config=SimpleNamespace(multiplex_profiles=False),
                             _adapter_for_source=lambda source: None)
    authority = SessionAuthority(runner, profile_id='p', instance_id='owner',
                                 epoch=db.runtime.begin_epoch(instance_id='owner'), persistence=db)
    authority.sessions['owned'] = LiveSession(SimpleNamespace(platform=Platform.TELEGRAM, user_id='human'), 'route')
    ref = SessionRef('p', 'owned')
    child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'], stdin=subprocess.PIPE,  # noqa: ASYNC220 -- Popen returns immediately
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def scope(pid):
        return {'profile_id': 'p', 'session_id': 'owned', 'execution_id': f'compute-{pid}', 'generation': 0,
                'pid': pid, 'birth': worker_birth(psutil.Process(pid)), 'secret': 'w' * 32, 'kind': 'compute'}
    interactive = SimpleNamespace(authority=authority, actor=SessionActor(
        'human', 'p', frozenset({'session:create', 'session:read', 'session:submit', 'session:control'}), 't'))
    worker = SimpleNamespace(authority=authority, actor=SessionActor('human', 'p', frozenset({'worker:adopt'}), 't'))
    try:
        # A default interactive ticket, even naming a live process of the same user.
        with pytest.raises(AuthorityRefusal, match='permission_denied'):
            await worker_request(interactive, ref, scope(child.pid), operation='register')
        assert raw_holder(db)._read_one('SELECT COUNT(*) FROM worker_executions')[0] == 0
        registered = await worker_request(worker, ref, scope(child.pid), operation='register')
        assert registered['status'] == 'registered'
    finally:
        child.stdin.close()
        child.wait(timeout=5)
        db.close()
