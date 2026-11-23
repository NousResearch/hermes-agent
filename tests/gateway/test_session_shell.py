"""Ink ``!cmd`` runs for its own local session, in that session's directory (dokterdok N22)."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tests.gateway.fixtures.local_recovery_probe import child_env
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


@pytest.mark.platforms("posix")
def test_shell_exec_runs_in_the_session_cwd_behind_the_safety_gates(tmp_path):
    """The owner serves ``shell.exec`` with the ``session_id`` Ink sends: the command runs in the
    session's frozen launch cwd (not the daemon's), its exit code and output come back, the
    owner's credentials are scrubbed from the child, and a dangerous or sessionless command is
    refused."""
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir()
    user.mkdir()
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
               PYTHONPATH=str(root), PYTHONUNBUFFERED='1')
    result = subprocess.run([sys.executable, str(root / 'tests/gateway/fixtures/shell_exec_peer.py')],
                            cwd=root, env=env, capture_output=True, text=True, timeout=90, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((home / 'receipt.json').read_text())
    assert receipt['owner_cwd'] != receipt['session_cwd']
    ran = receipt['ran']
    assert isinstance(ran, dict) and ran.get('code') == 3, receipt
    assert Path(ran['stdout'].strip()).resolve() == Path(receipt['session_cwd']).resolve(), receipt
    assert ran['stderr'] == 'key=', receipt
    assert receipt['dangerous'].get('message') == 'dangerous_command', receipt
    assert receipt['sessionless'] == 4001, receipt


@pytest.mark.asyncio
@pytest.mark.parametrize('backend', ['docker', 'ssh'])
async def test_shell_exec_refusal_names_a_sandboxed_terminal_backend(tmp_path, monkeypatch, backend):
    """``shell.exec`` runs on the gateway host, so a session whose frozen terminal policy is a
    sandbox is refused (#134889 TUI-16) and the refusal names that backend for the client. A local
    session still runs. Monkeypatched: the session's policy lookup and local-source check."""
    from types import SimpleNamespace

    from gateway import session_local, session_policy
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_controls import AuthorityConnection
    from persistence_assembly import open_persistence, profile_scope
    from tests.fixtures.storage_seed import seed_session

    work = tmp_path / 'work'
    work.mkdir()
    terminal = {'TERMINAL_ENV': backend}
    monkeypatch.setattr(session_local, 'authorize_local_source', lambda runner, source: True)
    monkeypatch.setattr(session_policy, 'policy_for_source', lambda runner, source: SimpleNamespace(
        cwd=str(work), terminal_json=json.dumps(terminal)))
    with open_persistence(profile_scope(home=tmp_path)) as db:
        seed_session(db, 's', 'test')
        epoch = db.runtime.begin_epoch(instance_id='current')
        authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id=str(tmp_path / 'home'),
                                     instance_id='current', persistence=storage_handle(db.scope.profile_dir), epoch=epoch)
        authority.sessions['s'] = LiveSession(None, 'route')
        connection = AuthorityConnection(authority, object(), {'user_id': 'owner'})

        async def call():
            return await connection.dispatch({'id': 1, 'method': 'shell.exec',
                                              'params': {'session_id': 's', 'command': 'echo ran'}})

        refused = await call()
        assert refused['error']['data'] == {'reason': 'unsupported_terminal_backend',
                                            'field': 'terminal.backend', 'detail': backend}, refused
        terminal['TERMINAL_ENV'] = 'local'
        ran = await call()
        assert ran.get('result', {}).get('stdout', '').strip() == 'ran', ran


def _local_authority(tmp_path, monkeypatch, db, runner):
    """A SessionAuthority with one local session ``s``. Monkeypatched: policy lookup, local-source check."""
    from types import SimpleNamespace

    from gateway import session_local, session_policy
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_controls import AuthorityConnection
    from tests.fixtures.storage_seed import seed_session

    work = tmp_path / 'work'
    work.mkdir()
    monkeypatch.setattr(session_local, 'authorize_local_source', lambda runner, source: True)
    monkeypatch.setattr(session_policy, 'policy_for_source', lambda runner, source: SimpleNamespace(
        cwd=str(work), terminal_json=json.dumps({'TERMINAL_ENV': 'local'})))
    seed_session(db, 's', 'test')
    epoch = db.runtime.begin_epoch(instance_id='current')
    authority = SessionAuthority(runner, profile_id=str(tmp_path / 'home'), instance_id='current',
                                 persistence=storage_handle(db.scope.profile_dir), epoch=epoch)
    authority.sessions['s'] = LiveSession(None, 'route')
    return AuthorityConnection(authority, object(), {'user_id': 'owner'})


_EXEC = {'id': 1, 'method': 'shell.exec', 'params': {'session_id': 's', 'command': 'echo ran'}}


@pytest.mark.asyncio
async def test_shell_exec_scan_does_not_stall_the_event_loop(tmp_path, monkeypatch):
    """The dangerous-command scan can take seconds (#140091). It runs off the loop, so the loop
    keeps serving the owner's other sessions meanwhile. Monkeypatched: the scan, which waits for
    a task only a free loop can run."""
    import asyncio
    import threading
    from types import SimpleNamespace

    from persistence_assembly import open_persistence, profile_scope
    from tools import approval_detection

    loop_ran, during_scan = threading.Event(), []

    def scan(command):
        during_scan.append(loop_ran.wait(timeout=2))
        return (False, None)

    async def other_session_work():
        loop_ran.set()

    monkeypatch.setattr(approval_detection, 'detect_hardline_command', scan)
    with open_persistence(profile_scope(home=tmp_path)) as db:
        connection = _local_authority(tmp_path, monkeypatch, db, SimpleNamespace(_draining=False))
        task = asyncio.create_task(other_session_work())
        ran = await connection.dispatch(dict(_EXEC))
        await task
        assert ran.get('result', {}).get('stdout', '').strip() == 'ran', ran
        assert during_scan == [True]


@pytest.mark.asyncio
async def test_shell_exec_rechecks_admission_after_the_scan(tmp_path, monkeypatch):
    """Admission can close while the scan runs off the loop: the command is then refused, not run.
    Monkeypatched: the scan (it starts the drain) and the runner."""
    from types import SimpleNamespace

    from gateway import session_shell
    from persistence_assembly import open_persistence, profile_scope
    from tools import approval_detection

    runner = SimpleNamespace(_draining=False)

    def scan_while_drain_starts(command):
        runner._draining = True
        return (False, None)

    ran = []
    monkeypatch.setattr(approval_detection, 'detect_hardline_command', scan_while_drain_starts)
    monkeypatch.setattr(session_shell, '_run', lambda command, cwd: ran.append(command) or {})
    with open_persistence(profile_scope(home=tmp_path)) as db:
        connection = _local_authority(tmp_path, monkeypatch, db, runner)
        refused = await connection.dispatch(dict(_EXEC))
        assert refused['error']['data'].get('reason') == 'runtime_draining', refused
        assert ran == []
