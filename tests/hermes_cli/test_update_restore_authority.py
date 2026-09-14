"""Maintenance must not replace the database behind a live authority."""
import asyncio
from contextlib import closing
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import sqlite3
import threading

from tests.gateway.fixtures.local_recovery_probe import Model, daemon, rpc, websocket


def test_restore_refuses_live_authority_without_changing_data(tmp_path):
    from hermes_cli.backup import create_quick_snapshot
    from hermes_cli.update_cmd_maint import _restore_state_db_from_snapshot

    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests = []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False},
        'model': {'provider': 'custom', 'default': 'fixture', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False}},
        'platform_toolsets': {'cli': []},
    }))
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url)

    def rows():
        with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
            return list(db.execute('SELECT request_id,status FROM session_admissions ORDER BY rowid'))

    async def exercise(desc):
        async with websocket(home, desc) as ws:
            created = await rpc(ws, 'session.create', request_id='create', source='tui', cwd=str(home), toolsets=[])
            session = created['result']['session_id']
            await rpc(ws, 'prompt.submit', session_id=session, input_id='warm', text='WARM')
            async with asyncio.timeout(30):
                while dict(rows()).get('warm') != 'terminal':
                    await asyncio.sleep(.05)
            snapshot = create_quick_snapshot(hermes_home=home)
            assert snapshot, snapshot
            await rpc(ws, 'prompt.submit', session_id=session, input_id='started', text='BLOCK_STARTED')
            assert await asyncio.to_thread(peer.blocked.wait, 25)
            await rpc(ws, 'prompt.submit', session_id=session, input_id='queued', text='FOLLOWER')
            before = rows()
            assert dict(before) == {'warm': 'terminal', 'started': 'started', 'queued': 'queued'}
            import hashlib
            bundle = [Path(str(home / 'state.db') + suffix) for suffix in ('', '-wal', '-shm')]
            hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle if p.exists()}
            restored = _restore_state_db_from_snapshot(home / 'state.db', home / 'state-snapshots' / snapshot / 'state.db')
            assert hashes == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle if p.exists()}
            after = rows()
            print(json.dumps({'live_restore': restored, 'before': before, 'after': after}), flush=True)
            assert not restored and after == before
            return snapshot

    try:
        with daemon(root, home, env, barrier=False) as (proc, desc):
            snapshot = asyncio.run(exercise(desc))
            old_epoch = desc['authority_epoch']
        peer.release.set()
        assert _restore_state_db_from_snapshot(home / 'state.db', home / 'state-snapshots' / snapshot / 'state.db')
        assert rows() == [('warm', 'terminal')]
        with daemon(root, home, env, barrier=False) as (_, restarted):
            assert restarted['instance_id'] != desc['instance_id']
            assert restarted['authority_epoch'] > old_epoch
            assert rows() == [('warm', 'terminal')]
            later_epoch = restarted['authority_epoch']
        # Restoring the same older snapshot must not reuse a previous owner's epoch.
        assert _restore_state_db_from_snapshot(home / 'state.db', home / 'state-snapshots' / snapshot / 'state.db')
        with daemon(root, home, env, barrier=False) as (_, restored_again):
            assert restored_again['authority_epoch'] > later_epoch
            print(json.dumps({'offline_restore_restart_epochs': [old_epoch, later_epoch, restored_again['authority_epoch']],
                              'restart_instance_changed': restored_again['instance_id'] != restarted['instance_id']}), flush=True)
    finally:
        peer.release.set()
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)


def test_corrupt_epoch_refuses_profile_sweep_with_separate_recovery(tmp_path, monkeypatch, capsys):
    from hermes_cli import backup, update_cmd
    from hermes_cli.session_recovery import recover_session_database
    from hermes_state import SessionDB

    homes = [tmp_path / 'default', tmp_path / 'sibling']
    originals = {}
    snapshots = {}
    for home in homes:
        home.mkdir()
        with SessionDB(db_path=home / 'state.db') as db:
            db.create_session('retained', 'cli')
            db.append_message('retained', 'user', 'snapshot transcript')
        snapshot = backup.create_quick_snapshot(hermes_home=home)
        snapshots[home] = home / 'state-snapshots' / snapshot / 'state.db'
        (home / 'state.db').write_bytes(b'unreadable canonical runtime epoch')
        originals[home] = (home / 'state.db').read_bytes(), snapshots[home].read_bytes()
    monkeypatch.setattr(update_cmd, 'get_hermes_home', lambda: homes[0])
    monkeypatch.setattr(backup, '_sibling_profile_homes', lambda home: [('work', homes[1])])
    update_cmd._verify_and_restore_state_dbs_post_update()
    output = capsys.readouterr().out
    for home in homes:
        assert ((home / 'state.db').read_bytes(), snapshots[home].read_bytes()) == originals[home]
    assert 'Auto-restored' not in output
    assert 'separate' in output and 'hermes sessions recover' in output
    assert 'restored copy also failed integrity' not in output
    for home in homes:
        report = recover_session_database(snapshots[home], home / 'recovered.db')
        assert report['verified'] and report['source_unchanged']
        with closing(sqlite3.connect(home / 'recovered.db')) as db:
            assert db.execute('SELECT content FROM messages').fetchall() == [('snapshot transcript',)]
        assert (home / 'state.db').read_bytes() == originals[home][0]


def test_update_restore_excludes_startup_until_publication(tmp_path, monkeypatch):
    import subprocess
    import sys
    from gateway.runtime_ownership import ProfileOwnership
    from hermes_cli import backup
    from hermes_cli.update_cmd_maint import _restore_state_db_from_snapshot

    home = tmp_path / 'state'
    home.mkdir()
    source, destination = tmp_path / 'snapshot.db', home / 'state.db'
    for path, value in ((source, 'snapshot'), (destination, 'current')):
        with closing(sqlite3.connect(path)) as db:
            db.execute('CREATE TABLE marker(value TEXT)')
            db.execute('INSERT INTO marker VALUES(?)', (value,))
            db.commit()
    before = destination.read_bytes(), source.read_bytes()
    owner = ProfileOwnership()
    owner.reserve([home])
    try:
        assert not (home / 'gateway.pid').exists()
        assert not _restore_state_db_from_snapshot(destination, source)
        assert (destination.read_bytes(), source.read_bytes()) == before
    finally:
        owner.close()

    # Pause only the actual database publication, with maintenance already held.
    arrived, release = threading.Event(), threading.Event()
    publish = backup._restore_db_pages
    def barrier(*args):
        arrived.set()
        assert release.wait(30)
        return publish(*args)
    monkeypatch.setattr(backup, '_restore_db_pages', barrier)
    results = []
    thread = threading.Thread(target=lambda: results.append(_restore_state_db_from_snapshot(destination, source)))
    thread.start()
    try:
        assert arrived.wait(20)
        root = Path(__file__).resolve().parents[2]
        env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if k in os.environ}
        env.update(HOME=str(tmp_path / 'user'), HERMES_HOME=str(home), PYTHONPATH=str(root))
        contender = subprocess.run([sys.executable, '-c',
            "import logging, runpy; logging.basicConfig(level=logging.INFO); runpy.run_module('gateway.run', run_name='__main__')"],
            cwd=root, env=env, stdin=subprocess.DEVNULL, capture_output=True,
            text=True, encoding='utf-8', timeout=25)
        assert 'already owns profile' in contender.stdout + contender.stderr, contender
        assert (destination.read_bytes(), source.read_bytes()) == before
    finally:
        release.set()
        thread.join(timeout=30)
    assert not thread.is_alive() and results == [True]
    with closing(sqlite3.connect(destination)) as db:
        assert db.execute('SELECT value FROM marker').fetchall() == [('snapshot',)]
    assert source.read_bytes() == before[1]
    print(json.dumps({'startup_first_refused': True, 'maintenance_first_refused': True,
                      'snapshot_unchanged': True}), flush=True)
