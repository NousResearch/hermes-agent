"""Maintenance must not replace the database behind a live authority."""
import asyncio
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import sqlite3
import threading

from tests.gateway.fixtures.local_recovery_probe import Model, daemon, rpc, websocket


def test_restore_refuses_live_authority_without_changing_data(tmp_path):
    from hermes_cli.backup import create_quick_snapshot, restore_quick_snapshot

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
        with sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True) as db:
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
            restored = restore_quick_snapshot(snapshot, hermes_home=home)
            assert hashes == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle if p.exists()}
            after = rows()
            print(json.dumps({'live_restore': restored, 'before': before, 'after': after}), flush=True)
            assert not restored and after == before
            # The real daemon still holds the source DB. Separate-output salvage
            # must not require stopping it or installing recovered runtime work.
            from hermes_cli.session_recovery import recover_session_database
            hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle if p.exists()}
            report = recover_session_database(home / 'state.db', tmp_path / 'salvaged.db')
            assert report['verified'] and report['source_unchanged'], report
            assert report['runtime_state']['excluded_tables']['session_admissions']['rows'] == len(before)
            assert hashes == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle if p.exists()}
            assert rows() == before
            print(json.dumps({'live_recovery_verified': report['verified'], 'source_bundle_unchanged': True,
                              'excluded_admissions': len(before)}), flush=True)
            return snapshot

    try:
        with daemon(root, home, env, barrier=False) as (proc, desc):
            snapshot = asyncio.run(exercise(desc))
            old_epoch = desc['authority_epoch']
        peer.release.set()
        assert restore_quick_snapshot(snapshot, hermes_home=home)
        assert rows() == [('warm', 'terminal')]
        with daemon(root, home, env, barrier=False) as (_, restarted):
            assert restarted['instance_id'] != desc['instance_id']
            assert restarted['authority_epoch'] > old_epoch
            assert rows() == [('warm', 'terminal')]
            later_epoch = restarted['authority_epoch']
        # Restoring the same older snapshot must not reuse a previous owner's epoch.
        assert restore_quick_snapshot(snapshot, hermes_home=home)
        with daemon(root, home, env, barrier=False) as (_, restored_again):
            assert restored_again['authority_epoch'] > later_epoch
            print(json.dumps({'offline_restore_restart_epochs': [old_epoch, later_epoch, restored_again['authority_epoch']],
                              'restart_instance_changed': restored_again['instance_id'] != restarted['instance_id']}), flush=True)
    finally:
        peer.release.set()
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)


def test_import_and_startup_exclude_each_other_before_publication(tmp_path, monkeypatch):
    import subprocess
    import sys
    import zipfile
    from gateway.runtime_ownership import ProfileOwnership
    from hermes_cli import backup

    home = tmp_path / 'state'
    home.mkdir(mode=0o700)
    archive = tmp_path / 'restore.zip'
    with sqlite3.connect(tmp_path / 'donor.db') as db:
        db.execute('CREATE TABLE marker(value TEXT)')
        db.execute("INSERT INTO marker VALUES('restored')")
    with zipfile.ZipFile(archive, 'w') as zf:
        zf.writestr('config.yaml', 'gateway: {multiplex_profiles: false}\n')
        zf.write(tmp_path / 'donor.db', 'state.db')
    arrived, release = threading.Event(), threading.Event()
    publish = backup._extract_member_atomically

    def barrier(*args, **kwargs):
        arrived.set()
        assert release.wait(30)
        return publish(*args, **kwargs)

    monkeypatch.setattr(backup, '_extract_member_atomically', barrier)
    results = []

    def restore():
        with zipfile.ZipFile(archive) as zf:
            results.append(backup._import_members(zf, zf.namelist(), '', home, 2))

    thread = threading.Thread(target=restore)
    thread.start()
    try:
        assert arrived.wait(20)
        root = Path(__file__).resolve().parents[2]
        env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if k in os.environ}
        env.update(HOME=str(tmp_path / 'user'), HERMES_HOME=str(home), PYTHONPATH=str(root))
        contender = subprocess.run([sys.executable, '-c',
            "import logging, runpy; logging.basicConfig(level=logging.INFO); runpy.run_module('gateway.run', run_name='__main__')"], cwd=root, env=env,
                                   stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=25)
        assert 'already owns profile' in contender.stdout + contender.stderr, contender
        assert not (home / 'state.db').exists()
    finally:
        release.set()
        thread.join(timeout=30)
    assert not thread.is_alive() and results[0][0] == 2 and not results[0][2], results
    # Startup has reserved but not published a PID or opened the database.
    owner = ProfileOwnership()
    owner.reserve([home])
    try:
        before = (home / 'config.yaml').read_bytes(), (home / 'state.db').read_bytes()
        import pytest
        from gateway.runtime_ownership import OwnershipConflict
        with pytest.raises(OwnershipConflict, match='maintenance refused'):
            restore()
        assert before == ((home / 'config.yaml').read_bytes(), (home / 'state.db').read_bytes())
        from argparse import Namespace
        monkeypatch.setenv('HERMES_HOME', str(home))
        def forbidden_revival(*args):
            raise AssertionError('refused import must not revive a service')
        monkeypatch.setattr(backup, '_revive_gateway_after_import', forbidden_revival)
        with pytest.raises(SystemExit) as refused:
            backup.run_import(Namespace(zipfile=str(archive), force=True))
        assert refused.value.code == 1
    finally:
        owner.close()
    # Unrelated SQLite stores have no authority epoch; retain corruption restore.
    damaged = home / 'projects.db'
    damaged.write_bytes(b'not a SQLite database')
    assert backup._safe_restore_db(tmp_path / 'donor.db', damaged)
    with sqlite3.connect(damaged) as db:
        assert db.execute('SELECT value FROM marker').fetchall() == [('restored',)]


def test_recovery_output_exemption_excludes_canonical_activation(tmp_path):
    import hashlib
    import pytest
    from gateway.runtime_ownership import ProfileOwnership
    from hermes_cli.session_recovery import recover_session_database, SessionRecoverySafetyError
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch, admit_session_input

    source_home, target_home = tmp_path / 'source', tmp_path / 'target'
    source_home.mkdir()
    target_home.mkdir()
    source = source_home / 'state.db'
    owner = ProfileOwnership()
    owner.reserve([source_home, target_home])
    try:
        with SessionDB(db_path=source) as db:
            db.create_session('retained', 'cli')
            db.append_message('retained', 'user', 'retained content')
            epoch = begin_runtime_epoch(db, instance_id='source-owner')
            admit_session_input(db, epoch=epoch, principal_id='fixture', session_id='retained',
                                request_id='pending', payload={'text': 'must not execute in salvage'})
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        with pytest.raises(SessionRecoverySafetyError, match='maintenance refused'):
            recover_session_database(source, target_home / 'state.db')
        assert not (target_home / 'state.db').exists()
        report = recover_session_database(source, target_home / 'recovered.db')
        assert report['verified'] and report['source_unchanged'], report
        assert report['runtime_state']['excluded_tables']['session_admissions']['rows'] == 1
        with sqlite3.connect(target_home / 'recovered.db') as db:
            assert db.execute('SELECT content FROM messages').fetchall() == [('retained content',)]
            assert db.execute('SELECT COUNT(*) FROM session_admissions').fetchone()[0] == 0
            assert db.execute('SELECT COUNT(*) FROM runtime_epoch').fetchone()[0] == 0
        assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    finally:
        owner.close()
    report = recover_session_database(source, target_home / 'state.db')
    assert report['verified'] and not report['installed']
