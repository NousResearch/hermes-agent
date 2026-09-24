"""GET is canonical-data-read-only; SQLite WAL coordination is explicitly allowed."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
from contextlib import closing
import pytest
import yaml
spec = importlib.util.spec_from_file_location('storage_detail_fixture', Path(__file__).with_name('test_task_detail_native.py'))
assert spec is not None and spec.loader is not None
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)
rig = f.rig


def inventory(root):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()}


def rows(conn):
    # Include dependencies and every other canonical table, not just task rows.
    return '\n'.join(conn.iterdump())


@pytest.mark.asyncio
@pytest.mark.parametrize('live_wal', [False, True])
async def test_get_cold_baseline_and_latest_committed_wal_preserve_canonical_data(rig, live_wal):
    r = rig
    with closing(sqlite3.connect(r.path)) as conn:
        conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        journal_mode = str(conn.execute('PRAGMA journal_mode').fetchone()[0]).lower()
        canonical = rows(conn)
    # This baseline precedes ANY inspection helper's read-only open.
    cold = inventory(r.path.parent)
    assert r.path.name + '-wal' not in cold and r.path.name + '-shm' not in cold
    writer = sqlite3.connect(r.path) if live_wal else None
    try:
        if writer:
            if journal_mode == 'wal':
                writer.execute('PRAGMA wal_autocheckpoint=0')
            writer.execute('UPDATE tasks SET title=? WHERE id=?', ('Latest committed WAL title', r.tid))
            writer.commit()
            canonical = rows(writer)
            if journal_mode == 'wal':
                assert r.path.with_name(r.path.name + '-wal').stat().st_size > 0
        before = inventory(r.path.parent)
        response = await f.http(r.app, headers=r.headers)
        after = inventory(r.path.parent)
        assert response.status == 200
        assert after[r.path.name] == before[r.path.name]
        sidecars = ({r.path.name + '-wal', r.path.name + '-shm'}
                    if journal_mode == 'wal' else set())
        assert set(after) - set(before) <= sidecars
        with closing(sqlite3.connect(r.path.as_uri() + '?mode=ro', uri=True)) as conn:
            assert rows(conn) == canonical
        if live_wal:
            assert json.loads(response.text)['title'] == 'Latest committed WAL title'
        else:
            assert set(after) - set(before) == sidecars
    finally:
        if writer:
            writer.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('policy', ['valid', 'user-null', 'user-list', 'managed-malformed', 'managed-list'])
async def test_denied_get_never_projects_or_backs_up_changed_policy(rig, monkeypatch, policy):
    r = rig
    backups = r.home / 'backups/config'
    for path in tuple(backups.glob('config.yaml.good.*')):
        path.rename(path.with_name(path.name + '.retained-prior'))
    r.config['synthetic_revision'] = 'new valid revision avoids filename collision'
    if policy.startswith('user'):
        r.config['kanban'] = None if policy == 'user-null' else ['invalid']
    (r.home / 'config.yaml').write_text(yaml.safe_dump(r.config))
    if policy.startswith('managed'):
        managed = r.home / 'managed'
        managed.mkdir()
        monkeypatch.setenv('HERMES_MANAGED_DIR', str(managed))
        (managed / 'config.yaml').write_text('kanban: [unterminated' if policy == 'managed-malformed' else '- invalid')
        # Prove an ordinary fail-open cache cannot supply policy to GET.
        from hermes_cli.config_effective import load_user_config_effective
        load_user_config_effective()
    before = {str(p.relative_to(r.home)): p.read_bytes() for p in r.home.rglob('*') if p.is_file()}
    def forbidden(_):
        pytest.fail('Denied request reached projection')
    monkeypatch.setattr(r.service, '_project', forbidden)
    headers = r.headers if policy != 'valid' else {**r.headers, 'X-Telegram-Init-Data': 'invalid'}
    response = await f.http(r.app, headers=headers)
    assert response.status == 403 and json.loads(response.text) == {'error': 'unavailable'}
    assert {str(p.relative_to(r.home)): p.read_bytes() for p in r.home.rglob('*') if p.is_file()} == before
