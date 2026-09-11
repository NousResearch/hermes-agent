"""Real copy/restore regressions for the independent quality review's two blockers."""
from pathlib import Path
import sqlite3
from types import SimpleNamespace
import zipfile
from unittest.mock import Mock

import pytest
from hermes_cli import backup, profiles, kanban_history as history, kanban_db as kb


def test_clone_collision_preserves_competing_profile(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'config.yaml').write_bytes(b'source: true\n')
    target = tmp_path / 'profiles' / 'target'
    monkeypatch.setenv('HERMES_HOME', str(source))
    monkeypatch.setattr(profiles, 'get_profile_dir', lambda name: target)
    registration = Mock()
    monkeypatch.setattr(profiles, '_maybe_register_gateway_service', registration)
    original = history.refuse_authority_copy
    sentinels = {'config.yaml': b'competitor: true\r\n', 'gateway.pid': b'not-a-real-pid\x00', '.env': b'fixture-only\n'}

    def competing_creation(staged):
        original(staged)
        target.mkdir(parents=True)
        for name, data in sentinels.items():
            (target / name).write_bytes(data)

    monkeypatch.setattr(history, 'refuse_authority_copy', competing_creation)
    caught = None
    try:
        profiles.create_profile('target', clone_all=True, no_alias=True)
    except FileExistsError as exc:
        caught = exc
    actual = {p.relative_to(target).as_posix(): p.read_bytes() for p in target.rglob('*') if p.is_file()}
    assert actual == sentinels, 'competing profile must remain byte-for-byte unchanged, without nested clone'
    registration.assert_not_called()
    assert caught is not None, 'destination collision must fail, not return a competing profile'


@pytest.mark.parametrize('prefix', ['', '.hermes/'])
@pytest.mark.parametrize('location', [
    'skills/example/assets/kanban.db',
    'plugins/example/kanban.db',
    'profiles/outer/skills/fixture/kanban.db',
])
@pytest.mark.parametrize('kind', ['ordinary', 'unrelated_authority_table', 'enrolled'])
def test_zip_nonstandard_database_accepted(tmp_path, monkeypatch, prefix, location, kind):
    source = tmp_path / 'incoming' / location
    source.parent.mkdir(parents=True)
    if kind == 'enrolled':
        with kb.connect_closing(source) as db:
            kb.enroll_authority_history(db)
    else:
        with sqlite3.connect(source) as db:
            db.execute('CREATE TABLE ' + ('authority_roles' if kind == 'unrelated_authority_table' else 'asset') + '(name TEXT)')
    history.refuse_authority_copy(tmp_path / 'incoming')
    expected = source.read_bytes()
    target = tmp_path / 'target'
    # run_import resolves the root through get_hermes_home() (upstream moved it
    # off get_default_hermes_root so a profile restore cannot silently retarget
    # the live root). Patch both seams so the import lands in the fixture target.
    monkeypatch.setattr(backup, 'get_default_hermes_root', lambda: target)
    monkeypatch.setattr(backup, 'get_hermes_home', lambda: target)
    archive = tmp_path / 'fixture.zip'
    with zipfile.ZipFile(archive, 'w') as zf:
        zf.writestr(prefix + 'config.yaml', 'fixture: true\n')
        zf.write(source, prefix + location)
    backup.run_import(SimpleNamespace(zipfile=str(archive), force=True))
    assert (target / 'config.yaml').read_bytes() == b'fixture: true\n'
    assert (target / location).read_bytes() == expected


@pytest.mark.parametrize('prefix', ['', '.hermes/'])
@pytest.mark.parametrize('location', [
    'kanban.db', 'kanban/boards/example/kanban.db',
    'profiles/outer/kanban.db',
    'profiles/outer/profiles/inner/kanban/named/kanban.db',
    'skills/../kanban.db',
])
@pytest.mark.parametrize('kind', ['enrolled', 'malformed'])
def test_zip_standard_database_refused_before_overlay(tmp_path, monkeypatch, prefix, location, kind):
    source = tmp_path / 'fixture.db'
    if kind == 'enrolled':
        with kb.connect_closing(source) as db:
            kb.enroll_authority_history(db)
    else:
        source.write_bytes(b'not sqlite')
    target = tmp_path / 'target'
    target.mkdir()
    (target / 'config.yaml').write_bytes(b'original: true\n')
    # run_import resolves the root through get_hermes_home() (upstream moved it
    # off get_default_hermes_root so a profile restore cannot silently retarget
    # the live root). Patch both seams so the import lands in the fixture target.
    monkeypatch.setattr(backup, 'get_default_hermes_root', lambda: target)
    monkeypatch.setattr(backup, 'get_hermes_home', lambda: target)
    archive = tmp_path / 'fixture.zip'
    with zipfile.ZipFile(archive, 'w') as zf:
        zf.writestr(prefix + 'config.yaml', 'replacement: true\n')
        zf.write(source, prefix + location)
    with pytest.raises(history.AuthorityHistoryError):
        backup.run_import(SimpleNamespace(zipfile=str(archive), force=True))
    assert {p.relative_to(target).as_posix(): p.read_bytes() for p in target.rglob('*') if p.is_file()} == {'config.yaml': b'original: true\n'}


@pytest.mark.parametrize('prefix', ['', '.hermes/'])
def test_zip_authority_in_wal_refused(tmp_path, monkeypatch, prefix):
    source = tmp_path / 'fixture.db'
    archive = tmp_path / 'fixture.zip'
    with kb.connect_closing(source) as db:
        if db.execute('PRAGMA journal_mode').fetchone()[0].lower() != 'wal':
            # hermes_state refuses WAL on SQLite versions affected by the
            # wal-reset corruption bug and uses journal_mode=DELETE instead.
            # This fixture needs enrollment to live in the -wal sidecar, so the
            # precondition cannot be established here. A check that cannot run
            # must not return a verdict: skip rather than fail.
            pytest.skip('journal_mode is not WAL on this SQLite build')
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        base = source.read_bytes()
        kb.enroll_authority_history(db)
        assert source.read_bytes() == base, 'enrollment must reside in WAL for this fixture'
        wal = Path(str(source) + '-wal')
        assert wal.stat().st_size > 0
        with zipfile.ZipFile(archive, 'w') as zf:
            zf.writestr(prefix + 'config.yaml', 'fixture: true\n')
            zf.write(source, prefix + 'profiles/outer/kanban/named/kanban.db')
            zf.write(wal, prefix + 'profiles/outer/kanban/named/kanban.db-wal')
    target = tmp_path / 'target'
    # run_import resolves the root through get_hermes_home() (upstream moved it
    # off get_default_hermes_root so a profile restore cannot silently retarget
    # the live root). Patch both seams so the import lands in the fixture target.
    monkeypatch.setattr(backup, 'get_default_hermes_root', lambda: target)
    monkeypatch.setattr(backup, 'get_hermes_home', lambda: target)
    with pytest.raises(history.AuthorityHistoryError, match='authority'):
        backup.run_import(SimpleNamespace(zipfile=str(archive), force=True))
    assert not target.exists()


def test_zip_traversal_stays_blocked(tmp_path, monkeypatch, capsys):
    target = tmp_path / 'target'
    outside = tmp_path / 'outside' / 'kanban.db'
    outside.parent.mkdir()
    outside.write_bytes(b'outside fixture sentinel')
    # run_import resolves the root through get_hermes_home() (upstream moved it
    # off get_default_hermes_root so a profile restore cannot silently retarget
    # the live root). Patch both seams so the import lands in the fixture target.
    monkeypatch.setattr(backup, 'get_default_hermes_root', lambda: target)
    monkeypatch.setattr(backup, 'get_hermes_home', lambda: target)
    archive = tmp_path / 'fixture.zip'
    with zipfile.ZipFile(archive, 'w') as zf:
        zf.writestr('config.yaml', 'fixture: true\n')
        zf.writestr('../outside/kanban.db', 'replacement')
    backup.run_import(SimpleNamespace(zipfile=str(archive), force=True))
    assert outside.read_bytes() == b'outside fixture sentinel'
    assert 'path traversal blocked' in capsys.readouterr().out


def test_clone_uses_copy_not_cross_device_rename(tmp_path, monkeypatch):
    import errno
    import os
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'config.yaml').write_bytes(b'fixture: true\n')
    (source / 'gateway.pid').write_bytes(b'not-a-real-pid')
    target = tmp_path / 'profiles' / 'target'
    monkeypatch.setenv('HERMES_HOME', str(source))
    monkeypatch.setattr(profiles, 'get_profile_dir', lambda name: target)
    monkeypatch.setattr(profiles, '_maybe_register_gateway_service', lambda *a: None)
    # Simulate an environment where rename cannot cross devices. Real copytree
    # still performs all file I/O; this is not a physical second-volume test.
    monkeypatch.setattr(os, 'rename', Mock(side_effect=OSError(errno.EXDEV, 'fixture cross-device boundary')))
    assert profiles.create_profile('target', clone_all=True, no_alias=True) == target
    assert (target / 'config.yaml').read_bytes() == b'fixture: true\n'
    assert not (target / 'gateway.pid').exists()
    assert (source / 'gateway.pid').read_bytes() == b'not-a-real-pid'
