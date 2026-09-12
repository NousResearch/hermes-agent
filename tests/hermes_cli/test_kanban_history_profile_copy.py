"""Supported profile copy/import must not activate a copied authority identity."""
from pathlib import Path
import tarfile
import zipfile
from types import SimpleNamespace
from hermes_cli import backup
import pytest
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import profiles


@pytest.mark.parametrize('operation', ['clone', 'import'])
@pytest.mark.parametrize('enrolled', [False, True])
def test_profile_copy_refuses_enrolled_identity(tmp_path, monkeypatch, operation, enrolled):
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    root = tmp_path / '.hermes'
    root.mkdir()
    source = root / 'profiles' / 'source'
    source.mkdir(parents=True)
    monkeypatch.setenv('HERMES_HOME', str(source))
    # Gateway service registration is a process activation boundary; never run it.
    monkeypatch.setattr(profiles, '_maybe_register_gateway_service', lambda *a: None)
    with kbc.connect_closing(source / 'kanban.db') as db:
        if enrolled: kb.enroll_authority_history(db)
    archive = tmp_path / 'source.tar.gz'
    if operation == 'import':
        with tarfile.open(archive, 'w:gz') as tf:
            tf.add(source, arcname='source')
    def invoke():
        if operation == 'clone':
            return profiles.create_profile('target', clone_from='source', clone_all=True, no_alias=True)
        return profiles.import_profile(str(archive), name='target')
    if enrolled:
        with pytest.raises(ValueError, match='authority'):
            invoke()
        assert not (root / 'profiles' / 'target').exists()
    else:
        target = invoke()
        assert (target / 'kanban.db').is_file()


@pytest.mark.parametrize('operation', ['zip', 'quick'])
@pytest.mark.parametrize('authority_location', ['incoming', 'target', 'ordinary'])
def test_restore_refuses_copied_or_rolled_back_authority(tmp_path, monkeypatch, operation, authority_location):
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    root = tmp_path / '.hermes'
    root.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(root))
    incoming = tmp_path / 'incoming'
    incoming.mkdir()
    for directory, location in ((root, 'target'), (incoming, 'incoming')):
        with kbc.connect_closing(directory / 'kanban.db') as db:
            if authority_location == location: kb.enroll_authority_history(db)
    before = (root / 'kanban.db').read_bytes()
    if operation == 'zip':
        archive = tmp_path / 'backup.zip'
        with zipfile.ZipFile(archive, 'w') as zf:
            zf.writestr('config.yaml', 'fixture: true')
            zf.write(incoming / 'kanban.db', 'kanban.db')
        invoke = lambda: backup.run_import(SimpleNamespace(zipfile=str(archive), force=True))
    else:
        import json
        import shutil
        snapshot = root / 'state-snapshots' / 'fixture'
        snapshot.mkdir(parents=True)
        shutil.copy2(incoming / 'kanban.db', snapshot / 'kanban.db')
        (snapshot / 'manifest.json').write_text(json.dumps({'files': {'kanban.db': 1}}))
        invoke = lambda: backup.restore_quick_snapshot('fixture', hermes_home=root)
    if authority_location != 'ordinary':
        with pytest.raises(ValueError, match='authority'): invoke()
        assert (root / 'kanban.db').read_bytes() == before
        assert not (root / 'config.yaml').exists()
    else:
        invoke()
        with kbc.connect_closing(root / 'kanban.db') as db:
            assert kb.authority_history_capability(db) is None
