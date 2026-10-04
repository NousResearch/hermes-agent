"""Exercise archive installs through the real headless connector."""
from unittest.mock import Mock
import pytest
from tools import skill_usage, skills_sync, skills_hub
from tools.connectors.catalog import HostInstaller
from hermes_cli import skills_hub as cli


@pytest.fixture
def archive(tmp_path, monkeypatch):
    root = tmp_path / 'skills'
    archived = root / '.archive' / 'pdf'
    archived.mkdir(parents=True)
    (archived / 'SKILL.md').write_text('---\nname: pdf\ndescription: test\n---\nOLD\n')
    monkeypatch.setattr(skill_usage, '_skills_dir', lambda: root)
    monkeypatch.setattr(skills_sync, '_skills_dir', lambda: root)
    monkeypatch.setattr(skills_hub, '_skills_dir', lambda: root)
    monkeypatch.setattr(skills_hub, '_hub_dir', lambda: root / '.hub')
    monkeypatch.setattr(cli, '_finish_change', lambda *a: None)
    return root


def test_real_host_installer_restore(archive, monkeypatch):
    fetch = Mock(side_effect=AssertionError('network must not run'))
    monkeypatch.setattr(cli, '_install_skill', fetch)
    result = HostInstaller().install_skill('pdf', force=False)
    assert result['name'] == 'pdf'
    assert result['already_installed'] is False
    assert (archive / 'pdf' / 'SKILL.md').exists()
    fetch.assert_not_called()
    # A local archive cannot establish any remote registry provenance. Keep it
    # curator-managed rather than inventing a lock entry with a guessed source.
    assert skills_hub.HubLockFile().get_installed('pdf') is None
    assert skill_usage.archive_skill('pdf')[0]
    assert not (archive / 'pdf').exists()
    assert skill_usage.restore_skill('pdf')[0]


@pytest.mark.parametrize('identifier,source', [('official/pdf',''), ('https://host/pdf',''), ('org/repo/pdf',''), ('pdf','official')])
def test_qualified_identifier_not_hijacked(archive, monkeypatch, identifier, source):
    fetch = Mock(return_value=(None, 'cancelled'))
    monkeypatch.setattr(cli, '_install_skill', fetch)
    cli.do_install(identifier, source_id=source, invalidate_cache=False)
    fetch.assert_called_once()
    assert (archive / '.archive' / 'pdf').exists()


def test_existing_hub_update_keeps_registry_pin(archive, monkeypatch):
    lock = skills_hub.HubLockFile()
    lock.record_install(name='pdf', source='github', identifier='pdf',
                        trust_level='community', scan_verdict='safe', skill_hash='old',
                        install_path='pdf', files=['SKILL.md'])
    before = lock.path.read_bytes()
    fetch = Mock(return_value=(None, 'cancelled'))
    monkeypatch.setattr(cli, '_install_skill', fetch)
    from tools import skills_hub_install
    monkeypatch.setattr(skills_hub_install, 'check_for_skill_updates', lambda **kw: [
        {'name': 'pdf', 'source': 'github', 'identifier': 'pdf', 'status': 'update_available'}])
    monkeypatch.setattr(cli, '_has_local_edits', lambda entry: False)
    cli.do_update('pdf')
    fetch.assert_called_once()
    assert fetch.call_args.args[-1] == 'github'
    assert lock.path.read_bytes() == before
    assert (archive / '.archive' / 'pdf').exists()


def test_explicit_bundled_install_uses_current_source(archive, tmp_path, monkeypatch):
    bundled = tmp_path / 'bundled'
    current = bundled / 'category' / 'pdf'
    current.mkdir(parents=True)
    (current / 'SKILL.md').write_text('---\nname: pdf\ndescription: test\n---\nCURRENT\n')
    (archive / '.curator_suppressed').write_text('pdf\n')
    monkeypatch.setattr(skills_sync, '_get_bundled_dir', lambda: bundled)
    monkeypatch.setattr(cli, '_install_skill', Mock(side_effect=AssertionError('network')))
    HostInstaller().install_skill('pdf', force=False)
    assert 'CURRENT' in (archive / 'category' / 'pdf' / 'SKILL.md').read_text()
    assert 'OLD' in (archive / '.archive' / 'pdf' / 'SKILL.md').read_text()
    assert 'pdf' not in skill_usage.read_suppressed_names()
    assert skills_sync._read_manifest()['pdf'] == skills_sync._dir_hash(current)
