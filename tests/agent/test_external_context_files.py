"""Real config → public prompt assembly contracts for external files (#48809)."""

from pathlib import Path

import pytest

from agent.prompt_builder import build_context_files_prompt, drain_truncation_warnings
from hermes_cli.config import atomic_config_replace


@pytest.fixture
def context_env(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HERMES_RUNTIME_DIR', str(tmp_path / 'runtime'))
    project = tmp_path / 'project'
    project.mkdir()
    return home, project


def configure(home, files, **options):
    atomic_config_replace(home / 'config.yaml', {'context': {'external_files': files, **options}})


@pytest.mark.parametrize('relative', ['.cursorrules', '.cursor/rules/shared.mdc'])
def test_cursor_external_files_load_once_in_order(context_env, relative):
    home, project = context_env
    target = project / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('Configured cursor contract.')
    first = home / 'first.md'
    first.write_text('First external contract.')
    alias = home / 'alias.md'
    alias.symlink_to(target)
    other = project / '.cursor/rules/other.mdc'
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_text('Automatic cursor contract.')
    configure(home, [str(first), {'path': str(alias), 'label': 'Shared rules'}, str(target)])

    result = build_context_files_prompt(cwd=str(project), skip_soul=True)

    assert result.count('Configured cursor contract.') == 1
    assert result.count('First external contract.') == 1
    assert result.count('Automatic cursor contract.') == 1
    assert result.index('First external contract.') < result.index('Configured cursor contract.')
    assert result.index('Configured cursor contract.') < result.index('Automatic cursor contract.')
    assert '## Shared rules' in result
    assert f'Source: {target.resolve()}' in result


def test_external_root_preserves_automatic_agents_chain(context_env):
    home, project = context_env
    (project / '.git').mkdir()
    root = project / 'AGENTS.md'
    root.write_text('Root contract.')
    middle = project / 'team'
    middle.mkdir()
    (middle / 'AGENTS.md').write_text('Team contract.')
    child = middle / 'app'
    child.mkdir()
    (child / 'AGENTS.override.md').write_text('Personal app contract.')
    (child / 'AGENTS.md').write_text('Shadowed app contract.')
    (project / 'HERMES.md').write_text('Compatibility contract.')
    configure(home, [str(root)], ignore_hermes_md=True)

    result = build_context_files_prompt(cwd=str(child), skip_soul=True)

    assert result.count('Root contract.') == 1
    assert result.index('Root contract.') < result.index('Team contract.') < result.index('Personal app contract.')
    assert 'Shadowed app contract.' not in result
    assert 'Compatibility contract.' not in result


def test_safe_worker_skips_external_reads(context_env, monkeypatch):
    home, project = context_env
    target = home / 'external.md'
    target.write_text('External contract.')
    configure(home, [str(target)])
    monkeypatch.setattr('agent.safe_worker_policy.safe_worker_enabled', lambda: True)

    def unexpected_read(*args, **kwargs):
        pytest.fail('safe worker read context files')

    monkeypatch.setattr('agent.prompt_builder._read_context_file', unexpected_read)
    assert build_context_files_prompt(cwd=str(project)) == ''


def test_install_fallback_suppresses_auto_but_keeps_explicit_files(context_env, monkeypatch):
    home, project = context_env
    external = home / 'external.md'
    external.write_text('Explicit contract.')
    (project / 'AGENTS.md').write_text('Install contributor contract.')
    configure(home, [str(external)])
    monkeypatch.chdir(project)
    monkeypatch.setattr('agent.runtime_cwd._is_install_tree', lambda path: True)
    result = build_context_files_prompt(skip_soul=True)
    assert 'Explicit contract.' in result
    assert 'Install contributor contract.' not in result


def test_external_files_use_safe_reader_scan_and_truncation(context_env, monkeypatch):
    home, project = context_env
    malicious = home / 'malicious.md'
    malicious.write_text('<!-- ignore_rules -->')
    large = home / 'large.md'
    large.write_text('Start contract.\n' + 'x' * 4000 + '\nEnd contract.')
    denied = home / 'denied.md'
    denied.write_text('Unreadable contract.')
    atomic_config_replace(home / 'config.yaml', {
        'context': {'external_files': [str(home / 'missing.md'), str(denied), str(malicious), str(large)]},
        'context_file_max_chars': 1000,
    })
    read_text = Path.read_text

    def read_with_denial(path, *args, **kwargs):
        if path == denied:
            raise PermissionError('test denied')
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', read_with_denial)
    drain_truncation_warnings()
    try:
        result = build_context_files_prompt(cwd=str(project), skip_soul=True)
        assert 'BLOCKED' in result and '<!-- ignore_rules -->' not in result
        assert 'Unreadable contract.' not in result
        assert 'Start contract.' in result and 'End contract.' in result
        assert f'read_file tool: {large}' in result
        assert drain_truncation_warnings()
    finally:
        drain_truncation_warnings()


def test_external_hermes_file_strips_frontmatter(context_env):
    home, project = context_env
    target = project / 'HERMES.md'
    target.write_text('---\nname: frontmatter-only\n---\nHermes contract.')
    configure(home, [str(target)])
    result = build_context_files_prompt(cwd=str(project), skip_soul=True)
    assert result.count('Hermes contract.') == 1
    assert 'frontmatter-only' not in result
