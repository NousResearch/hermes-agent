"""Desktop install reviews the actual Git candidate before PM can publish it."""
import os
import subprocess

import pytest

from hermes_cli import plugin_catalog, plugins_cmd as pc


@pytest.fixture
def world(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    plugins = home / 'plugins'
    plugins.mkdir(parents=True)
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('GIT_CONFIG_COUNT', '1')
    monkeypatch.setenv('GIT_CONFIG_KEY_0', 'core.longpaths')
    monkeypatch.setenv('GIT_CONFIG_VALUE_0', 'true')
    monkeypatch.setattr(pc, '_plugins_dir', lambda: plugins)
    monkeypatch.setattr(plugin_catalog, 'fetch_live_catalog', lambda **kw: None)
    monkeypatch.setattr(plugin_catalog, 'load_removed_list', lambda *args, **kw: [])
    monkeypatch.setattr(pc.sys.stdin, 'isatty', lambda: False)
    monkeypatch.setattr(pc.sys.stdout, 'isatty', lambda: False)
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'plugin.yaml').write_text('name: consent-test\npython_dependencies: ["requests>=2,<3"]\n')
    (source / '__init__.py').write_text('def register(ctx):\n    pass\n')
    env = {k: v for k, v in os.environ.items() if k not in {'GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE'}}
    for args in [('init', '-q'), ('add', '.'), ('-c', 'user.name=test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'fixture')]:
        subprocess.run(['git', *args], cwd=source, env=env, check=True, capture_output=True, timeout=30)
    target = plugins / 'consent-test'
    target.mkdir()
    (target / 'plugin.yaml').write_text('name: consent-test\n')
    (target / '__init__.py').write_text('# old installation\n')
    (home / 'config.yaml').write_text('plugins:\n  enabled: [consent-test]\n')
    calls = []
    # No resolver, package install or environment swap in this local fixture.
    monkeypatch.setattr('pm.client.sync_venv', lambda **kw: calls.append(kw))
    return source, target, home, calls


def test_legacy_callers_keep_noninteractive_replacement_refusal(world):
    source, target, home, calls = world
    result = pc.dashboard_install_plugin(source.as_uri(), force=True, enable=False)
    assert not result['ok']
    assert 'dependency install skipped (non-interactive)' in result['error']
    assert not result.get('consent_required')
    assert not calls


def test_non_tty_replacement_returns_review_without_publication(world):
    source, target, home, calls = world
    before = {p: p.read_bytes() for p in [target / '__init__.py', target / 'plugin.yaml', home / 'config.yaml']}
    result = pc.dashboard_install_plugin(source.as_uri(), review_python_dependencies=True, force=True, enable=False)
    assert all(p.read_bytes() == content for p, content in before.items())
    assert not calls
    assert result.get('consent_required') is True, result
    assert result['python_dependencies'] == ['requests>=2,<3']
    assert result['dependency_consent']


@pytest.mark.parametrize('fresh', [False, True])
def test_acceptance_reaches_real_publication_and_stale_answer_does_not(world, monkeypatch, tmp_path, fresh):
    import shutil
    from pm import environments
    from pm.publication import StagedPlugin, PluginSelection
    from pm.plugin_inputs import StagedUpdate
    source, target, home, calls = world
    if fresh:
        shutil.rmtree(target)
        (home / 'config.yaml').write_text('plugins:\n  enabled: []\n')
    monkeypatch.setattr('hermes_cli.plugins_activation.activate_plugin_now', lambda name: {
        'gateway_reloaded': False, 'activation': None, 'restart_required': False})
    # Keep the actual journal + tree swap, substituting only dependency resolution.
    state = tmp_path / 'pm-state'
    monkeypatch.setattr(environments, 'dependency_home_root', lambda: home)
    monkeypatch.setattr('pm.publication.dependency_home_root', lambda: home)
    monkeypatch.setattr('pm.publication.install_state_dir', lambda project: state)
    monkeypatch.setattr('pm.publication.runtime_facts_path', lambda project: state / 'facts.json')
    def publish(**kwargs):
        calls.append(kwargs)
        kind = StagedPlugin if isinstance(kwargs['plugins'], StagedUpdate) else PluginSelection
        change = kind(dict(kwargs['plugins'].data))
        change.publish(tmp_path)
    monkeypatch.setattr('pm.client.sync_venv', publish)
    first = pc.dashboard_install_plugin(source.as_uri(), review_python_dependencies=True, force=True, enable=fresh)
    token = first['dependency_consent']
    # Changed install intent cannot borrow a previous answer.
    stale = pc.dashboard_install_plugin(source.as_uri(), review_python_dependencies=True, force=True, enable=not fresh, dependency_consent=token)
    assert stale['consent_required']
    assert not calls
    result = pc.dashboard_install_plugin(source.as_uri(), review_python_dependencies=True, force=True, enable=fresh, dependency_consent=token)
    assert result['ok'], result
    assert len(calls) == (2 if fresh else 1)
    if fresh:
        import hermes_yaml as yaml
        assert 'consent-test' in yaml.safe_load((home / 'config.yaml').read_text())['plugins']['enabled']
    assert (target / '__init__.py').read_text() == 'def register(ctx):\n    pass\n'
    assert (target.parent / '.install-metadata.json').is_file()


def test_real_rpc_preserves_review(world):
    from tui_gateway import server
    source, target, home, calls = world
    response = server.handle_request({'id': 'review', 'method': 'plugins.manage', 'params': {
        'action': 'install', 'identifier': source.as_uri(), 'force': True, 'enable': False,
    }})
    assert 'error' not in response, response
    assert response['result']['consent_required'] is True
    assert response['result']['python_dependencies'] == ['requests>=2,<3']
    assert not calls


@pytest.mark.parametrize('drift', ['commit', 'declaration', 'profile', 'source'])
def test_review_answer_cannot_follow_a_changed_candidate(world, tmp_path, drift):
    from hermes_cli.plugin_dependency_review import DependencyConsentRequired, review_install_dependencies
    source, target, home, calls = world
    record = {'source': source.as_uri(), 'revision': 'a' * 40}
    def review(answer=None):
        review_install_dependencies(source, target, record, enable=True, force=True, accepted=answer)
    with pytest.raises(DependencyConsentRequired) as pending:
        review()
    token = pending.value.token
    if drift == 'commit':
        record['revision'] = 'b' * 40
    elif drift == 'source':
        record['source'] = 'https://example.invalid/other'
    elif drift == 'profile':
        target = tmp_path / 'other-profile' / 'plugins' / 'consent-test'
    else:
        (source / 'plugin.yaml').write_text('name: consent-test\npython_dependencies: ["httpx>=0.28,<1"]\n')
    with pytest.raises(DependencyConsentRequired) as changed:
        review(token)
    assert changed.value.token != token
    assert not calls


def test_accepted_review_preserves_pm_refusal(world, monkeypatch):
    source, target, home, calls = world
    first = pc.dashboard_install_plugin(source.as_uri(), review_python_dependencies=True, force=True, enable=False)
    old = (target / '__init__.py').read_bytes()
    def refuse(**kwargs):
        raise RuntimeError('fixture dependency conflict')
    monkeypatch.setattr('pm.client.sync_venv', refuse)
    result = pc.dashboard_install_plugin(source.as_uri(), review_python_dependencies=True, force=True, enable=False,
                                         dependency_consent=first['dependency_consent'])
    assert not result['ok']
    assert 'fixture dependency conflict' in result['error']
    assert (target / '__init__.py').read_bytes() == old


def test_catalog_install_reviews_pinned_candidate_and_keeps_kill_list(world, monkeypatch):
    from hermes_cli import plugins_cmd_catalog as catalog
    source, target, home, calls = world
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    entry = plugin_catalog.PluginCatalogEntry(name='consent-test', repo=source.as_uri(), sha=sha,
                                               description='fixture', maintainer='test')
    monkeypatch.setattr(catalog, 'get_live_catalog_entry', lambda name: entry)
    result = pc.dashboard_install_plugin('', review_python_dependencies=True, force=True, enable=False, catalog_name='consent-test')
    assert result['consent_required'], result
    assert result['python_dependencies'] == ['requests>=2,<3']
    monkeypatch.setattr(plugin_catalog, 'load_removed_list', lambda *args, **kwargs: [
        plugin_catalog.RemovedEntry(name='consent-test', repo=source.as_uri(), reason='fixture recall')])
    result = pc.dashboard_install_plugin('', review_python_dependencies=True, force=True, enable=False, catalog_name='consent-test',
                                         dependency_consent=result['dependency_consent'])
    assert not result['ok']
    assert 'fixture recall' in result['error']
    assert not calls


@pytest.mark.parametrize('kind', ['fresh', 'pyproject', 'external', 'none', 'invalid'])
def test_declaration_boundaries(world, kind):
    import shutil
    source, target, home, calls = world
    if kind == 'fresh':
        shutil.rmtree(target)
    else:
        (source / 'plugin.yaml').write_text('name: consent-test\n' + (
            'python_runtime: external\npython_dependencies: ["requests>=2,<3"]\n' if kind == 'external' else ''))
        if kind == 'pyproject':
            (source / 'pyproject.toml').write_text('[project]\nname="consent-test"\nversion="1.0"\n')
        elif kind == 'invalid':
            (source / 'pyproject.toml').write_text('[project\n')
        subprocess.run(['git', 'add', '.'], cwd=source, check=True, capture_output=True)
        subprocess.run(['git', '-c', 'user.name=test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', kind], cwd=source, check=True, capture_output=True)
    result = pc.dashboard_install_plugin(source.as_uri(), review_python_dependencies=True, force=True, enable=False)
    if kind in {'fresh', 'pyproject'}:
        assert result['consent_required'], result
        assert not calls
    elif kind == 'invalid':
        assert not result['ok']
        assert not calls
    else:
        assert not result.get('consent_required')
        assert calls

