"""RAM-only source contract exercising native scanning and invocation."""
from pathlib import Path
import pytest
from hermes_constants import set_hermes_home_override, reset_hermes_home_override
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from agent.skill_commands import get_interactive_skill_commands, build_skill_invocation_message


@pytest.fixture
def scope(tmp_path):
    token = set_hermes_home_override(str(tmp_path))
    try:
        yield tmp_path
    finally:
        reset_hermes_home_override(token)


def source(catalog=None, loader=None):
    manager = PluginManager()
    manager._discovered = True
    context = PluginContext(PluginManifest(name='remote-test', version='1'), manager)
    handle = context.register_skill_source(
        'catalog', list_skills=catalog or (lambda: [
            {'name': 'interview-me', 'description': 'Ask one question', 'uri': 'test://interview'}]),
        load_skill=loader or (lambda uri: {'name': 'interview-me', 'content': 'Ask only one question.'}),
    )
    return manager, handle


def test_public_registration(scope):
    manager, handle = source()
    assert manager.list_skill_source_commands()['/interview-me']['name'] == 'interview-me'


def test_native_scan_and_invocation_without_files(scope, monkeypatch):
    manager, handle = source()
    monkeypatch.setattr('hermes_cli.plugins.get_plugin_manager', lambda: manager)
    # Let existing native profile bootstrapping run before auditing the remote path.
    get_interactive_skill_commands()
    before = set(scope.rglob('*'))
    commands = get_interactive_skill_commands()
    assert '/interview-me' in commands
    message = build_skill_invocation_message('/interview-me', 'help me plan')
    assert 'Ask only one question.' in message
    assert 'help me plan' in message
    assert 'Skill directory:' not in message
    assert not any(p.name == 'SKILL.md' for p in scope.rglob('*'))
    assert not any(p.suffix == '.md' for p in set(scope.rglob('*')) - before)


def test_revocation_not_cached(scope, monkeypatch):
    catalog = [{'name': 'interview-me', 'uri': 'test://interview'}]
    manager, handle = source(catalog=lambda: catalog)
    monkeypatch.setattr('hermes_cli.plugins.get_plugin_manager', lambda: manager)
    assert '/interview-me' in get_interactive_skill_commands()
    catalog.clear()
    assert '/interview-me' not in get_interactive_skill_commands()
    assert build_skill_invocation_message('/interview-me') is None


def test_unload_removes_source(scope):
    manager, handle = source()
    handle.dispose()
    assert manager.list_skill_source_commands() == {}


def test_invalid_and_builtin_slugs_filtered(scope):
    manager, handle = source(catalog=lambda: [
        {'name': 'help', 'uri': 'test://help'}, {'name': '../other', 'uri': 'test://other'},
        {'name': 'interview-me', 'uri': 'test://interview'},
        {'name': 'interview-me', 'uri': 'test://duplicate'},
    ])
    assert manager.list_skill_source_commands() == {}


def test_remote_no_shell_preprocessing(scope, monkeypatch):
    manager, handle = source(loader=lambda uri: {
        'name': 'interview-me', 'content': '!`touch should-not-exist`'})
    monkeypatch.setattr('hermes_cli.plugins.get_plugin_manager', lambda: manager)
    message = build_skill_invocation_message('/interview-me')
    assert '!`touch should-not-exist`' in message
    assert not Path('should-not-exist').exists()


def test_load_failure_visible_and_sanitized(scope, monkeypatch):
    def fail(uri):
        raise RuntimeError('SECRET do not expose')
    manager, handle = source(loader=fail)
    monkeypatch.setattr('hermes_cli.plugins.get_plugin_manager', lambda: manager)
    message = build_skill_invocation_message('/interview-me')
    assert 'unavailable' in message
    assert 'SECRET' not in message


def test_remote_frontmatter_cannot_read_local_config(scope, monkeypatch):
    sentinel = 'LOCAL_CONFIG_VALUE_NOT_AUTHORIZED_FOR_REMOTE'
    (scope / 'config.yaml').write_text(
        'skills:\n  config:\n    private:\n      token: ' + sentinel + '\n')
    content = ('---\nmetadata:\n  hermes:\n    config:\n'
               '      - key: private.token\n        description: retrieve local value\n---\nInterview.')
    manager, handle = source(loader=lambda uri: {'name': 'interview-me', 'content': content})
    monkeypatch.setattr('hermes_cli.plugins.get_plugin_manager', lambda: manager)
    message = build_skill_invocation_message('/interview-me')
    assert sentinel not in message
    assert '[Skill config' not in message


def test_remote_invocation_does_not_write_usage(scope, monkeypatch):
    manager, handle = source()
    monkeypatch.setattr('hermes_cli.plugins.get_plugin_manager', lambda: manager)
    get_interactive_skill_commands()
    assert not (scope / 'skills' / '.usage.json').exists()
    build_skill_invocation_message('/interview-me')
    assert not list((scope / 'skills').glob('.usage*'))


def test_canonical_plugin_disable_revokes_source(scope, monkeypatch):
    manager = PluginManager()
    context = PluginContext(PluginManifest(name='display-name', version='1'), manager)
    monkeypatch.setattr(type(context), 'plugin_id', property(lambda self: 'canonical-key'))
    handle = context.register_skill_source('remote',
        list_skills=lambda: [{'name': 'interview-me', 'uri': 'test://interview'}],
        load_skill=lambda uri: {'name': 'interview-me', 'content': 'TEXT_NOT_ALLOWED'})
    info = manager.list_skill_source_commands()['/interview-me']
    (scope / 'config.yaml').write_text('plugins:\n  disabled:\n    - canonical-key\n')
    assert not manager.list_skill_source_commands()
    assert 'TEXT_NOT_ALLOWED' not in manager.load_skill_source_payload(info)['content']
    handle.dispose()


def test_duplicate_remote_commands_are_unavailable(scope):
    manager, handle = source()
    stale = manager.list_skill_source_commands()['/interview-me']
    context = PluginContext(PluginManifest(name='other', version='1'), manager)
    other = context.register_skill_source('remote',
        list_skills=lambda: [{'name': 'interview-me', 'uri': 'test://other'}],
        load_skill=lambda uri: {'name': 'interview-me', 'content': 'AMBIGUOUS'})
    assert '/interview-me' not in manager.list_skill_source_commands()
    assert 'Ask only' not in manager.load_skill_source_payload(stale)['content']
    other.dispose()
    assert '/interview-me' in manager.list_skill_source_commands()
    handle.dispose()


def test_revocation_during_load_does_not_return_content(scope):
    catalog = [{'name': 'interview-me', 'uri': 'test://interview'}]
    def loader(uri):
        catalog.clear()
        return {'name': 'interview-me', 'content': 'REVOKED_CONTENT'}
    manager, handle = source(catalog=lambda: catalog, loader=loader)
    info = manager.list_skill_source_commands()['/interview-me']
    assert 'REVOKED_CONTENT' not in manager.load_skill_source_payload(info)['content']
    handle.dispose()


def test_unload_during_load_does_not_return_content(scope):
    def loader(uri):
        handle.dispose()
        return {'name': 'interview-me', 'content': 'UNLOADED_CONTENT'}
    manager, handle = source(loader=loader)
    info = manager.list_skill_source_commands()['/interview-me']
    assert 'UNLOADED_CONTENT' not in manager.load_skill_source_payload(info)['content']


def test_profile_switch_does_not_reuse_source(scope):
    manager, handle = source()
    token = set_hermes_home_override(str(scope / 'other'))
    try:
        assert manager.list_skill_source_commands() == {}
    finally:
        reset_hermes_home_override(token)
    assert '/interview-me' in manager.list_skill_source_commands()
