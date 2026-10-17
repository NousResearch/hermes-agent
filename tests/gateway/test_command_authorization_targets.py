import shlex

import pytest

from gateway.hooks import HookRegistry
from tests.gateway.test_slash_access_dispatch import _make_event, _make_runner, _make_source


def _runner(extra):
    runner = _make_runner(platform_extra=extra)
    runner._draining = False
    runner._restart_requested = False
    runner.hooks = HookRegistry()
    return runner


def _rewrite(runner, tmp_path, target):
    hook_dir = tmp_path / 'hooks' / 'rewrite'
    hook_dir.mkdir(parents=True)
    (hook_dir / 'HOOK.yaml').write_text('name: rewrite\nevents: ["command:help"]\n')
    (hook_dir / 'handler.py').write_text(
        f'def handle(event, context):\n    return {{"decision": "rewrite", "command_name": {target!r}}}\n'
    )
    runner.hooks.discover_and_load()


@pytest.mark.asyncio
@pytest.mark.parametrize('listed,denied', [('safe_cmd', 'safe-cmd'), ('safe-cmd', 'safe_cmd')])
@pytest.mark.parametrize('chat_type', ['dm', 'group'])
async def test_exec_quick_command_grants_keep_exact_keys(tmp_path, listed, denied, chat_type):
    prefix = '' if chat_type == 'dm' else 'group_'
    runner = _runner({prefix + 'allow_admin_from': ['owner'], prefix + 'user_allowed_commands': [listed]})
    markers = {name: tmp_path / name for name in (listed, denied)}
    runner.config.quick_commands = {
        name: {'type': 'exec', 'command': f'printf {shlex.quote(name)} > {shlex.quote(str(marker))}; printf {shlex.quote(name)}'}
        for name, marker in markers.items()
    }
    source = _make_source(user_id='reader', chat_type=chat_type)
    result = await runner._handle_message(_make_event('/' + denied, source))
    assert f'/{denied} is admin-only here' in result
    assert not markers[denied].exists()
    result = await runner._handle_message(_make_event('/' + listed, source))
    assert result == listed
    assert markers[listed].read_text() == listed


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['denied', 'admin', 'allowlisted', 'ungated'])
async def test_hook_rewrite_authorizes_restart_target(tmp_path, monkeypatch, mode):
    import gateway.run
    monkeypatch.setattr(gateway.run, '_hermes_home', tmp_path)
    monkeypatch.setattr('gateway.hooks.HOOKS_DIR', tmp_path / 'hooks')
    extra = {'allow_admin_from': ['owner']} if mode != 'ungated' else {}
    if mode == 'allowlisted':
        extra['user_allowed_commands'] = ['restart']
    runner = _runner(extra)
    _rewrite(runner, tmp_path, 'restart')
    requests = []
    runner.request_restart = lambda **kwargs: requests.append(kwargs)
    source = _make_source(user_id='owner' if mode == 'admin' else 'reader')
    result = await runner._handle_message(_make_event('/help', source))
    if mode == 'denied':
        assert '/restart is admin-only here' in result
        assert requests == []
        assert not (tmp_path / '.restart_notify.json').exists()
    else:
        assert 'Restarting' in str(result)
        assert len(requests) == 1
        assert (tmp_path / '.restart_notify.json').is_file()


def _plugin(monkeypatch):
    from hermes_cli import plugins
    manager = plugins.PluginManager()
    manager._discovered = True
    calls = []

    def handler(args):
        calls.append(args)
        return 'PLUGIN-RAN'

    context = plugins.PluginContext(plugins.PluginManifest(name='probe', source='user'), manager)
    context.register_command('my-cmd', handler)
    monkeypatch.setattr(plugins, 'get_plugin_manager', lambda: manager)
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize('spelling', ['my-cmd', 'my_cmd'])
@pytest.mark.parametrize('mode', ['denied', 'admin', 'allowlisted', 'ungated', 'hook-denied'])
async def test_plugin_alias_uses_registered_policy_and_hook(tmp_path, monkeypatch, spelling, mode):
    calls = _plugin(monkeypatch)
    extra = {'allow_admin_from': ['owner']} if mode != 'ungated' else {}
    if mode in {'allowlisted', 'hook-denied'}:
        extra['user_allowed_commands'] = ['my-cmd']
    runner = _runner(extra)
    if mode == 'hook-denied':
        hook_dir = tmp_path / 'hooks' / 'deny'
        hook_dir.mkdir(parents=True)
        (hook_dir / 'HOOK.yaml').write_text('name: deny\nevents: ["command:my-cmd"]\n')
        (hook_dir / 'handler.py').write_text(
            'def handle(event, context):\n    return {"decision": "deny", "message": "HOOK-DENIED"}\n'
        )
        monkeypatch.setattr('gateway.hooks.HOOKS_DIR', tmp_path / 'hooks')
        runner.hooks.discover_and_load()
    source = _make_source(user_id='owner' if mode == 'admin' else 'reader')
    result = await runner._handle_message(_make_event('/' + spelling + ' payload', source))
    if mode == 'denied':
        assert '/my-cmd is admin-only here' in result
        assert calls == []
    elif mode == 'hook-denied':
        assert result == 'HOOK-DENIED'
        assert calls == []
    else:
        assert result == 'PLUGIN-RAN'
        assert calls == ['payload']


@pytest.mark.asyncio
async def test_exec_quick_key_takes_precedence_over_plugin_alias(tmp_path, monkeypatch):
    calls = _plugin(monkeypatch)
    runner = _runner({'allow_admin_from': ['owner'], 'user_allowed_commands': ['my_cmd']})
    marker = tmp_path / 'quick'
    runner.config.quick_commands = {
        'my_cmd': {'type': 'exec', 'command': f'printf QUICK > {shlex.quote(str(marker))}; printf QUICK'}
    }
    source = _make_source(user_id='reader')
    result = await runner._handle_message(_make_event('/my_cmd', source))
    assert result == 'QUICK'
    assert marker.read_text() == 'QUICK'
    assert calls == []
    result = await runner._handle_message(_make_event('/my-cmd', source))
    assert '/my-cmd is admin-only here' in result
    assert calls == []
