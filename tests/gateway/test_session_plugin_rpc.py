"""A plugin's frontend reaches its own Python half through the session authority, and back."""
import threading
from types import SimpleNamespace

import pytest

import hermes_yaml as yaml
from tests.gateway.test_surface_admission import _authority

_PLUGIN = '''
import threading


async def _thread(params, *, session_id, **kw):
    return threading.get_ident()


def register(ctx):
    ctx.register_rpc("pane.state", lambda params, *, session_id, **kw: {"n": params["n"], "session_id": session_id})
    ctx.register_rpc("pane.thread", _thread)
'''


async def _answer(event):
    return 'ok'


def _live(platform, chat_id, **source):
    from gateway.session import SessionSource
    from gateway.session_authority import LiveSession
    return LiveSession(SessionSource(platform=platform, chat_id=chat_id, **source), chat_id)


@pytest.mark.asyncio
async def test_plugin_method_is_gated_like_a_session_verb_and_leaves_with_its_plugin(tmp_path, monkeypatch):
    from gateway.config import Platform
    from gateway.session_controls import AuthorityConnection
    from hermes_cli import plugins

    authority = await _authority(tmp_path, monkeypatch, _answer)
    plugin = tmp_path / 'plugins' / 'pane-demo'
    plugin.mkdir(parents=True)
    (plugin / 'plugin.yaml').write_text(yaml.safe_dump({'name': 'pane-demo', 'version': '0.1.0'}))
    (plugin / '__init__.py').write_text(_PLUGIN)
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump({'plugins': {'enabled': ['pane-demo']}}))
    manager = plugins.PluginManager()
    monkeypatch.setattr(plugins, '_plugin_manager', manager)
    authority.sessions['local-x'] = _live(Platform.LOCAL, 'local-x', user_id='someone-else')
    owner = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    reader = AuthorityConnection(authority, object(), {'user_id': 'owner', 'capabilities': ['session:read']})

    async def call(connection, **params):
        reply = await connection.dispatch({'id': 1, 'method': 'plugin.pane-demo.pane.state', 'params': params})
        return reply.get('result', reply.get('error'))
    try:
        assert await call(owner, session_id='s', n=3) == {'n': 3, 'session_id': 's'}
        assert (await call(owner, n=4))['n'] == 4
        assert (await call(reader, session_id='s', n=3))['data']['reason'] == 'permission_denied'
        assert (await call(owner, session_id='local-x', n=3))['data']['reason'] == 'permission_denied'
        assert (await call(owner, session_id='s'))['data']['reason'] == 'plugin_rpc_failed'
        # An async handler's body runs in the worker too, so blocking work in it never stalls the loop.
        reply = await owner.dispatch({'id': 2, 'method': 'plugin.pane-demo.pane.thread', 'params': {}})
        assert reply['result'] != threading.get_ident()
        manager.unload('pane-demo')
        assert (await call(owner, session_id='s', n=3))['code'] == -32601
    finally:
        await owner.close()
        await reader.close()


@pytest.mark.asyncio
async def test_plugin_session_event_reaches_only_that_sessions_viewers(tmp_path, monkeypatch):
    from gateway import run_api
    from gateway.config import Platform
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    authority = await _authority(tmp_path, monkeypatch, _answer)
    authority.sessions['t'] = _live(Platform.TELEGRAM, 't')
    frames = {key: [] for key in ('s', 't')}
    for key, seen in frames.items():
        authority.sessions[key].event_stream.observers.add(seen.append)
    # Compression gave the agent of conversation 's' a new id; its viewers stay on 's'.
    agent = SimpleNamespace(session_id='s-compressed')
    authority.runner._cached_agent_for = lambda route: agent if route == 's' else None
    monkeypatch.setattr(run_api, '_ACTIVE_RUNNER', authority.runner)
    ctx = PluginContext(PluginManifest(name='pane-demo'), PluginManager())

    assert ctx.emit_to_session('s-compressed', 'pane.changed', {'n': 1}) is True
    assert [(f['params']['type'], f['params']['session_id'], f['params']['payload']) for f in frames['s']] == [
        ('plugin.pane-demo.pane.changed', 's', {'n': 1})]
    assert frames['t'] == []
    assert ctx.emit_to_session('nobody', 'pane.changed') is False
