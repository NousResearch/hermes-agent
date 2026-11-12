"""S4: plugins read which client created the session (``session_client()``, prompt-section info)."""
import contextlib
import json
from types import SimpleNamespace

import pytest

from tests.hermes_cli.test_plugin_host import _home_with_plugins

TERN = {'name': 'hermes-tern', 'version': '0.4.1'}

SECTION_PLUGIN = '''
def register(ctx):
    ctx.register_system_prompt_section(
        "probe.client", lambda info: "platform=" + info["platform"] + " client=" + info["client"]
        + " version=" + info["client_version"])
'''


@contextlib.contextmanager
def _turn_of(tmp_path, monkeypatch, **request):
    """The task-local state a local session's turn runs under (``policy_scope`` sets the same var)."""
    from gateway import session_policy
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    policy = session_policy.build_policy({'source': 'tui', 'cwd': str(tmp_path), **request}, {'model': {'default': 'm'}})
    token = session_policy._ACTIVE_POLICY.set(policy)
    try:
        yield policy
    finally:
        session_policy._ACTIVE_POLICY.reset(token)


def test_session_client_answers_for_the_running_session_only(tmp_path, monkeypatch):
    from gateway.session_context import clear_session_vars, set_session_vars
    from hermes_cli.plugin_session import session_client
    assert session_client() is None  # outside any session
    with _turn_of(tmp_path, monkeypatch, client=dict(TERN)):
        first = session_client()
        assert first == TERN
        first['name'] = 'changed'  # a copy: the frozen request never changes
        assert session_client() == TERN
    with _turn_of(tmp_path, monkeypatch):
        assert session_client() is None  # Ink / Desktop / CLI / ACP send none
    tokens = set_session_vars(platform='telegram', chat_id='1', session_key='k')
    try:
        assert session_client() is None  # a messaging chat
    finally:
        clear_session_vars(tokens)


def test_a_stored_label_that_is_not_valid_reads_as_none():
    from gateway.session_client import policy_client
    for request in ({'client': {'name': 'Bad Name'}}, {'client': 'x'}, [], 'nope'):
        assert policy_client(SimpleNamespace(request_json=json.dumps(request))) is None
    assert policy_client(SimpleNamespace(request_json='{')) is None


def test_prompt_section_info_carries_the_client_as_strings(tmp_path, monkeypatch):
    from agent.system_prompt import _plugin_session_info
    agent = SimpleNamespace(session_id='s', model='m', provider='p', platform='tui')
    info = _plugin_session_info(agent)
    assert (info['client'], info['client_version']) == ('', '')
    with _turn_of(tmp_path, monkeypatch, client={'name': 'hermes-tern'}):
        info = _plugin_session_info(agent)
    assert (info['client'], info['client_version']) == ('hermes-tern', '')
    assert all(isinstance(value, str) for value in info.values())


@pytest.mark.platforms('any')  # the host is a child process
@pytest.mark.parametrize('isolation', ['host', 'in_process'])
def test_a_plugins_prompt_section_sees_the_client_in_either_isolation_mode(tmp_path, monkeypatch, isolation):
    """A hosted section's callable gets the session info as plain data (a read-only mapping crosses the
    host wire as a dict, not an opaque placeholder), so it reads ``client`` exactly as in-process."""
    from agent.system_prompt import _plugin_session_info
    from hermes_cli.plugins import PluginManager
    home = _home_with_plugins(tmp_path, monkeypatch, {'clientprobe': SECTION_PLUGIN}, isolation=isolation)
    manager = PluginManager()
    manager.discover_and_load()
    agent = SimpleNamespace(session_id='s', model='m', provider='p', platform='tui')
    try:
        assert manager._plugins['clientprobe'].error is None
        with _turn_of(home, monkeypatch, client=dict(TERN)):
            named = manager.render_system_prompt_sections(_plugin_session_info(agent))
        plain = manager.render_system_prompt_sections(_plugin_session_info(agent))
        assert [s.content for s in named] == ['platform=tui client=hermes-tern version=0.4.1']
        assert [s.content for s in plain] == ['platform=tui client= version=']
    finally:
        manager.unload()
        if isolation == 'host':
            manager._plugin_host().shutdown()
