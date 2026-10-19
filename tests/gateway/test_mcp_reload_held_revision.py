"""A ``reload.mcp`` that names the revision the calling session already holds has nothing left to load:
it answers the success shape (naming the configured server that is down) instead of the named refusal, so
a client that only advances its accepted revision on ``status: reloaded`` can acknowledge it (#141769).

The refusal stays for a caller that names no revision, and for one that names a revision the session does
not hold: those still describe a reload whose work did not take effect.
"""
import json
from types import SimpleNamespace

import pytest

from tests.gateway.test_session_resume_config_refresh import _call, _connect, _owner_authority

CONFIG = ('model: {default: m1, provider: custom, base_url: "http://127.0.0.1:9/v1"}\n'
          'platform_toolsets: {cli: [terminal]}\nterminal: {backend: local}\n'
          'mcp_servers: {old: {command: old-mcp}}\n')


def _home(monkeypatch):
    from gateway import run
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / 'config.yaml').write_text(CONFIG)
    monkeypatch.setattr(run, '_hermes_home', home)
    return home


def _reloaded_profile(authority, monkeypatch):
    """reload.mcp's profile side: the owner's reconcile has nothing to add or remove, and one configured
    server is down — the state after a gateway restart whose TUI re-attached."""
    authority.runner.config = SimpleNamespace(multiplex_profiles=False)
    authority.runner._mcp_reload_refresh_cached_agents = lambda multiplex, profile: None
    monkeypatch.setattr('gateway.session_authorities.served_profile_name', lambda profile_id: 'default')
    monkeypatch.setattr('tools.mcp_tool_discovery.reconcile_mcp_servers_with_config',
                        lambda: {'removed': [], 'added': [], 'pending': []})
    monkeypatch.setattr('tools.mcp_tool_discovery.mcp_servers_not_connected', lambda: ['parked'])
    monkeypatch.setattr('tools.mcp_tool_discovery.mcp_servers_ready', lambda: ['old'])


def _refusal(reply):
    assert 'result' not in reply, reply
    return json.dumps(reply['error'])


@pytest.mark.asyncio
async def test_a_reload_naming_the_revision_the_session_holds_is_not_refused(tmp_path, monkeypatch):
    home = _home(monkeypatch)
    monkeypatch.setattr('gateway.session_policy.launch_key', lambda authority, policy: None)
    store, authority = await _owner_authority(home)
    _reloaded_profile(authority, monkeypatch)
    owner = _connect(authority)
    try:
        sid = (await _call(owner, 'session.create', request_id='a', source='cli', cwd=str(tmp_path)))['session_id']
        rev = (await _call(owner, 'config.get', key='mtime', session_id=sid))['mcp_rev']
        assert rev and rev != 'deadbeefcafe'
        # The revision the poller was told to load is the one this session already holds: the acknowledged
        # handshake must close, and the down server is named beside the success instead of replacing it.
        result = await _call(owner, 'reload.mcp', session_id=sid, confirm=True, rev=rev)
        assert result['status'] == 'reloaded' and result['unavailable'] == ['parked'], result
        assert result['loaded_rev'] == rev and result['session'] == 'unchanged', result
        assert 'did not connect: parked' in result['message'], result
        # A client that names no revision describes a reload whose own work did not take effect: refused.
        assert 'parked' in _refusal(await owner.dispatch(
            {'id': 2, 'method': 'reload.mcp', 'params': {'session_id': sid, 'confirm': True}}))
        # A revision this session does not hold is still a reload that has work left to do: refused.
        assert 'parked' in _refusal(await owner.dispatch(
            {'id': 3, 'method': 'reload.mcp',
             'params': {'session_id': sid, 'confirm': True, 'rev': 'deadbeefcafe'}}))
    finally:
        await owner.close()
        store.close_all_db_handles()
