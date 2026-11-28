"""``reload.all`` on the shared owner (``/reload``, #159205): the classic CLI (rewired) and Ink send ``/reload``
to the gateway that runs the session. Consent first; refused while the session's turn runs; otherwise the
owner's own verbs in the ``/reload`` order: ``.env`` → plugins → MCP servers (its reconcile, then the cached
agents' tools[] rebuild that publishes the rediscovered plugin tools) → skills, with one model note waiting
in the session's one-shot slot."""
import pytest

import hermes_yaml as yaml
from tests.gateway.test_mcp_reload_new_server import _home, _reloadable
from tests.gateway.test_session_resume_config_refresh import _call, _connect, _owner_authority


@pytest.fixture
def steps(monkeypatch):
    order = []

    def record(name, result):
        return lambda *a, **k: order.append(name) or result
    monkeypatch.setattr('hermes_cli.config.reload_env', record('env', 2))
    monkeypatch.setattr('hermes_cli.plugins_activation.rediscover_plugins',
                        record('plugins', {'loaded': ['probe'], 'changed': ['probe']}))
    monkeypatch.setattr('agent.skill_commands.reload_skills',
                        record('skills', {'added': [{'name': 'fresh', 'description': 'd'}], 'removed': [], 'total': 3}))
    return order


async def _owner(tmp_path, monkeypatch, order):
    home = _home(monkeypatch)
    monkeypatch.setattr('gateway.session_policy.launch_key', lambda authority, policy: None)
    store, authority = await _owner_authority(home)
    _reloadable(authority, monkeypatch)
    real_reconcile = __import__('tools.mcp_tool_discovery', fromlist=['x']).reconcile_mcp_servers_with_config
    monkeypatch.setattr('tools.mcp_tool_discovery.reconcile_mcp_servers_with_config',
                        lambda: order.append('mcp') or real_reconcile())
    authority.runner._mcp_reload_refresh_cached_agents = lambda multiplex, profile: order.append('publish')
    owner = _connect(authority)
    sid = (await _call(owner, 'session.create', request_id='r', source='cli', cwd=str(tmp_path)))['session_id']
    return home, store, authority, owner, sid


@pytest.mark.asyncio
async def test_reload_all_asks_then_reloads_every_step_in_order(tmp_path, monkeypatch, steps):
    home, store, authority, owner, sid = await _owner(tmp_path, monkeypatch, steps)
    try:
        asked = await _call(owner, 'reload.all', session_id=sid)
        assert asked['status'] == 'confirm_required' and '/reload now' in asked['message'], asked
        assert steps == []
        result = await _call(owner, 'reload.all', session_id=sid, confirm=True)
        assert result['status'] == 'reloaded', result
        assert steps == ['env', 'plugins', 'mcp', 'publish', 'skills']
        report = result['report']
        assert report['env_updated'] == 2 and report['plugins']['changed'] == ['probe']
        assert 'changed: probe' in result['output'] and 'added: fresh' in result['output']
        note = authority.runner._pending_skills_reload_notes[authority.sessions[sid].route]
        assert 'fresh' in note and 'probe' in note
        assert 'mcp_reload_confirm' not in (home / 'config.yaml').read_text()
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_reload_all_always_turns_the_gate_off(tmp_path, monkeypatch, steps):
    home, store, _authority, owner, sid = await _owner(tmp_path, monkeypatch, steps)
    try:
        assert (await _call(owner, 'reload.all', session_id=sid, confirm=True, always=True))['status'] == 'reloaded'
        assert yaml.safe_load((home / 'config.yaml').read_text())['approvals']['mcp_reload_confirm'] is False
        steps.clear()
        assert (await _call(owner, 'reload.all', session_id=sid))['status'] == 'reloaded'  # no question now
        assert steps[0] == 'env'
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_reload_all_is_refused_while_the_session_runs_a_turn(tmp_path, monkeypatch, steps):
    _home_dir, store, authority, owner, sid = await _owner(tmp_path, monkeypatch, steps)
    try:
        authority.runner._running_agents = {authority.sessions[sid].route: object()}
        result = await _call(owner, 'reload.all', session_id=sid, confirm=True)
        assert result['status'] == 'busy' and 'nothing was reloaded' in result['message']
        assert steps == []
    finally:
        await owner.close()
        store.close_all_db_handles()
