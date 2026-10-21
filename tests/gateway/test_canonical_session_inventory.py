"""A canonical session's snapshot carries main's banner/status inventory for its frozen policy.

Main's ``tui_gateway/server.py::_session_info`` put tools (by toolset), skills (by category), MCP
servers, version / release date, git branch and provider on every create / resume / ``session.info``
``info``; the Ink banner and the classic CLI banner render exactly those. A lazy canonical session
(agent built on the first turn) must report what that turn's agent WILL load: the toolsets frozen
at creation, never the owner process's whole registry."""
import subprocess
from types import SimpleNamespace

import pytest


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / 'repo'
    path.mkdir()
    subprocess.run(['git', 'init', '-q', '-b', 'inventory-branch', str(path)], check=True, timeout=30)
    return path


@pytest.mark.asyncio
async def test_created_session_snapshot_carries_inventory_for_frozen_toolsets(tmp_path, monkeypatch, repo):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from hermes_cli import banner
    from tui_gateway.contracts.registry import CANONICAL_METHODS

    home = tmp_path / 'home'
    skill = home / 'skills' / 'research' / 'arxiv'
    skill.mkdir(parents=True)
    (skill / 'SKILL.md').write_text('---\nname: arxiv\ndescription: Search papers.\n---\nBody.\n')
    (home / 'config.yaml').write_text(
        'model: {default: gpt-5.4, provider: custom, base_url: "http://127.0.0.1:9/v1"}\n'
        'platform_toolsets: {cli: [file, terminal]}\n')
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(run, '_hermes_home', home)
    monkeypatch.setattr(banner, '_available_skills_cache', None)
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id=str(home), instance_id='first')
    connection = AuthorityConnection(authority, object(), {'user_id': 'human'})

    async def call(method, **params):
        reply = await connection.dispatch({'id': 1, 'method': method, 'params': params})
        assert 'result' in reply, reply
        return reply['result']

    created = await call('session.create', request_id='inv', source='tui', cwd=str(repo))
    sid = created['session_id']
    CANONICAL_METHODS['session.create'].result.model_validate(created)
    info_reply = await call('session.info', session_id=sid)
    CANONICAL_METHODS['session.info'].result.model_validate(info_reply)
    for info in (created['info'], info_reply, (await call('session.resume', session_id=sid))['info']):
        assert info['lazy'] is True
        tools = info['tools']
        # The frozen policy's toolsets, grouped by toolset as main grouped ``agent.tools`` (deferrable
        # tools sit behind the tool_search bridge in both, under ``other``).
        assert 'read_file' in tools.get('file', []), tools
        assert tools.get('terminal') == ['terminal'], tools
        assert not ({'web', 'browser', 'image_gen'} & set(tools)), tools
        assert info['skills'].get('research') == ['arxiv'], info['skills']
        assert info['mcp_servers'] == []
        assert info['branch'] == 'inventory-branch'
        assert info['provider'] == 'custom'
        assert info['version'] and info['release_date'], info
        assert sorted(info['enabled_toolsets']) == info['enabled_toolsets']
        assert {'file', 'terminal'} <= set(info['enabled_toolsets'])

    narrowed = await call('session.create', request_id='narrow', source='cli', cwd=str(repo), toolsets=['terminal'])
    assert set(narrowed['info']['tools']) - {'other'} == {'terminal'}, narrowed['info']['tools']


@pytest.mark.asyncio
async def test_cli_surface_snapshot_carries_main_cli_toolsets_and_startup_warnings(tmp_path, monkeypatch, repo):
    """``source: cli`` is the classic CLI surface: exactly main's ``_get_platform_tools(cfg, 'cli')`` and
    the warnings main's HermesCLI printed under its banner, computed in the process that runs the tools."""
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from hermes_cli import banner
    from hermes_cli.tools_config import _get_platform_tools
    from tools import browser_use_cli

    home = tmp_path / 'home'
    home.mkdir()
    (home / 'config.yaml').write_text(
        'model: {default: gpt-5.4, provider: custom, base_url: "http://127.0.0.1:9/v1"}\n')
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(run, '_hermes_home', home)
    monkeypatch.setattr(banner, '_available_skills_cache', None)
    monkeypatch.setattr(browser_use_cli, 'default_downgrade_notice', lambda: 'browser-harness is missing')
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id=str(home), instance_id='first')
    connection = AuthorityConnection(authority, object(), {'user_id': 'human'})

    async def call(method, **params):
        reply = await connection.dispatch({'id': 1, 'method': method, 'params': params})
        assert 'result' in reply, reply
        return reply['result']

    from hermes_cli.config import load_config
    cli = (await call('session.create', request_id='cli', source='cli', cwd=str(repo)))['info']
    assert cli['enabled_toolsets'] == sorted(_get_platform_tools(load_config(), 'cli'))
    assert isinstance(cli['tool_availability_warnings'], list)
    assert cli['browser_backend_notice'] == 'browser-harness is missing'
    tui = (await call('session.create', request_id='tui', source='tui', cwd=str(repo)))['info']
    assert 'tool_availability_warnings' not in tui and 'browser_backend_notice' not in tui


@pytest.mark.asyncio
async def test_cli_surface_counts_kanban_skills_like_main_cli_without_leaking(tmp_path, monkeypatch, repo):
    """Main's ``cmd_chat`` pins ``HERMES_KANBAN_BOARD`` before HermesCLI starts, so its banner lists
    ``environments: [kanban]`` skills. A CLI-surface snapshot counts them; other surfaces and the
    process-wide catalog do not."""
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from hermes_cli import banner
    from tools.skills_tool import find_all_skills

    home = tmp_path / 'home'
    for name, extra in (('arxiv', ''), ('sdlc-review', 'environments:\n  - kanban\n')):
        skill = home / 'skills' / 'devops' / name
        skill.mkdir(parents=True)
        (skill / 'SKILL.md').write_text(f'---\nname: {name}\ndescription: d.\n{extra}---\nBody.\n')
    (home / 'config.yaml').write_text(
        'model: {default: gpt-5.4, provider: custom, base_url: "http://127.0.0.1:9/v1"}\n')
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.delenv('HERMES_KANBAN_BOARD', raising=False)
    monkeypatch.delenv('HERMES_KANBAN_TASK', raising=False)
    monkeypatch.setattr(run, '_hermes_home', home)
    monkeypatch.setattr(banner, '_available_skills_cache', None)
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id=str(home), instance_id='first')
    connection = AuthorityConnection(authority, object(), {'user_id': 'human'})

    async def call(method, **params):
        reply = await connection.dispatch({'id': 1, 'method': method, 'params': params})
        assert 'result' in reply, reply
        return reply['result']

    cli = (await call('session.create', request_id='cli', source='cli', cwd=str(repo)))['info']
    assert sorted(cli['skills']['devops']) == ['arxiv', 'sdlc-review'], cli['skills']
    tui = (await call('session.create', request_id='tui', source='tui', cwd=str(repo)))['info']
    assert tui['skills']['devops'] == ['arxiv'], tui['skills']
    assert [s['name'] for s in find_all_skills()] == ['arxiv']
