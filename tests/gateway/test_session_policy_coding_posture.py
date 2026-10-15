"""The gateway, not the client, applies the coding posture to a session that did not pick toolsets."""
import pytest

FOCUS = {'agent': {'coding_context': 'focus'}, 'platform_toolsets': {'cli': ['terminal'], 'acp': ['terminal']}}


def create(params, config, **kwargs):
    """The ``session.create`` build (``stage_local_session``), the one caller that opts in."""
    from gateway.session_policy import build_policy
    return build_policy(params, config, workspace_posture=True, **kwargs)


@pytest.fixture
def project(tmp_path):
    root = tmp_path / 'project'
    root.mkdir()
    (root / 'pyproject.toml').write_text('[project]\nname = "demo"\n')
    return root


@pytest.mark.parametrize('source', ['cli', 'tui', 'gui', 'acp', 'oneshot'])
def test_focus_in_a_code_workspace_selects_the_coding_posture(project, source):
    policy = create({'source': source, 'cwd': str(project)}, FOCUS)
    assert 'coding' in policy.toolsets and 'terminal' not in policy.toolsets


def test_surface_toolsets_still_fold_in_over_the_posture(project):
    assert 'project' in create({'source': 'tui', 'cwd': str(project)}, FOCUS).toolsets
    gui = create({'source': 'gui', 'cwd': str(project)}, FOCUS).toolsets
    assert {'coding', 'desktop_ui', 'catalog'} <= set(gui)


def test_profile_default_outside_a_workspace_or_without_focus(project, tmp_path):
    plain = tmp_path / 'notes'
    plain.mkdir()
    assert create({'source': 'cli', 'cwd': str(plain)}, FOCUS).toolsets == ('terminal',)
    auto = {**FOCUS, 'agent': {'coding_context': 'auto'}}
    assert create({'source': 'cli', 'cwd': str(project)}, auto).toolsets == ('terminal',)


def test_an_explicit_pick_wins_over_the_posture(project):
    policy = create({'source': 'cli', 'cwd': str(project), 'toolsets': ['terminal']}, FOCUS)
    assert policy.toolsets == ('terminal',)


def test_a_following_session_keeps_the_posture_for_its_own_cwd(project):
    policy = create({'source': 'cli', 'cwd': str(project), 'follow_profile_config': True}, FOCUS)
    assert 'coding' in policy.current_toolsets(FOCUS)


def test_a_project_file_written_mid_session_does_not_switch_toolsets(tmp_path):
    plain = tmp_path / 'notes'
    plain.mkdir()
    policy = create({'source': 'cli', 'cwd': str(plain), 'follow_profile_config': True}, FOCUS)
    (plain / 'pyproject.toml').write_text('[project]\nname = "late"\n')
    assert policy.current_toolsets(FOCUS) == ('terminal',)


def test_a_following_session_still_takes_a_profile_change(project):
    policy = create({'source': 'cli', 'cwd': str(project), 'follow_profile_config': True}, FOCUS)
    off = {**FOCUS, 'agent': {'coding_context': 'off'}}
    assert policy.current_toolsets(off) == ('terminal',)


def test_plugin_mcp_servers_stay_in_the_posture(project, monkeypatch):
    monkeypatch.setattr('hermes_cli.plugins.get_portable_mcp_server_names_nowait', lambda: {'plug'})
    assert {'coding', 'plug'} <= set(create({'source': 'gui', 'cwd': str(project)}, FOCUS).toolsets)


@pytest.mark.parametrize('source', ['tool', 'gui'])
def test_the_daemons_own_directory_is_not_a_workspace(project, monkeypatch, source):
    monkeypatch.chdir(project)
    monkeypatch.setattr('agent.runtime_cwd.resolve_agent_cwd', lambda: project)
    assert 'coding' not in create({'source': source}, FOCUS, profile_terminal=False).toolsets


def test_a_profile_default_listing_coding_is_not_replayed_as_a_workspace(tmp_path):
    plain = tmp_path / 'notes'
    plain.mkdir()
    config = {**FOCUS, 'platform_toolsets': {'cli': ['coding', 'web']}}
    policy = create({'source': 'cli', 'cwd': str(plain), 'follow_profile_config': True}, config)
    assert not policy.coding_posture and 'web' in policy.toolsets
    assert policy.current_toolsets(config) == policy.toolsets


@pytest.mark.parametrize('params', [{'source': 'tool'}, {'source': 'cli', 'toolsets': ['terminal']}])
def test_tool_sources_and_picks_keep_their_toolsets_in_a_workspace(project, params):
    policy = create({**params, 'cwd': str(project)}, FOCUS)
    assert not policy.coding_posture and policy.toolsets == ('terminal',)


def test_only_session_create_opts_into_the_workspace_posture(project):
    # Kanban workers, legacy-row adoption, cron and hosted rooms build without opting in.
    from gateway.session_policy import build_policy
    assert build_policy({'source': 'cli', 'cwd': str(project)}, FOCUS).toolsets == ('terminal',)


@pytest.mark.asyncio
async def test_session_create_selects_the_posture_for_its_named_folder(project, tmp_path, monkeypatch):
    """The real ``session.create`` path (``create_local_session``), not a direct ``build_policy`` call."""
    from types import SimpleNamespace
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    from hermes_state_local import local_receipt
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {**FOCUS, 'model': {'default': 'm'}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False)
    try:
        authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='posture')
        actor = Principal('owner', str(tmp_path), frozenset({'session:create', 'session:read'}), 'socket')
        ref = create_local_session(authority, actor, {'request_id': 'r', 'source': 'cli', 'cwd': str(project)})
        policy = local_receipt(store._db, ref.session_id)['policy']
    finally:
        store._db.close()
    assert policy['coding_posture'] is True and policy['toolsets'] == ['coding']
