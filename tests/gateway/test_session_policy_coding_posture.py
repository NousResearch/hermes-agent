"""The gateway, not the client, applies the coding posture to a session that did not pick toolsets."""
import pytest

FOCUS = {'agent': {'coding_context': 'focus'}, 'platform_toolsets': {'cli': ['terminal'], 'acp': ['terminal']}}


@pytest.fixture
def project(tmp_path):
    root = tmp_path / 'project'
    root.mkdir()
    (root / 'pyproject.toml').write_text('[project]\nname = "demo"\n')
    return root


@pytest.mark.parametrize('source', ['cli', 'tui', 'gui', 'acp', 'oneshot'])
def test_focus_in_a_code_workspace_selects_the_coding_posture(project, source):
    from gateway.session_policy import build_policy
    policy = build_policy({'source': source, 'cwd': str(project)}, FOCUS)
    assert 'coding' in policy.toolsets and 'terminal' not in policy.toolsets


def test_surface_toolsets_still_fold_in_over_the_posture(project):
    from gateway.session_policy import build_policy
    assert 'project' in build_policy({'source': 'tui', 'cwd': str(project)}, FOCUS).toolsets
    gui = build_policy({'source': 'gui', 'cwd': str(project)}, FOCUS).toolsets
    assert {'coding', 'desktop_ui', 'catalog'} <= set(gui)


def test_profile_default_outside_a_workspace_or_without_focus(project, tmp_path):
    from gateway.session_policy import build_policy
    plain = tmp_path / 'notes'
    plain.mkdir()
    assert build_policy({'source': 'cli', 'cwd': str(plain)}, FOCUS).toolsets == ('terminal',)
    auto = {**FOCUS, 'agent': {'coding_context': 'auto'}}
    assert build_policy({'source': 'cli', 'cwd': str(project)}, auto).toolsets == ('terminal',)


def test_an_explicit_pick_wins_over_the_posture(project):
    from gateway.session_policy import build_policy
    policy = build_policy({'source': 'cli', 'cwd': str(project), 'toolsets': ['terminal']}, FOCUS)
    assert policy.toolsets == ('terminal',)


def test_a_following_session_keeps_the_posture_for_its_own_cwd(project):
    from gateway.session_policy import build_policy
    policy = build_policy({'source': 'cli', 'cwd': str(project), 'follow_profile_config': True}, FOCUS)
    assert 'coding' in policy.current_toolsets(FOCUS)


def test_a_project_file_written_mid_session_does_not_switch_toolsets(tmp_path):
    from gateway.session_policy import build_policy
    plain = tmp_path / 'notes'
    plain.mkdir()
    policy = build_policy({'source': 'cli', 'cwd': str(plain), 'follow_profile_config': True}, FOCUS)
    (plain / 'pyproject.toml').write_text('[project]\nname = "late"\n')
    assert policy.current_toolsets(FOCUS) == ('terminal',)


def test_a_following_session_still_takes_a_profile_change(project):
    from gateway.session_policy import build_policy
    policy = build_policy({'source': 'cli', 'cwd': str(project), 'follow_profile_config': True}, FOCUS)
    off = {**FOCUS, 'agent': {'coding_context': 'off'}}
    assert policy.current_toolsets(off) == ('terminal',)


def test_plugin_mcp_servers_stay_in_the_posture(project, monkeypatch):
    from gateway.session_policy import build_policy
    monkeypatch.setattr('hermes_cli.plugins.get_portable_mcp_server_names_nowait', lambda: {'plug'})
    assert {'coding', 'plug'} <= set(build_policy({'source': 'gui', 'cwd': str(project)}, FOCUS).toolsets)


@pytest.mark.parametrize('source', ['tool', 'gui'])
def test_the_daemons_own_directory_is_not_a_workspace(project, monkeypatch, source):
    from gateway.session_policy import build_policy
    monkeypatch.chdir(project)
    monkeypatch.setattr('agent.runtime_cwd.resolve_agent_cwd', lambda: project)
    assert 'coding' not in build_policy({'source': source}, FOCUS, profile_terminal=False).toolsets
