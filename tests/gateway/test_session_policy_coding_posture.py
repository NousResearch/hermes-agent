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
