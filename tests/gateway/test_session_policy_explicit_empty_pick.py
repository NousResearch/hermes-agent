"""An explicit empty toolsets pick is a zero-tool session, not the profile default.

`-t none` reaches the gateway as ``toolsets: []``. ``_get_platform_tools`` reads an empty value as
"no opinion" and fills it back in from the profile's plugin and MCP toolsets, so the one launch
that must be inert came back with the widest surface on the machine — the reason career-ops trusts
no other CLI's zero-tool mode and not this one.
"""

import pytest

from gateway.session_policy import build_policy

CONFIG = {'platform_toolsets': {'cli': ['file', 'terminal'], 'acp': ['terminal']}}


@pytest.fixture
def cwd(tmp_path):
    root = tmp_path / 'work'
    root.mkdir()
    return str(root)


def _policy(cwd, **params):
    return build_policy({'source': 'cli', 'cwd': cwd, **params}, CONFIG, workspace_posture=False)


def test_an_explicit_empty_pick_selects_no_tools(cwd):
    assert _policy(cwd, toolsets=[]).toolsets == ()


def test_a_named_pick_still_narrows_to_its_toolsets(cwd):
    toolsets = _policy(cwd, toolsets=['file']).toolsets
    assert 'file' in toolsets and 'terminal' not in toolsets


def test_an_absent_pick_still_resolves_the_profile_default(cwd):
    assert set(_policy(cwd).toolsets) == {'file', 'terminal'}
