"""config.yaml -> enabled toolsets -> tool schemas for the read-only skills split
(``skills_read`` / ``skills_manage``), through the real resolver and registry."""

import model_tools
from hermes_cli.tools_config import _get_platform_tools


def _tool_names(enabled, disabled=None):
    defs = model_tools.get_tool_definitions(
        enabled_toolsets=sorted(enabled), disabled_toolsets=disabled, quiet_mode=True)
    return {d["function"]["name"] for d in defs}


def test_platform_list_with_skills_read_exposes_no_skill_management():
    config = {"platform_toolsets": {"api_server": ["skills_read", "web"]}}
    enabled = _get_platform_tools(config, "api_server")

    assert "skills_read" in enabled and "skills" not in enabled
    names = _tool_names(enabled)
    assert {"skills_list", "skill_view"} <= names
    assert "skill_manage" not in names


def test_disabling_skills_manage_keeps_skills_readable():
    config = {"platform_toolsets": {"api_server": ["hermes-api-server"]},
              "agent": {"disabled_toolsets": ["skills_manage"]}}
    enabled = _get_platform_tools(config, "api_server")

    assert "skills" in enabled
    names = _tool_names(enabled, disabled=["skills_manage"])
    assert {"skills_list", "skill_view"} <= names
    assert "skill_manage" not in names


def test_default_skills_toolset_is_unchanged():
    names = _tool_names({"skills"})
    assert {"skills_list", "skill_view", "skill_manage"} <= names
