"""The welcome banner's "Available Skills" gate must agree with the session's tool surface.

Regression for #136201: the CLI seeds a session with a composite toolset name (``hermes-cli``)
whose resolved tools include ``skill_view`` / ``skills_list`` / ``skill_manage``, yet the banner
tested for the literal name ``skills`` and reported "Skills toolset disabled" / "0 skills".
"""

import os
from unittest.mock import patch

import pytest
from rich.console import Console

from hermes_cli import banner
import model_tools
import toolsets
import tools.mcp_tool_discovery

_SKILLS = {"research": ["arxiv", "deep-dive"], "ops": ["docker"]}


def _render(enabled_toolsets):
    with (
        patch.object(model_tools, "check_tool_availability", return_value=([], [])),
        patch.object(banner, "get_available_skills", return_value=_SKILLS),
        patch.object(banner, "get_update_result", return_value=None),
        patch.object(banner, "get_latest_release_tag", return_value=None),
        patch.object(tools.mcp_tool_discovery, "get_mcp_status", return_value=[]),
        patch("shutil.get_terminal_size", return_value=os.terminal_size((160, 50))),
    ):
        console = Console(record=True, force_terminal=False, color_system=None, width=160)
        banner.build_welcome_banner(
            console=console, model="m", cwd="/tmp/project", tools=[],
            enabled_toolsets=enabled_toolsets, get_toolset_for_tool=lambda _: None)
        return console.export_text()


def _carries_skill_tools(toolset_name: str) -> bool:
    return bool(set(toolsets.resolve_toolset(toolset_name)) & set(toolsets.resolve_toolset("skills")))


def test_composite_cli_bundle_lists_skills_and_counts_them():
    """``hermes-cli`` is the CLI's default seed and resolves to the skill tools, so the
    catalog must be listed and counted, not reported as disabled."""
    assert _carries_skill_tools("hermes-cli")
    text = _render(["hermes-cli"])
    assert "Skills toolset disabled" not in text
    assert "arxiv" in text and "docker" in text
    assert f"{sum(len(v) for v in _SKILLS.values())} skills" in text


def test_blank_slate_still_reports_skills_toolset_disabled():
    """The original intent (#50497) holds: a session without any skill tool must not advertise
    the on-disk catalog."""
    assert not _carries_skill_tools("file") and not _carries_skill_tools("terminal")
    text = _render(["file", "terminal"])
    assert "Skills toolset disabled" in text
    assert "arxiv" not in text
    assert "0 skills" in text


@pytest.mark.parametrize("toolset_name", sorted(toolsets.get_toolset_names()))
def test_banner_skills_gate_matches_resolved_tool_surface(toolset_name):
    """For every static toolset, the banner shows the catalog iff the resolved tool list reaches a
    skill tool. Composite bundles (``hermes-*``, ``coding``) and the bare ``skills`` toolset agree."""
    text = _render([toolset_name])
    assert ("Skills toolset disabled" not in text) == _carries_skill_tools(toolset_name), toolset_name
