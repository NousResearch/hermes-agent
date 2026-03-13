"""Tests for hermes_cli.tools_config platform tool persistence."""

from hermes_cli.tools_config import _get_platform_tools, _platform_toolset_summary


def test_get_platform_tools_uses_default_when_platform_not_configured():
    config = {}

    enabled = _get_platform_tools(config, "cli")

    assert enabled


def test_get_platform_tools_preserves_explicit_empty_selection():
    config = {"platform_toolsets": {"cli": []}}

    enabled = _get_platform_tools(config, "cli")

    assert enabled == set()


def test_platform_toolset_summary_uses_explicit_platform_list():
    config = {}

    summary = _platform_toolset_summary(config, platforms=["cli"])

    assert set(summary.keys()) == {"cli"}
    assert summary["cli"] == _get_platform_tools(config, "cli")
