"""Cron per-job ``enabled_toolsets`` must include plugin-registered toolsets, mirroring the
existing MCP-server merge.  A job created with ``[\"terminal\", \"file\"]`` on a profile that has an
enabled plugin must also see that plugin's toolset in the resolved allowlist.

Fixes: https://github.com/NousResearch/hermes-agent/issues/134311
Related: _merge_mcp_into_per_job_toolsets in cron/scheduler.py
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from cron.scheduler import _merge_mcp_into_per_job_toolsets, _resolve_cron_enabled_toolsets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _register_plugin_toolset(name: str, tool_name: str):
    """Register a plugin toolset with one tool and return a cleanup callable."""
    from tools.registry import registry

    registry.register(
        name=tool_name,
        toolset=name,
        schema={"name": tool_name, "description": "test", "parameters": {"type": "object", "properties": {}}},
        handler=lambda a, **k: "{}",
    )
    return lambda: registry.deregister(tool_name)


# ---------------------------------------------------------------------------
# _merge_mcp_into_per_job_toolsets — plugin merging
# ---------------------------------------------------------------------------

def test_plugin_toolset_merged_when_not_in_per_job_list():
    """A plugin toolset absent from the per-job list is unioned in (mirrors MCP merge)."""
    undo = _register_plugin_toolset("my-plugin", "my_plugin__search")
    try:
        result = _merge_mcp_into_per_job_toolsets(
            ["terminal", "file"],
            {},  # no MCP servers configured
        )
    finally:
        undo()

    assert "my-plugin" in result
    assert "terminal" in result
    assert "file" in result


def test_plugin_toolset_not_duplicated_when_already_listed():
    """If the plugin's toolset is already in the per-job list, it is not added again."""
    undo = _register_plugin_toolset("my-plugin", "my_plugin__search")
    try:
        result = _merge_mcp_into_per_job_toolsets(["terminal", "my-plugin"], {})
    finally:
        undo()

    assert result.count("my-plugin") == 1


def test_no_plugins_sentinel_suppresses_plugin_merge():
    """``no_plugins`` in the per-job list suppresses plugin toolset merging and is stripped."""
    undo = _register_plugin_toolset("my-plugin", "my_plugin__search")
    try:
        result = _merge_mcp_into_per_job_toolsets(["terminal", "no_plugins"], {})
    finally:
        undo()

    assert "my-plugin" not in result
    assert "no_plugins" not in result
    assert "terminal" in result


def test_no_mcp_sentinel_still_works_alongside_plugin_merge():
    """``no_mcp`` suppresses only MCP merging; plugin toolsets are still merged."""
    undo = _register_plugin_toolset("my-plugin", "my_plugin__search")
    try:
        with patch("hermes_cli.tools_config.enabled_mcp_server_names", return_value={"notion"}):
            result = _merge_mcp_into_per_job_toolsets(["terminal", "no_mcp"], {})
    finally:
        undo()

    assert "notion" not in result
    assert "no_mcp" not in result
    assert "my-plugin" in result


def test_explicit_empty_per_job_list_unaffected_by_plugin_merge():
    """An explicit empty allowlist ([]) is handled by _resolve_cron_enabled_toolsets before the
    merge function is called, so it stays empty — plugin merge never fires."""
    assert _resolve_cron_enabled_toolsets({"enabled_toolsets": []}, {}) == []


def test_plugin_merge_failure_is_non_fatal():
    """If _get_plugin_toolset_names raises, the merge falls back gracefully (no crash)."""
    with patch("toolsets._get_plugin_toolset_names", side_effect=RuntimeError("boom")):
        result = _merge_mcp_into_per_job_toolsets(["terminal"], {})

    assert "terminal" in result  # the base list is preserved


# ---------------------------------------------------------------------------
# _resolve_cron_enabled_toolsets — integration with plugin merge
# ---------------------------------------------------------------------------

def test_resolve_per_job_list_includes_plugin_toolset():
    """End-to-end: _resolve_cron_enabled_toolsets with a per-job list propagates plugin merge."""
    undo = _register_plugin_toolset("my-plugin", "my_plugin__search")
    try:
        resolved = _resolve_cron_enabled_toolsets(
            {"enabled_toolsets": ["terminal", "file"]},
            {},
        )
    finally:
        undo()

    assert "my-plugin" in resolved
    assert "terminal" in resolved
