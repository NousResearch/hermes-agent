"""F1 role toolset enforcement — fixture tests against the frozen spec.

Frozen contract: profiles/smokey/workspace/f1-expected-inventory-spec-v1.md
Covers (offline, fixture-only, Gate 2/3 phase):
  I1  pinned direct inventory resolves to exactly the allowlisted toolsets
  I3  project/desktop_ui never appear unless the pin lists them
  I4  unknown toolset name in the pin fails CLOSED (no tools, no fallback)
  D8  agent.disabled_toolsets subtracts AFTER allowlist expansion
  plus: read-only bundles exist and resolve to read-only tool sets.
"""

import logging

import pytest

from hermes_cli.tools_config import _get_platform_tools
from toolsets import resolve_toolset


# ─── read-only bundles ────────────────────────────────────────────────────────

def test_file_readonly_bundle_resolves_to_read_only_tools():
    assert set(resolve_toolset("file_readonly")) == {"read_file", "search_files"}


def test_skills_readonly_bundle_resolves_to_read_only_tools():
    assert set(resolve_toolset("skills_readonly")) == {"skills_list", "skill_view"}


# ─── I1: pin is authoritative ─────────────────────────────────────────────────

def test_profile_pin_is_authoritative_allowlist():
    config = {
        "tools": {"enabled_toolsets": ["file_readonly", "skills_readonly", "web"]},
        "platform_toolsets": {"cli": ["hermes-cli"]},  # ignored when pinned
    }
    enabled = _get_platform_tools(config, "cli", include_default_mcp_servers=False)
    assert enabled == {"file_readonly", "skills_readonly", "web"}


def test_unpinned_profile_keeps_platform_toolsets_behavior():
    config = {"platform_toolsets": {"cli": ["hermes-cli"]}}
    enabled = _get_platform_tools(config, "cli", include_default_mcp_servers=False)
    default_enabled = _get_platform_tools({}, "cli", include_default_mcp_servers=False)
    assert enabled == default_enabled


def test_pinned_profile_applies_across_platforms():
    """One pin governs every platform key (CLI, cron, telegram, ...)."""
    config = {"tools": {"enabled_toolsets": ["web", "memory"]}}
    for platform in ("cli", "cron", "telegram", "desktop"):
        assert _get_platform_tools(config, platform, include_default_mcp_servers=False) == {
            "web",
            "memory",
        }


# ─── I3: GUI extras never injected implicitly ─────────────────────────────────

def test_pinned_profile_does_not_gain_project_or_desktop_ui():
    config = {"tools": {"enabled_toolsets": ["web", "memory"]}}
    enabled = _get_platform_tools(config, "desktop", include_default_mcp_servers=False)
    assert "project" not in enabled
    assert "desktop_ui" not in enabled


def test_pinned_profile_can_explicitly_include_gui_toolsets():
    config = {"tools": {"enabled_toolsets": ["web", "project", "desktop_ui"]}}
    enabled = _get_platform_tools(config, "desktop", include_default_mcp_servers=False)
    assert {"project", "desktop_ui"} <= enabled


# ─── I4: fail closed ──────────────────────────────────────────────────────────

def test_unknown_toolset_name_fails_closed(caplog):
    config = {"tools": {"enabled_toolsets": ["web", "bogus_toolset_xyz"]}}
    with caplog.at_level(logging.ERROR, logger="hermes_cli.tools_config"):
        enabled = _get_platform_tools(config, "cli", include_default_mcp_servers=False)
    assert enabled == set()
    assert any("bogus_toolset_xyz" in r.getMessage() for r in caplog.records)


def test_unknown_mcp_name_fails_closed():
    config = {"tools": {"enabled_toolsets": ["web", "not-a-mcp-server:tool"]}}
    assert _get_platform_tools(config, "cli", include_default_mcp_servers=True) == set()


def test_empty_pin_list_disables_everything():
    """An explicit empty list means 'no tools' — not 'unpinned'."""
    config = {"tools": {"enabled_toolsets": []}}
    assert _get_platform_tools(config, "cli", include_default_mcp_servers=False) == set()


# ─── D8: disabled_toolsets subtracts after expansion ──────────────────────────

def test_disabled_toolsets_subtract_after_allowlist():
    config = {
        "tools": {"enabled_toolsets": ["web", "terminal", "file"]},
        "agent": {"disabled_toolsets": ["terminal"]},
    }
    enabled = _get_platform_tools(config, "cli", include_default_mcp_servers=False)
    assert "terminal" not in enabled
    assert "web" in enabled
    assert "file" in enabled


def test_smokey_f7_canary_cron_denial_survives_pin():
    """Canary: the F7 structural cron denial must hold under a role pin."""
    config = {
        "tools": {"enabled_toolsets": ["web", "file_readonly", "cronjob"]},
        "agent": {"disabled_toolsets": ["cronjob"]},
    }
    enabled = _get_platform_tools(config, "cli", include_default_mcp_servers=False)
    assert "cronjob" not in enabled


# ─── MCP layering ─────────────────────────────────────────────────────────────

def test_pinned_profile_includes_enabled_mcp_servers():
    config = {
        "tools": {"enabled_toolsets": ["web"]},
        "mcp_servers": {"my_server": {"command": "foo", "enabled": True}},
    }
    enabled = _get_platform_tools(config, "cli", include_default_mcp_servers=True)
    assert "my_server" in enabled


def test_pinned_profile_no_mcp_sentinel():
    config = {
        "tools": {"enabled_toolsets": ["web", "no_mcp"]},
        "mcp_servers": {"my_server": {"command": "foo", "enabled": True}},
    }
    enabled = _get_platform_tools(config, "cli", include_default_mcp_servers=True)
    assert "my_server" not in enabled
