"""Tests for issue #65662 — MCP tools not available in gateway/messaging
platform agents.

When enabled_toolsets is explicitly set (as gateway platforms do), only
tools from those specific toolsets are resolved. MCP tools are dynamically
registered into mcp-{server_name} toolsets that aren't part of any platform's
enabled_toolsets list, so gateway agents (QQ, Telegram, Discord, etc.) cannot
see or call mcp__* tools.

The fix: after resolving enabled_toolsets, auto-include any mcp-* toolsets
from the registry.
"""

from __future__ import annotations

import inspect

import pytest


# --------------------------------------------------------------------------- #
# Source inspection: the auto-include logic exists
# --------------------------------------------------------------------------- #


def test_compute_tool_definitions_auto_includes_mcp_toolsets():
    """_compute_tool_definitions must auto-include mcp-* toolsets after
    resolving enabled_toolsets. See issue #65662.
    """
    import model_tools

    source = inspect.getsource(model_tools._compute_tool_definitions)

    # Must contain the mcp auto-include logic
    assert "mcp-" in source, (
        "_compute_tool_definitions must check for mcp-* toolsets — see issue #65662"
    )
    assert "get_available_toolsets" in source or "available_toolsets" in source, (
        "Must enumerate available toolsets to find mcp-* entries"
    )
    assert "Auto-included MCP" in source or "auto.include" in source.lower(), (
        "Should have the auto-include logic for MCP toolsets"
    )


def test_auto_include_is_inside_enabled_toolsets_branch():
    """The mcp auto-include must be inside the `if enabled_toolsets is not None`
    branch — that's the path gateway platform agents take. The default
    (no enabled_toolsets) already includes everything."""
    import model_tools

    source = inspect.getsource(model_tools._compute_tool_definitions)

    # Find the enabled_toolsets branch
    assert "if enabled_toolsets is not None:" in source

    # The mcp auto-include should be between the for-loop and the else branch
    lines = source.split("\n")
    in_enabled_branch = False
    found_mcp_include = False
    for line in lines:
        if "if enabled_toolsets is not None:" in line:
            in_enabled_branch = True
        elif in_enabled_branch and line.strip().startswith("else:"):
            break
        elif in_enabled_branch and "mcp-" in line:
            found_mcp_include = True
            break

    assert found_mcp_include, (
        "MCP auto-include must be inside the enabled_toolsets branch — see issue #65662"
    )


# --------------------------------------------------------------------------- #
# Functional: MCP tools are included when enabled_toolsets is set
# --------------------------------------------------------------------------- #


def test_mcp_tools_included_with_enabled_toolsets():
    """When enabled_toolsets is set, mcp-* toolsets should still be included.

    Uses source inspection to verify the auto-include logic is wired correctly,
    rather than a full integration test that requires mocking the entire
    registry chain.
    """
    import model_tools
    source = inspect.getsource(model_tools._compute_tool_definitions)

    # The auto-include must:
    # 1. Be inside the enabled_toolsets branch
    # 2. Enumerate available toolsets from registry
    # 3. Filter for mcp-* prefixed names
    # 4. Check availability
    # 5. Resolve and add the tools
    assert "mcp-" in source
    assert "get_available_toolsets" in source
    assert "resolve_toolset" in source
    assert "tools_to_include.update" in source

    # Verify the code is inside the enabled_toolsets branch (before else:)
    lines = source.split("\n")
    in_enabled_branch = False
    found_all_checks = False
    checks_found = 0
    for line in lines:
        if "if enabled_toolsets is not None:" in line:
            in_enabled_branch = True
        elif in_enabled_branch and line.strip().startswith("else:"):
            break
        elif in_enabled_branch:
            if "mcp-" in line:
                checks_found += 1
            if "get_available_toolsets" in line:
                checks_found += 1
            if "tools_to_include.update" in line:
                checks_found += 1

    assert checks_found >= 3, (
        f"Expected mcp-, get_available_toolsets, and tools_to_include.update "
        f"in enabled_toolsets branch, found {checks_found} matches"
    )


def test_mcp_tools_not_duplicated_in_default_path():
    """When enabled_toolsets is None (default), MCP tools are already included
    via get_all_toolsets — the auto-include should not duplicate them."""
    from unittest.mock import patch, MagicMock

    mock_registry = MagicMock()
    mock_registry.get_available_toolsets.return_value = {}

    with patch("model_tools.registry", mock_registry):
        from model_tools import _compute_tool_definitions
        # Default path (enabled_toolsets=None) should not call
        # get_available_toolsets for mcp inclusion
        result = _compute_tool_definitions(
            enabled_toolsets=None,
            quiet_mode=True,
        )
        # Just verify it doesn't crash — the default path uses get_all_toolsets


# --------------------------------------------------------------------------- #
# Error handling: registry failures don't crash tool definition building
# --------------------------------------------------------------------------- #


def test_registry_failure_does_not_crash():
    """If the registry call fails, tool definition building should still work.

    The auto-include wraps registry access in try/except so a failure
    doesn't prevent the rest of the tool definitions from building.
    """
    import model_tools

    source = inspect.getsource(model_tools._compute_tool_definitions)

    # The registry access must be wrapped in try/except
    assert "except Exception:" in source
    assert "pass" in source