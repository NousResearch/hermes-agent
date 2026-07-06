"""Integration test: gateway /resume workspace guard blocks cross-workspace restores."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_gateway_resume_blocks_cross_workspace():
    """When resuming a session from a different workspace, the guard should block it."""
    # This tests the validate_session_workspace logic used in slash_commands.py resume handler
    
    from hermes_cli.workspace_guard import validate_session_workspace
    
    # Create two fake sessions with different git_repo_root values
    stored_session = {
        "id": "session-1",
        "git_repo_root": "/path/to/other-repo",
    }
    
    current_cwd = "/path/to/current-repo"
    
    result = validate_session_workspace(
        session_row=stored_session,
        current_cwd=current_cwd,
    )
    
    assert not result.ok, "Cross-workspace resume should be blocked"
    assert result.reason == "workspace_mismatch", \
        f"Expected workspace_mismatch, got {result.reason}"


@pytest.mark.asyncio
async def test_gateway_resume_allows_same_workspace():
    """When resuming a session from the same workspace, it should be allowed."""
    from hermes_cli.workspace_guard import validate_session_workspace
    
    stored_session = {
        "id": "session-1",
        "git_repo_root": "/path/to/current-repo",
    }
    
    current_cwd = "/path/to/current-repo"
    
    result = validate_session_workspace(
        session_row=stored_session,
        current_cwd=current_cwd,
    )
    
    assert result.ok, "Same-workspace resume should be allowed"


@pytest.mark.asyncio
async def test_gateway_resume_warns_on_legacy_session():
    """When resuming a legacy session (null git_repo_root), it should warn but allow."""
    from hermes_cli.workspace_guard import validate_session_workspace
    
    stored_session = {
        "id": "session-1",
        "git_repo_root": None,  # Legacy session with no workspace identity
    }
    
    current_cwd = "/path/to/current-repo"
    
    result = validate_session_workspace(
        session_row=stored_session,
        current_cwd=current_cwd,
    )
    
    assert result.ok, "Legacy sessions should be allowed through"
    assert result.warning is not None, \
        "Legacy sessions should produce a warning"
