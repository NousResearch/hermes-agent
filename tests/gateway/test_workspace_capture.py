"""Integration test: gateway captures git_repo_root at session creation."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_record_gateway_session_peer_captures_git_repo_root(tmp_path):
    """When _record_gateway_session_peer is called, it should resolve and persist git_repo_root."""
    from gateway.session import SessionStore
    
    # Create a mock DB that tracks calls to record_gateway_session_peer
    mock_db = MagicMock()
    mock_db.record_gateway_session_peer = MagicMock()
    
    # Create session store with mocked DB (pass sessions_dir + config)
    from gateway.config import GatewayConfig
    config = GatewayConfig()
    store = SessionStore(tmp_path, config)
    store._db = mock_db
    
    # Mock the source
    from gateway.config import Platform
    source = MagicMock()
    source.platform = Platform.TELEGRAM
    source.user_id = "user123"
    source.chat_id = "chat456"
    source.chat_name = "test-chat"
    source.to_dict.return_value = {"platform": "telegram"}
    
    # Create a temp dir with .git marker for git_probe to resolve
    repo_dir = tmp_path / "test-repo"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()  # Simple .git dir marker
    
    original_cwd = os.getcwd()
    try:
        os.chdir(repo_dir)
        
        # Mock git_probe to return our temp dir as the repo root
        with patch("tui_gateway.git_probe.repo_root", return_value=str(repo_dir)):
            store._record_gateway_session_peer(
                session_id="test-session-id",
                session_key="agent:test-key",
                source=source,
            )
        
        # Verify DB was called with git_repo_root parameter
        mock_db.record_gateway_session_peer.assert_called_once()
        call_kwargs = mock_db.record_gateway_session_peer.call_args[1]
        
        assert "git_repo_root" in call_kwargs, \
            "git_repo_root should be passed to DB recorder"
        assert call_kwargs["git_repo_root"] == str(repo_dir), \
            f"Expected {repo_dir}, got {call_kwargs['git_repo_root']}"
    finally:
        os.chdir(original_cwd)


def test_record_gateway_session_peer_handles_git_probe_failure(tmp_path):
    """When git_probe fails, _record_gateway_session_peer should still work (non-critical)."""
    from gateway.session import SessionStore
    
    mock_db = MagicMock()
    mock_db.record_gateway_session_peer = MagicMock()
    
    from gateway.config import GatewayConfig
    config = GatewayConfig()
    store = SessionStore(tmp_path, config)
    store._db = mock_db
    
    from gateway.config import Platform
    source = MagicMock()
    source.platform = Platform.TELEGRAM
    source.user_id = "user123"
    source.chat_id = "chat456"
    source.chat_name = "test-chat"
    source.to_dict.return_value = {"platform": "telegram"}
    
    # Patch git_probe to raise an exception
    with patch("tui_gateway.git_probe.repo_root", side_effect=OSError("probe failed")):
        store._record_gateway_session_peer(
            session_id="test-session-id",
            session_key="agent:test-key",
            source=source,
        )
        
        # Should still call DB recorder (with empty git_repo_root)
        mock_db.record_gateway_session_peer.assert_called_once()
        call_kwargs = mock_db.record_gateway_session_peer.call_args[1]
        
        assert "git_repo_root" in call_kwargs
        assert call_kwargs["git_repo_root"] == "", \
            "git_probe failure should result in empty git_repo_root, not crash"
