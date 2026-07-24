"""Regression test for _pending_native_image_paths_by_session conversation scope clearing."""

from unittest.mock import MagicMock
from gateway.run import GatewayRunner, _CONVERSATION_SCOPED_STATE


def test_conversation_scoped_state_contains_pending_native_image_paths():
    """Verify _pending_native_image_paths_by_session is registered in _CONVERSATION_SCOPED_STATE."""
    assert "_pending_native_image_paths_by_session" in _CONVERSATION_SCOPED_STATE


def test_clear_conversation_scope_clears_pending_native_images():
    """Verify _clear_conversation_scope purges staged native image paths for session_key."""
    runner = object.__new__(GatewayRunner)
    session_key = "test_session_key_123"
    runner._pending_native_image_paths_by_session = {
        session_key: ["/tmp/staged_image1.png", "/tmp/staged_image2.jpg"],
        "other_session": ["/tmp/other.png"],
    }

    # Mock security boundary helper call inside _clear_conversation_scope
    runner._clear_session_boundary_security_state = MagicMock()

    runner._clear_conversation_scope(session_key, reason="test_reset")

    assert session_key not in runner._pending_native_image_paths_by_session
    assert "other_session" in runner._pending_native_image_paths_by_session
    assert runner._pending_native_image_paths_by_session["other_session"] == ["/tmp/other.png"]
