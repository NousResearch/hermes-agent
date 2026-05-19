"""Tests for OpenVikingMemoryProvider.on_session_switch.

Covers #28296: after /new, /branch, or context compression the provider's
cached ``_session_id`` must be updated so subsequent sync_turn writes land
in the correct session.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def provider():
    """Create an OpenVikingMemoryProvider with faked config."""
    with patch.dict(
        "os.environ",
        {"OPENVIKING_ENDPOINT": "http://localhost:9101", "OPENVIKING_API_KEY": "test"},
    ):
        from plugins.memory.openviking import OpenVikingMemoryProvider

        p = OpenVikingMemoryProvider()
        p._session_id = "old-session"
        p._turn_count = 5
        return p


class TestOpenVikingSessionSwitch:
    def test_updates_session_id(self, provider):
        """on_session_switch should update _session_id to the new value."""
        provider.on_session_switch("new-session")
        assert provider._session_id == "new-session"

    def test_resets_turn_count(self, provider):
        """on_session_switch should reset _turn_count to 0."""
        assert provider._turn_count == 5
        provider.on_session_switch("new-session")
        assert provider._turn_count == 0

    def test_noop_when_same_session(self, provider):
        """on_session_switch should be a no-op when session_id unchanged."""
        provider.on_session_switch("old-session")
        assert provider._session_id == "old-session"
        assert provider._turn_count == 5

    def test_noop_when_empty_new_session(self, provider):
        """on_session_switch should be a no-op when new_session_id is empty."""
        provider.on_session_switch("")
        assert provider._session_id == "old-session"
        assert provider._turn_count == 5

    def test_passes_parent_and_reset_kwargs(self, provider):
        """on_session_switch should accept parent_session_id and reset kwargs."""
        provider.on_session_switch(
            "branch-session", parent_session_id="old-session", reset=True
        )
        assert provider._session_id == "branch-session"
        assert provider._turn_count == 0
