"""Tests for Discord username allowlist resolution guard.

Covers the fix for #44802: _resolve_allowed_usernames rewrites the allowlist
to empty when all configured usernames fail to resolve, causing operator lockout.
"""

import os
from unittest.mock import MagicMock

import pytest

from gateway.config import PlatformConfig


@pytest.fixture()
def adapter(tmp_path, monkeypatch):
    """Create a DiscordAdapter with minimal setup for testing."""
    monkeypatch.setenv("DISCORD_TOKEN", "test-token")
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "alice,bob")

    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._client = MagicMock()
    adapter._client.guilds = []
    return adapter


class TestResolveAllowedUsernamesGuard:
    """Verify that unresolved usernames don't cause operator lockout."""

    @pytest.mark.asyncio
    async def test_all_unresolved_keeps_original_entries(self, adapter, caplog):
        """When ALL usernames fail to resolve, keep original entries."""
        import logging
        adapter._allowed_user_ids = {"alice", "bob"}

        # No guilds → nothing resolves
        adapter._client.guilds = []

        with caplog.at_level(logging.WARNING):
            await adapter._resolve_allowed_usernames()

        # Original entries should be preserved (not overwritten to empty)
        assert "alice" in adapter._allowed_user_ids or "bob" in adapter._allowed_user_ids
        # Warning should be logged about unresolved usernames
        assert any("None of" in r.message and "resolved" in r.message
                    for r in caplog.records), \
            f"Expected warning not found: {[r.message for r in caplog.records]}"

    @pytest.mark.asyncio
    async def test_partial_resolution_updates_with_resolved_ids(self, adapter):
        """When some usernames resolve, update with resolved IDs."""
        adapter._allowed_user_ids = {"alice", "12345"}

        # Mock a guild with alice as a member
        member = MagicMock()
        member.name = "alice"
        member.display_name = "Alice Display"
        member.global_name = None
        member.id = 99999
        member.discriminator = "0"

        guild = MagicMock()
        guild.members = [member]
        guild.member_count = 1
        adapter._client.guilds = [guild]

        await adapter._resolve_allowed_usernames()

        # Should have both the pre-existing numeric ID and the resolved one
        assert "12345" in adapter._allowed_user_ids
        assert "99999" in adapter._allowed_user_ids

    @pytest.mark.asyncio
    async def test_all_numeric_no_resolution_needed(self, adapter):
        """When all entries are numeric, no resolution happens."""
        adapter._allowed_user_ids = {"12345", "67890"}

        await adapter._resolve_allowed_usernames()

        # Unchanged
        assert adapter._allowed_user_ids == {"12345", "67890"}

    @pytest.mark.asyncio
    async def test_empty_allowlist_returns_early(self, adapter):
        """Empty allowlist should return immediately."""
        adapter._allowed_user_ids = set()

        await adapter._resolve_allowed_usernames()
        # No error raised

    @pytest.mark.asyncio
    async def test_mixed_resolution_with_partial_failure(self, adapter, caplog):
        """When some resolve and some don't (but IDs exist), update normally."""
        import logging
        adapter._allowed_user_ids = {"alice", "charlie", "12345"}

        member = MagicMock()
        member.name = "alice"
        member.display_name = "Alice"
        member.global_name = None
        member.id = 11111
        member.discriminator = "0"

        guild = MagicMock()
        guild.members = [member]
        guild.member_count = 1
        adapter._client.guilds = [guild]

        with caplog.at_level(logging.WARNING):
            await adapter._resolve_allowed_usernames()

        # Should have resolved IDs + original numeric
        assert "12345" in adapter._allowed_user_ids
        assert "11111" in adapter._allowed_user_ids
        # charlie should not be in the set (unresolved, but IDs exist)
        assert "charlie" not in adapter._allowed_user_ids
