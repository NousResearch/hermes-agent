"""Test Discord slash command sync respects the 100-command hard limit."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import sys

import pytest

from gateway.config import PlatformConfig


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return
    if sys.modules.get("discord") is None:
        discord_mod = MagicMock()
        discord_mod.Intents.default.return_value = MagicMock()
        sys.modules["discord"] = discord_mod
        sys.modules["discord.ext"] = MagicMock()
        sys.modules["discord.ext.commands"] = MagicMock()


_ensure_discord_mock()

from plugins.platforms.discord.adapter import DiscordAdapter


class _FakeTreeCommand:
    """Minimal command stub matching discord.py tree command API."""

    def __init__(self, name: str, command_type: int = 1):
        self.name = name
        self.type = command_type

    def to_dict(self, _tree):
        return {"name": self.name, "type": self.type}


@pytest.fixture
def adapter():
    """Create a Discord adapter with mocked Discord client."""
    _ensure_discord_mock()
    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = DiscordAdapter(config)

    # Mock the Discord client and tree
    adapter._client = MagicMock()
    adapter._client.tree = MagicMock()
    adapter._client.http = AsyncMock()
    adapter._client.application_id = "test_app_id"

    adapter._sleep_between_command_sync_mutations = AsyncMock()
    adapter._existing_command_to_payload = MagicMock(side_effect=lambda cmd: {"name": cmd.name})
    adapter._canonicalize_app_command_payload = MagicMock(side_effect=lambda p: p)
    adapter._patchable_app_command_payload = MagicMock(side_effect=lambda p: p)

    return adapter


@pytest.mark.asyncio
async def test_safe_sync_never_exceeds_the_100_command_cap():
    """Sync must never push the live command total above Discord's cap.

    Discord's 100-command limit is enforced on upsert; exceeding it fails the
    ENTIRE sync with error 30032, silently breaking every slash command. This
    is a regression guard for the samuraiheart bug.

    Two shapes satisfy the invariant and both are accepted here:

    * a single atomic PUT bulk-overwrite (``tree.sync()``) — one request that
      replaces the whole registry, so the live total is never transiently
      wrong, and a 429 leaves Discord untouched; or
    * the per-command fallback, which must delete obsolete commands BEFORE
      creating new ones.

    The original form of this test asserted the ordering *literally*, which
    pinned the delete-then-create behaviour that could vacate a live command
    when a 429 landed mid-sync (the 2026-09-20 loss of /stop and /restart).
    The invariant — never over the cap, never a vacated command — is what
    actually matters.
    """
    _ensure_discord_mock()
    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = DiscordAdapter(config)

    adapter._client = MagicMock()
    adapter._client.tree = MagicMock()
    adapter._client.http = AsyncMock()
    adapter._client.application_id = "test_app_id"
    adapter._sleep_between_command_sync_mutations = AsyncMock()
    adapter._existing_command_to_payload = MagicMock(side_effect=lambda cmd: {"name": cmd.name})
    adapter._canonicalize_app_command_payload = MagicMock(side_effect=lambda p: p)
    adapter._patchable_app_command_payload = MagicMock(side_effect=lambda p: p)

    # Existing on Discord: cmd_0..cmd_99 (100, i.e. at the cap)
    # Desired locally:     cmd_1..cmd_99 + cmd_new (100)
    # So: delete cmd_0, create cmd_new.
    existing_commands = [
        SimpleNamespace(id=f"id_{i}", name=f"cmd_{i}", type=1)
        for i in range(100)
    ]
    adapter._client.tree.fetch_commands = AsyncMock(return_value=existing_commands)
    adapter._client.tree.sync = AsyncMock(return_value=[])

    adapter._client.tree.get_commands = MagicMock(
        return_value=[
            _FakeTreeCommand(name=f"cmd_{i}", command_type=1)
            for i in range(1, 100)
        ] + [_FakeTreeCommand(name="cmd_new", command_type=1)]
    )

    mutation_log = []

    async def mock_delete(*args):
        mutation_log.append(("delete", args[-1]))

    async def mock_upsert(*args):
        mutation_log.append(("create", args[-1].get("name")))

    adapter._client.http.delete_global_command = mock_delete
    adapter._client.http.upsert_global_command = mock_upsert
    adapter._client.http.edit_global_command = AsyncMock()

    summary = await adapter._safe_sync_slash_commands()

    assert summary["deleted"] == 1
    assert summary["created"] == 1

    if summary.get("bulk"):
        # One atomic PUT replaced the registry — the live total goes from 100
        # straight to 100 with no intermediate state to exceed the cap.
        adapter._client.tree.sync.assert_awaited_once()
        assert not mutation_log, (
            "a bulk overwrite must not also issue per-command mutations"
        )
        return

    # Fallback path: deletions must precede creations.
    deletes = [m for m in mutation_log if m[0] == "delete"]
    creates = [m for m in mutation_log if m[0] == "create"]
    assert len(deletes) >= 1, "At least one command should be deleted"
    assert len(creates) >= 1, "At least one command should be created"

    last_delete_idx = max(i for i, m in enumerate(mutation_log) if m[0] == "delete")
    first_create_idx = min(i for i, m in enumerate(mutation_log) if m[0] == "create")

    assert last_delete_idx < first_create_idx, (
        f"Deletions must happen before creations to avoid exceeding 100-command limit. "
        f"Last delete at index {last_delete_idx}, first create at index {first_create_idx}"
    )


@pytest.mark.asyncio
async def test_fallback_path_still_deletes_before_creating_over_the_cap():
    """Above the cap, bulk is unavailable — the ordered fallback must apply.

    With more desired commands than Discord's hard cap, a single PUT cannot be
    used, so the per-command path runs and the delete-before-create ordering
    remains the only thing keeping the live total under 100 mid-sync.
    """
    _ensure_discord_mock()
    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = DiscordAdapter(config)

    adapter._client = MagicMock()
    adapter._client.tree = MagicMock()
    adapter._client.http = AsyncMock()
    adapter._client.application_id = "test_app_id"
    adapter._sleep_between_command_sync_mutations = AsyncMock()
    adapter._existing_command_to_payload = MagicMock(side_effect=lambda cmd: {"name": cmd.name})
    adapter._canonicalize_app_command_payload = MagicMock(side_effect=lambda p: p)
    adapter._patchable_app_command_payload = MagicMock(side_effect=lambda p: p)

    existing_commands = [
        SimpleNamespace(id=f"id_{i}", name=f"cmd_{i}", type=1)
        for i in range(101)
    ]
    adapter._client.tree.fetch_commands = AsyncMock(return_value=existing_commands)
    adapter._client.tree.sync = AsyncMock(return_value=[])
    adapter._client.tree.get_commands = MagicMock(
        return_value=[
            _FakeTreeCommand(name=f"cmd_{i}", command_type=1)
            for i in range(1, 101)
        ] + [_FakeTreeCommand(name="cmd_new", command_type=1)]
    )

    mutation_log = []

    async def mock_delete(*args):
        mutation_log.append(("delete", args[-1]))

    async def mock_upsert(*args):
        mutation_log.append(("create", args[-1].get("name")))

    adapter._client.http.delete_global_command = mock_delete
    adapter._client.http.upsert_global_command = mock_upsert
    adapter._client.http.edit_global_command = AsyncMock()

    summary = await adapter._safe_sync_slash_commands()

    assert not summary.get("bulk"), "over the cap, bulk overwrite must not be used"
    adapter._client.tree.sync.assert_not_awaited()

    last_delete_idx = max(i for i, m in enumerate(mutation_log) if m[0] == "delete")
    first_create_idx = min(i for i, m in enumerate(mutation_log) if m[0] == "create")
    assert last_delete_idx < first_create_idx
