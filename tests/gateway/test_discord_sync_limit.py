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


# The sync-limit tests above use the fixture's identity-canonicalize mock; the
# churn-regression tests below deliberately exercise the REAL canonicalizer
# (del of the instance attribute falls back to the bound method) so field
# normalization matches production.

class _UnsetPrefTreeCommand:
    """Desired command whose tree sets no installation/context preference.

    Mirrors discord.py's ``to_dict()``: emits ``None`` for unset
    ``integration_types``/``contexts``, while Discord materializes defaults
    server-side.
    """

    def __init__(self, name: str, description: str = "Show or change the model"):
        self.name = name
        self.type = 1
        self._description = description

    def to_dict(self, _tree):
        return {
            "name": self.name, "type": 1,
            "description": self._description,
            "dm_permission": True, "nsfw": False,
            "contexts": None, "integration_types": None, "options": [],
        }


class _PinnedTreeCommand(_UnsetPrefTreeCommand):
    """Desired command that EXPLICITLY pins installation/context values."""

    def __init__(self, name: str):
        super().__init__(name)
        self._description = "d"

    def to_dict(self, _tree):
        return {
            "name": self.name, "type": 1, "description": self._description,
            "dm_permission": True, "nsfw": False,
            "contexts": [0], "integration_types": [0], "options": [],
        }


@pytest.mark.asyncio
async def test_sync_treats_unset_installation_context_as_matching(adapter):
    """Desired integration_types/contexts=None must match server-materialized values.

    Regression guard for the churn bug: discord.py's to_dict() emits None when
    the tree sets no preference, but Discord stores [0, 1]; naive payload
    equality flagged every command changed on every sync, the reconciler
    delete+recreated all of them, the 600s sync timeout meant last_success_at
    was never stamped, and the rate bucket burned — the 'slash commands break
    on every update' signature. Unset on the DESIRED side means no preference.
    """
    adapter._existing_command_to_payload = MagicMock(side_effect=lambda cmd: {
        "name": cmd.name, "type": 1, "description": "Show or change the model",
        "default_member_permissions": None, "dm_permission": True, "nsfw": False,
        "contexts": [0, 1, 2], "integration_types": [0, 1], "options": [],
    })
    del adapter._canonicalize_app_command_payload

    adapter._client.tree.fetch_commands = AsyncMock(
        return_value=[SimpleNamespace(id="id_model", name="model", type=1)])
    adapter._client.tree.get_commands = MagicMock(
        return_value=[_UnsetPrefTreeCommand(name="model")])

    summary = await adapter._safe_sync_slash_commands()

    assert summary["unchanged"] == 1
    assert summary["created"] == 0 and summary["recreated"] == 0
    assert summary["updated"] == 0 and summary["deleted"] == 0
    assert adapter._client.http.delete_global_command.await_count == 0
    assert adapter._client.http.upsert_global_command.await_count == 0


@pytest.mark.asyncio
async def test_sync_still_mutates_when_desired_pins_installation_explicitly(adapter):
    """An EXPLICIT desired integration_types/contexts value still compares exactly.

    Unset-tolerance must mean 'no preference', never 'anything matches': a tree
    that pins [0] against a server-materialized [0, 1] is a real change and must
    still be reconciled (patchable fields equal → the recreate path).
    """
    adapter._existing_command_to_payload = MagicMock(side_effect=lambda cmd: {
        "name": cmd.name, "type": 1, "description": "d",
        "default_member_permissions": None, "dm_permission": True, "nsfw": False,
        "contexts": [0], "integration_types": [0, 1], "options": [],
    })
    del adapter._canonicalize_app_command_payload
    # Exercise the REAL patchable shape (name/description/options only) so the
    # pinned-but-patchable-equal case takes the production recreate path.
    del adapter._patchable_app_command_payload

    adapter._client.tree.fetch_commands = AsyncMock(
        return_value=[SimpleNamespace(id="id_x", name="model", type=1)])
    adapter._client.tree.get_commands = MagicMock(
        return_value=[_PinnedTreeCommand(name="model")])

    summary = await adapter._safe_sync_slash_commands()

    assert summary["recreated"] == 1
    assert summary["unchanged"] == 0


@pytest.mark.asyncio
async def test_safe_sync_deletes_before_creating():
    """Sync must delete obsolete commands BEFORE creating new ones.

    Discord's 100-command limit is enforced when trying to upsert. If we
    have 100 commands on Discord, try to add 1 new one, and haven't deleted
    any yet, Discord rejects with error 30032.

    The fix: identify and delete obsolete commands first, then create/update.
    This ensures we never temporarily exceed 100 during the sync operation.

    This is a regression guard for the samuraiheart bug where sync would fail
    with error 30032 even though the registration code properly capped at 100.
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

    # Simulate having 100 commands on Discord, with 1 that's no longer desired
    # and 1 new command that should be created.
    # Existing on Discord: cmd_0, cmd_1, ..., cmd_99 (100 total)
    # Desired locally: cmd_1, cmd_2, ..., cmd_99, cmd_new (100 total)
    # So: delete cmd_0 (1 deletion), create cmd_new (1 creation)

    existing_commands = [
        SimpleNamespace(id=f"id_{i}", name=f"cmd_{i}", type=1)
        for i in range(100)
    ]
    adapter._client.tree.fetch_commands = AsyncMock(return_value=existing_commands)

    adapter._client.tree.get_commands = MagicMock(
        return_value=[
            _FakeTreeCommand(name=f"cmd_{i}", command_type=1)
            for i in range(1, 100)
        ] + [_FakeTreeCommand(name="cmd_new", command_type=1)]
    )

    # Track the order of mutations
    mutation_log = []

    async def mock_delete(*args):
        mutation_log.append(("delete", args[-1]))

    async def mock_upsert(*args):
        mutation_log.append(("create", args[-1].get("name")))

    adapter._client.http.delete_global_command = mock_delete
    adapter._client.http.upsert_global_command = mock_upsert
    adapter._client.http.edit_global_command = AsyncMock()

    # Call sync
    await adapter._safe_sync_slash_commands()

    # Verify that:
    # 1. A deletion happened (cmd_0)
    # 2. It happened BEFORE any creation
    # 3. The creation of cmd_new happened AFTER deletion
    deletes = [m for m in mutation_log if m[0] == "delete"]
    creates = [m for m in mutation_log if m[0] == "create"]

    assert len(deletes) >= 1, "At least one command should be deleted"
    assert len(creates) >= 1, "At least one command should be created"

    # The key assertion: all deletions should come before all creations.
    # Find the index of the last delete and the first create.
    last_delete_idx = max(i for i, m in enumerate(mutation_log) if m[0] == "delete")
    first_create_idx = min(i for i, m in enumerate(mutation_log) if m[0] == "create")

    assert last_delete_idx < first_create_idx, (
        f"Deletions must happen before creations to avoid exceeding 100-command limit. "
        f"Last delete at index {last_delete_idx}, first create at index {first_create_idx}"
    )
