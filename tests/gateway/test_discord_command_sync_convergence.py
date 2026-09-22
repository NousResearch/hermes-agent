"""A converged Discord command tree must not be re-registered, and a stalled run must not lose work.

Discord's own REST payload is the contract the sync diffs against, so the stubs here mirror what
``GET /applications/{id}/commands`` returns for a command discord.py created: the API fills in the
default installation context and reports ``nsfw``/``guild_only``/``default_member_permissions``
only on the model attributes, never inside ``to_dict()``.
"""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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


class _DesiredCommand:
    """A command on discord.py's tree: ``to_dict(tree)`` is the payload that would be sent."""

    def __init__(self, payload):
        self._payload = payload

    def to_dict(self, tree):
        assert tree is not None
        return dict(self._payload)


class _ExistingCommand:
    """An ``AppCommand`` built from Discord's stored payload."""

    def __init__(self, command_id, name, description, *, integration_types, options=None, nsfw=False):
        self.id = command_id
        self.name = name
        self.description = description
        self.type = SimpleNamespace(value=1)
        self.nsfw = nsfw
        self.guild_only = False
        self.default_member_permissions = None
        self._integration_types = integration_types
        self._options = list(options or [])

    def to_dict(self):
        return {
            "id": self.id,
            "type": 1,
            "application_id": 999,
            "name": self.name,
            "description": self.description,
            "name_localizations": {},
            "description_localizations": {},
            "contexts": None,
            "integration_types": self._integration_types,
            "options": self._options,
        }


def _desired(name, description):
    """What ``Command.to_dict(tree)`` emits for a command registered without install contexts."""
    return {
        "name": name,
        "description": description,
        "type": 1,
        "options": [],
        "nsfw": False,
        "dm_permission": True,
        "default_member_permissions": None,
        "contexts": None,
        "integration_types": None,
    }


def _adapter(desired, existing):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._sleep_between_command_sync_mutations = AsyncMock()
    adapter._client = SimpleNamespace(
        tree=SimpleNamespace(
            get_commands=lambda: [_DesiredCommand(payload) for payload in desired],
            fetch_commands=AsyncMock(return_value=list(existing)),
        ),
        http=SimpleNamespace(
            upsert_global_command=AsyncMock(),
            edit_global_command=AsyncMock(),
            delete_global_command=AsyncMock(),
        ),
        application_id=999,
        user=SimpleNamespace(id=999),
    )
    return adapter


@pytest.mark.asyncio
async def test_discord_filled_in_install_contexts_are_not_a_change():
    """The install contexts Discord fills in are not a difference Hermes can act on.

    Discord stores the app's own default — GUILD_INSTALL, or both contexts once the app offers user
    install — for a command created without ``integration_types``, while discord.py reports that
    same command as ``None``. Diffing the raw shapes re-registered the whole app on every connect:
    68 commands x 2 paced mutations x 4.5s = 612s inside the 600s cap, so no connect finished and no
    sync was ever recorded as complete.
    """
    adapter = _adapter(
        [
            _desired("help", "Show available commands"),
            _desired("status", "Show Hermes session status"),
        ],
        [
            _ExistingCommand(42, "help", "Show available commands", integration_types=[0]),
            _ExistingCommand(43, "status", "Show Hermes session status", integration_types=[0, 1]),
        ],
    )

    summary = await adapter._safe_sync_slash_commands()

    assert summary["unchanged"] == 2
    adapter._client.http.upsert_global_command.assert_not_awaited()
    adapter._client.http.edit_global_command.assert_not_awaited()
    adapter._client.http.delete_global_command.assert_not_awaited()


@pytest.mark.asyncio
async def test_reregistering_a_command_does_not_remove_it_first():
    """A field discord.py's edit route cannot carry forces a re-registration.

    That re-registration must be one upsert (POST /commands upserts by name). The delete it used to
    run first left the command absent from the app whenever a connect was cut between the two calls
    — how /branch and /suggestions disappeared — and doubled the paced mutations.
    """
    adapter = _adapter(
        [_desired("voice", "Toggle voice reply mode")],
        [_ExistingCommand(44, "voice", "Toggle voice reply mode", integration_types=[0], nsfw=True)],
    )

    summary = await adapter._safe_sync_slash_commands()

    assert summary["recreated"] == 1
    adapter._client.http.upsert_global_command.assert_awaited_once()
    adapter._client.http.delete_global_command.assert_not_awaited()


@pytest.mark.asyncio
async def test_one_rejected_command_does_not_discard_the_rest_of_the_run():
    """A command Discord refuses must not abort the sync: the remaining diff is still applied."""
    adapter = _adapter(
        [_desired("help", "New help text"), _desired("status", "New status text")],
        [
            _ExistingCommand(45, "help", "Old help text", integration_types=[0]),
            _ExistingCommand(46, "status", "Old status text", integration_types=[0]),
        ],
    )
    adapter._client.http.edit_global_command = AsyncMock(
        side_effect=[RuntimeError("400 Bad Request (error code: 50035)"), None],
    )

    summary = await adapter._safe_sync_slash_commands()

    assert adapter._client.http.edit_global_command.await_count == 2
    assert summary["updated"] == 1
    assert summary["failed"] == 1


@pytest.mark.asyncio
async def test_a_run_that_cannot_fit_its_mutations_reports_them(monkeypatch):
    """Out of budget ⇒ stop between mutations and report the remainder.

    The caller needs to tell "synced" from "gave up" so it never records an unfinished run as
    successful, and the app is never left half-mutated by a cancellation.
    """
    adapter = _adapter(
        [_desired("help", "New help text")],
        [_ExistingCommand(47, "help", "Old help text", integration_types=[0])],
    )
    monkeypatch.setattr(adapter, "_command_sync_budget_seconds", lambda: 0.0)

    summary = await adapter._safe_sync_slash_commands()

    assert summary["updated"] == 0
    assert summary.get("deferred_mutations") == 1
    adapter._client.http.edit_global_command.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_run_that_runs_out_of_budget_stops_and_reports_the_boundary(monkeypatch):
    """The budget stops the run between mutations, and ``deferred_mutations`` says what it counted.

    The budget check runs before the pacing pause of the mutation it gates, so a run of three
    changed commands with a budget of one and a half paced mutations applies two and reports the
    third as a refused opportunity — the count is mutation opportunities, not commands left behind
    (a recreate here is a single upsert, so a stopped run never leaves a command half-applied).
    """
    import plugins.platforms.discord.adapter as adapter_module

    adapter = _adapter(
        [
            _desired("help", "New help text"),
            _desired("status", "New status text"),
            _desired("model", "New model text"),
        ],
        [
            _ExistingCommand(60, "help", "Old help text", integration_types=[0]),
            _ExistingCommand(61, "status", "Old status text", integration_types=[0]),
            _ExistingCommand(62, "model", "Old model text", integration_types=[0]),
        ],
    )
    interval = 1.0
    monkeypatch.setattr(adapter_module, "_DISCORD_COMMAND_SYNC_MUTATION_INTERVAL_SECONDS", interval)
    monkeypatch.setattr(adapter, "_command_sync_budget_seconds", lambda: interval * 1.5)

    async def _pace() -> None:
        # The real pacing pause (``_adapter`` stubs it out): the budget check adds the interval to
        # "now", so the third mutation is refused once the pace of the second one has elapsed.
        await asyncio.sleep(interval)

    monkeypatch.setattr(adapter, "_sleep_between_command_sync_mutations", _pace)

    summary = await adapter._safe_sync_slash_commands()

    assert adapter._client.http.edit_global_command.await_count == 2
    assert summary["updated"] == 2
    assert summary["total"] == 3
    assert summary["deferred_mutations"] == 1
    assert summary["failed"] == 0
