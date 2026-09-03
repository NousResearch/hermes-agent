import asyncio
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


class _FakeAllowedMentions:
    """Stand-in for ``discord.AllowedMentions`` — exposes the same four
    boolean flags as real attributes so tests can assert on safe defaults.
    """

    def __init__(self, *, everyone=True, roles=True, users=True, replied_user=True):
        self.everyone = everyone
        self.roles = roles
        self.users = users
        self.replied_user = replied_user


def _ensure_discord_mock():
    """Install (or augment) a mock ``discord`` module.

    Always force ``AllowedMentions`` onto whatever is in ``sys.modules`` —
    other test files also stub the module via ``setdefault``, and we need
    ``_build_allowed_mentions()``'s return value to have real attribute
    access regardless of which file loaded first.
    """
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        sys.modules["discord"].AllowedMentions = _FakeAllowedMentions
        return

    if sys.modules.get("discord") is None:
        discord_mod = MagicMock()
        discord_mod.Intents.default.return_value = MagicMock()
        discord_mod.Client = MagicMock
        discord_mod.File = MagicMock
        discord_mod.DMChannel = type("DMChannel", (), {})
        discord_mod.Thread = type("Thread", (), {})
        discord_mod.ForumChannel = type("ForumChannel", (), {})
        discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
        discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, danger=3, green=1, blurple=2, red=3, grey=4, secondary=5)
        discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4)
        discord_mod.Interaction = object
        discord_mod.Embed = MagicMock
        discord_mod.app_commands = SimpleNamespace(
            describe=lambda **kwargs: (lambda fn: fn),
            choices=lambda **kwargs: (lambda fn: fn),
            Choice=lambda **kwargs: SimpleNamespace(**kwargs),
        )
        discord_mod.opus = SimpleNamespace(is_loaded=lambda: True)

        ext_mod = MagicMock()
        commands_mod = MagicMock()
        commands_mod.Bot = MagicMock
        ext_mod.commands = commands_mod

        sys.modules["discord"] = discord_mod
        sys.modules.setdefault("discord.ext", ext_mod)
        sys.modules.setdefault("discord.ext.commands", commands_mod)

    sys.modules["discord"].AllowedMentions = _FakeAllowedMentions


_ensure_discord_mock()

import plugins.platforms.discord.adapter as discord_platform  # noqa: E402
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


@pytest.fixture(autouse=True)
def _speed_up_command_sync_mutation_pacing(monkeypatch):
    monkeypatch.setattr(
        DiscordAdapter,
        "_command_sync_mutation_interval_seconds",
        lambda self: 0.0,
    )


class FakeTree:
    def __init__(self):
        self.sync = AsyncMock(return_value=[])
        self.fetch_commands = AsyncMock(return_value=[])
        self._commands = []

    def command(self, *args, **kwargs):
        return lambda fn: fn

    def get_commands(self, *args, **kwargs):
        return list(self._commands)


class FakeBot:
    def __init__(self, *, intents, proxy=None, allowed_mentions=None, **_):
        self.intents = intents
        self.allowed_mentions = allowed_mentions
        self.application_id = 999
        self.user = SimpleNamespace(id=999, name="Hermes")
        self._events = {}
        self.tree = FakeTree()
        self.http = SimpleNamespace(
            upsert_global_command=AsyncMock(),
            edit_global_command=AsyncMock(),
            delete_global_command=AsyncMock(),
        )

    def event(self, fn):
        self._events[fn.__name__] = fn
        return fn

    async def start(self, token):
        if "on_ready" in self._events:
            await self._events["on_ready"]()

    async def close(self):
        return None


class SlowSyncTree(FakeTree):
    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()
        self.allow_finish = asyncio.Event()

        async def _slow_sync():
            self.started.set()
            await self.allow_finish.wait()
            return []

        self.sync = AsyncMock(side_effect=_slow_sync)


class SlowSyncBot(FakeBot):
    def __init__(self, *, intents, proxy=None):
        super().__init__(intents=intents, proxy=proxy)
        self.tree = SlowSyncTree()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "initial_allowed",
    [
        {"*"},
        {"769524422783664158", "*"},
    ],
)
async def test_resolve_allowed_usernames_preserves_wildcard(monkeypatch, initial_allowed):
    """Regression (#22334): ``_resolve_allowed_usernames`` must not strip ``"*"``.

    The method splits entries into numeric-IDs (kept) vs non-numeric (treated
    as usernames to resolve), then rewrites ``self._allowed_user_ids`` and the
    ``DISCORD_ALLOWED_USERS`` env var from the numeric set. Since ``"*"`` is
    non-numeric, without the wildcard branch it ends up in the resolution
    bucket, fails to match any guild member, and is silently dropped from both
    the in-memory set and the env var on the first ``on_ready`` — quietly
    undoing the wildcard fix in ``_is_allowed_user`` for every subsequent
    message.
    """
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._allowed_user_ids = set(initial_allowed)
    adapter._client = SimpleNamespace(guilds=[])  # no guilds → no resolution work

    monkeypatch.setenv("DISCORD_ALLOWED_USERS", ",".join(sorted(initial_allowed)))

    await adapter._resolve_allowed_usernames()

    assert "*" in adapter._allowed_user_ids, (
        "wildcard must survive _resolve_allowed_usernames; it is the open-mode "
        "marker, not a username to resolve"
    )
    env_entries = {
        e.strip()
        for e in os.environ.get("DISCORD_ALLOWED_USERS", "").split(",")
        if e.strip()
    }
    assert "*" in env_entries, (
        "DISCORD_ALLOWED_USERS env must still contain '*' after resolution; "
        "downstream readers (and any restart-time re-parse) rely on it"
    )


@pytest.mark.asyncio
async def test_reconnect_closes_previous_client_to_prevent_zombie_websocket(monkeypatch):
    """Regression for #18187: calling connect() twice without disconnect() in
    between (e.g. during an in-process reconnect attempt) must close the old
    commands.Bot before creating a new one. Without this guard, two websockets
    stay alive and both fire on_message, producing double responses with
    different wording.
    """
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))

    monkeypatch.setattr("gateway.status.acquire_scoped_lock", lambda scope, identity, metadata=None: (True, None))
    monkeypatch.setattr("gateway.status.release_scoped_lock", lambda scope, identity: None)

    intents = SimpleNamespace(
        message_content=False, dm_messages=False, guild_messages=False,
        members=False, voice_states=False,
    )
    monkeypatch.setattr(discord_platform.Intents, "default", lambda: intents)

    class TrackedBot(FakeBot):
        """FakeBot that records close() calls and reports open/closed state."""
        _closed = False

        def is_closed(self):
            return self._closed

        async def close(self):
            self._closed = True

    created: list[TrackedBot] = []

    def fake_bot_factory(*, command_prefix, intents, proxy=None, allowed_mentions=None, **_):
        bot = TrackedBot(intents=intents, allowed_mentions=allowed_mentions)
        created.append(bot)
        return bot

    monkeypatch.setattr(discord_platform.commands, "Bot", fake_bot_factory)
    monkeypatch.setattr(adapter, "_resolve_allowed_usernames", AsyncMock())

    # First connect — fresh adapter, no prior client.
    assert await adapter.connect() is True
    assert len(created) == 1
    first_bot = created[0]
    assert first_bot._closed is False, "first bot should still be open after connect()"

    # Second connect WITHOUT disconnect — simulates an in-process reconnect.
    # Without the fix, first_bot would remain open (zombie), and both would
    # receive every Discord event, causing double responses.
    assert await adapter.connect() is True
    assert len(created) == 2
    second_bot = created[1]

    # The first bot must be closed before the second is assigned.
    assert first_bot._closed is True, (
        "First Discord client must be closed on re-entry of connect() to prevent "
        "zombie websocket (#18187)"
    )
    assert second_bot._closed is False, "second bot should still be open"
    assert adapter._client is second_bot

    await adapter.disconnect()


@pytest.mark.asyncio
async def test_connect_timeout_cancels_bot_task(monkeypatch):
    """Regression: connect() timeout must cancel _bot_task so the zombie
    Discord client cannot fire on_message after the adapter is discarded.

    Without this fix, the orphaned task eventually completes its WebSocket
    handshake and a subsequent successful reconnect leaves two live clients
    that each process every message, producing duplicate threads.
    """
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))

    monkeypatch.setattr("gateway.status.acquire_scoped_lock", lambda scope, identity, metadata=None: (True, None))
    monkeypatch.setattr("gateway.status.release_scoped_lock", lambda scope, identity: None)

    intents = SimpleNamespace(
        message_content=False, dm_messages=False, guild_messages=False,
        members=False, voice_states=False,
    )
    monkeypatch.setattr(discord_platform.Intents, "default", lambda: intents)

    class NeverReadyBot(FakeBot):
        """Bot whose start() never fires on_ready — simulates a slow gateway handshake."""
        async def start(self, token):
            await asyncio.Event().wait()  # hang forever

    monkeypatch.setattr(
        discord_platform.commands,
        "Bot",
        lambda **kwargs: NeverReadyBot(
            intents=kwargs["intents"],
            proxy=kwargs.get("proxy"),
            allowed_mentions=kwargs.get("allowed_mentions"),
        ),
    )

    async def fake_wait_for_ready(ready_event, bot_task, timeout):
        raise asyncio.TimeoutError()

    monkeypatch.setattr(
        discord_platform, "_wait_for_ready_or_bot_exit", fake_wait_for_ready
    )

    ok = await adapter.connect()

    assert ok is False
    assert adapter._bot_task is None, (
        "_bot_task must be cancelled and cleared on connect() timeout; "
        "leaving it alive creates a zombie Discord client that produces duplicate threads"
    )

@pytest.mark.asyncio
async def test_disconnect_cancels_running_bot_task(monkeypatch):
    """Regression: disconnect() must cancel _bot_task even when connect() timed out.

    _dispose_unused_adapter calls disconnect() on adapters whose connect() returned
    False.  If _bot_task was still running (zombie), disconnect() must cancel it.
    """
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))

    monkeypatch.setattr("gateway.status.acquire_scoped_lock", lambda scope, identity, metadata=None: (True, None))
    monkeypatch.setattr("gateway.status.release_scoped_lock", lambda scope, identity: None)

    # Simulate a zombie bot_task that never finishes (as if discord.py is mid-handshake)
    async def _forever():
        await asyncio.Event().wait()  # hang forever

    zombie_task = asyncio.create_task(_forever())
    adapter._bot_task = zombie_task
    adapter._client = AsyncMock()
    adapter._post_connect_task = None
    adapter._voice_clients = {}
    adapter._running = True
    adapter._ready_event = asyncio.Event()

    await adapter.disconnect()

    # The task must have been cancelled (done + cancelled) and cleared from the adapter.
    assert adapter._bot_task is None, "disconnect() must clear _bot_task"
    assert zombie_task.done(), "disconnect() must have awaited the bot task to completion"
    assert zombie_task.cancelled(), "disconnect() must cancel the zombie bot task"


@pytest.mark.asyncio
async def test_start_post_connect_initialization_reuses_running_task(monkeypatch):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    entered = asyncio.Event()
    release = asyncio.Event()

    async def _blocking_initialization():
        entered.set()
        await release.wait()

    monkeypatch.setattr(adapter, "_run_post_connect_initialization", _blocking_initialization)

    adapter._start_post_connect_initialization()
    first_task = adapter._post_connect_task
    assert first_task is not None
    await asyncio.wait_for(entered.wait(), timeout=1)

    adapter._start_post_connect_initialization()

    assert adapter._post_connect_task is first_task
    release.set()
    await first_task


@pytest.mark.asyncio
async def test_cancel_post_connect_initialization_awaits_running_task(monkeypatch):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    entered = asyncio.Event()

    async def _blocking_initialization():
        entered.set()
        await asyncio.sleep(3600)

    monkeypatch.setattr(adapter, "_run_post_connect_initialization", _blocking_initialization)
    adapter._start_post_connect_initialization()
    task = adapter._post_connect_task
    assert task is not None
    await asyncio.wait_for(entered.wait(), timeout=1)

    await adapter._cancel_post_connect_initialization()

    assert task.cancelled()
    assert adapter._post_connect_task is None


@pytest.mark.asyncio
async def test_safe_sync_slash_commands_only_mutates_diffs():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))

    class _DesiredCommand:
        def __init__(self, payload):
            self._payload = payload

        def to_dict(self, tree):
            assert tree is not None
            return dict(self._payload)

    class _ExistingCommand:
        def __init__(self, command_id, payload):
            self.id = command_id
            self.name = payload["name"]
            self.type = SimpleNamespace(value=payload["type"])
            self._payload = payload

        def to_dict(self):
            return {
                "id": self.id,
                "application_id": 999,
                **self._payload,
                "name_localizations": {},
                "description_localizations": {},
            }

    desired_same = {
        "name": "status",
        "description": "Show Hermes session status",
        "type": 1,
        "options": [],
        "nsfw": False,
        "dm_permission": True,
        "default_member_permissions": None,
    }
    desired_updated = {
        "name": "help",
        "description": "Show available commands",
        "type": 1,
        "options": [],
        "nsfw": False,
        "dm_permission": True,
        "default_member_permissions": None,
    }
    desired_created = {
        "name": "metricas",
        "description": "Show Colmeio metrics dashboard",
        "type": 1,
        "options": [],
        "nsfw": False,
        "dm_permission": True,
        "default_member_permissions": None,
    }
    existing_same = _ExistingCommand(11, desired_same)
    existing_updated = _ExistingCommand(
        12,
        {
            **desired_updated,
            "description": "Old help text",
        },
    )
    existing_deleted = _ExistingCommand(
        13,
        {
            "name": "old-command",
            "description": "To be deleted",
            "type": 1,
            "options": [],
            "nsfw": False,
            "dm_permission": True,
            "default_member_permissions": None,
        },
    )

    fake_tree = SimpleNamespace(
        get_commands=lambda: [
            _DesiredCommand(desired_same),
            _DesiredCommand(desired_updated),
            _DesiredCommand(desired_created),
        ],
        fetch_commands=AsyncMock(return_value=[existing_same, existing_updated, existing_deleted]),
    )
    fake_http = SimpleNamespace(
        upsert_global_command=AsyncMock(),
        edit_global_command=AsyncMock(),
        delete_global_command=AsyncMock(),
    )
    adapter._client = SimpleNamespace(
        tree=fake_tree,
        http=fake_http,
        application_id=999,
        user=SimpleNamespace(id=999),
    )

    summary = await adapter._safe_sync_slash_commands()

    assert summary == {
        "total": 3,
        "unchanged": 1,
        "updated": 1,
        "recreated": 0,
        "created": 1,
        "deleted": 1,
    }
    fake_http.edit_global_command.assert_awaited_once_with(999, 12, desired_updated)
    fake_http.upsert_global_command.assert_awaited_once_with(999, desired_created)
    fake_http.delete_global_command.assert_awaited_once_with(999, 13)


@pytest.mark.asyncio
async def test_post_connect_initialization_retries_timeout_in_same_connection(
    tmp_path, monkeypatch
):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)

    class _DesiredCommand:
        def to_dict(self, tree):
            return {
                "name": "skill",
                "description": "Run a skill",
                "type": 1,
                "options": [],
            }

    adapter._client = SimpleNamespace(
        tree=SimpleNamespace(get_commands=lambda: [_DesiredCommand()]),
        application_id=999,
        user=SimpleNamespace(id=999),
    )
    desired_fingerprint = adapter._desired_command_sync_fingerprint()
    state_path = (
        tmp_path
        / discord_platform._DISCORD_COMMAND_SYNC_STATE_SUBDIR
        / discord_platform._DISCORD_COMMAND_SYNC_STATE_FILENAME
    )
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "999": {
                    "fingerprint": desired_fingerprint,
                    "last_attempt_at": 102.0,
                    "last_success_at": 101.0,
                    "summary": {"total": 1},
                }
            }
        ),
        encoding="utf-8",
    )

    summary = {
        "total": 1,
        "unchanged": 0,
        "updated": 0,
        "recreated": 0,
        "created": 1,
        "deleted": 0,
    }
    sync = AsyncMock(side_effect=[asyncio.TimeoutError(), summary])
    monkeypatch.setattr(adapter, "_safe_sync_slash_commands", sync)
    sleep = AsyncMock()
    monkeypatch.setattr(discord_platform.asyncio, "sleep", sleep)

    await adapter._run_post_connect_initialization()

    assert sync.await_count == 2
    sleep.assert_awaited_once_with(
        discord_platform._DISCORD_COMMAND_SYNC_TIMEOUT_BACKOFF_SECONDS
    )
    recovered_entry = json.loads(state_path.read_text(encoding="utf-8"))["999"]
    assert recovered_entry["last_success_at"] >= recovered_entry["last_attempt_at"]
    assert recovered_entry["summary"] == summary


@pytest.mark.asyncio
async def test_safe_sync_reads_permission_attrs_from_existing_command():
    """Regression: AppCommand.to_dict() in discord.py does NOT include
    nsfw, dm_permission, or default_member_permissions — they live only
    on the attributes. Without reading those attrs, any command with
    non-default permissions false-diffs on every startup.
    """
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))

    class _DesiredCommand:
        def __init__(self, payload):
            self._payload = payload

        def to_dict(self, tree):
            return dict(self._payload)

    class _ExistingCommand:
        """Mirrors discord.py's AppCommand — to_dict() omits nsfw/dm/perms."""

        def __init__(self, command_id, name, description, *, nsfw, guild_only, default_permissions):
            self.id = command_id
            self.name = name
            self.description = description
            self.type = SimpleNamespace(value=1)
            self.nsfw = nsfw
            self.guild_only = guild_only
            self.default_member_permissions = (
                SimpleNamespace(value=default_permissions)
                if default_permissions is not None
                else None
            )

        def to_dict(self):
            # Match real AppCommand.to_dict() — no nsfw/dm_permission/default_member_permissions
            return {
                "id": self.id,
                "type": 1,
                "application_id": 999,
                "name": self.name,
                "description": self.description,
                "name_localizations": {},
                "description_localizations": {},
                "options": [],
            }

    desired = {
        "name": "admin",
        "description": "Admin-only command",
        "type": 1,
        "options": [],
        "nsfw": True,
        "dm_permission": False,
        "default_member_permissions": "8",
    }
    # Existing command has matching attrs — should report unchanged, NOT falsely diff.
    existing = _ExistingCommand(
        42,
        "admin",
        "Admin-only command",
        nsfw=True,
        guild_only=True,
        default_permissions=8,
    )

    fake_tree = SimpleNamespace(
        get_commands=lambda: [_DesiredCommand(desired)],
        fetch_commands=AsyncMock(return_value=[existing]),
    )
    fake_http = SimpleNamespace(
        upsert_global_command=AsyncMock(),
        edit_global_command=AsyncMock(),
        delete_global_command=AsyncMock(),
    )
    adapter._client = SimpleNamespace(
        tree=fake_tree,
        http=fake_http,
        application_id=999,
        user=SimpleNamespace(id=999),
    )

    summary = await adapter._safe_sync_slash_commands()

    # Without the fix, this would be unchanged=0, recreated=1 (false diff).
    assert summary == {
        "total": 1,
        "unchanged": 1,
        "updated": 0,
        "recreated": 0,
        "created": 0,
        "deleted": 0,
    }
    fake_http.edit_global_command.assert_not_awaited()
    fake_http.delete_global_command.assert_not_awaited()
    fake_http.upsert_global_command.assert_not_awaited()


@pytest.mark.asyncio
async def test_safe_sync_ignores_expanded_defaults_but_preserves_explicit_restrictions():
    """Server defaults are ignored only while the local policy is unspecified."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))

    desired = {
        "name": "new",
        "description": "Start a new conversation",
        "type": 1,
        "options": [],
        "nsfw": False,
        "dm_permission": True,
        "default_member_permissions": None,
        "contexts": None,
        "integration_types": None,
    }

    class _DesiredCommand:
        def to_dict(self, tree):
            return dict(desired)

    class _ExistingCommand:
        id = 42
        name = "new"
        description = desired["description"]
        type = SimpleNamespace(value=1)
        nsfw = False
        guild_only = False
        default_member_permissions = None

        def to_dict(self):
            return {
                "id": self.id,
                "application_id": 999,
                **desired,
                "contexts": [0, 1, 2],
                "integration_types": [0, 1],
            }

    fake_http = SimpleNamespace(
        upsert_global_command=AsyncMock(),
        edit_global_command=AsyncMock(),
        delete_global_command=AsyncMock(),
    )
    adapter._client = SimpleNamespace(
        tree=SimpleNamespace(
            get_commands=lambda: [_DesiredCommand()],
            fetch_commands=AsyncMock(return_value=[_ExistingCommand()]),
        ),
        http=fake_http,
        application_id=999,
        user=SimpleNamespace(id=999),
    )

    summary = await adapter._safe_sync_slash_commands()

    assert summary == {
        "total": 1,
        "unchanged": 1,
        "updated": 0,
        "recreated": 0,
        "created": 0,
        "deleted": 0,
    }
    fake_http.edit_global_command.assert_not_awaited()
    fake_http.delete_global_command.assert_not_awaited()
    fake_http.upsert_global_command.assert_not_awaited()

    desired["contexts"] = [0]
    restricted_summary = await adapter._safe_sync_slash_commands()

    assert restricted_summary == {
        "total": 1,
        "unchanged": 0,
        "updated": 0,
        "recreated": 1,
        "created": 0,
        "deleted": 0,
    }
    fake_http.delete_global_command.assert_awaited_once_with(999, 42)
    fake_http.upsert_global_command.assert_awaited_once_with(999, desired)


@pytest.mark.asyncio
async def test_post_connect_initialization_retries_rate_limit_in_same_connection(
    tmp_path, monkeypatch
):
    """A logged retry must happen without waiting for a reconnect."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)

    class _DesiredCommand:
        def to_dict(self, tree):
            return {
                "name": "status",
                "description": "Show Hermes status",
                "type": 1,
                "options": [],
            }

    adapter._client = SimpleNamespace(
        tree=SimpleNamespace(get_commands=lambda: [_DesiredCommand()]),
        application_id=999,
        user=SimpleNamespace(id=999),
    )

    class _DiscordRateLimit(RuntimeError):
        retry_after = 2.0

    summary = {
        "total": 1,
        "unchanged": 1,
        "updated": 0,
        "recreated": 0,
        "created": 0,
        "deleted": 0,
    }
    sync = AsyncMock(side_effect=[_DiscordRateLimit("rate limited"), summary])
    monkeypatch.setattr(adapter, "_safe_sync_slash_commands", sync)
    sleep = AsyncMock()
    monkeypatch.setattr(discord_platform.asyncio, "sleep", sleep)

    await adapter._run_post_connect_initialization()

    assert sync.await_count == 2
    sleep.assert_awaited_once_with(2.0)
    entry = adapter._read_command_sync_state()["999"]
    assert entry["last_success_at"] >= entry["last_attempt_at"]
    assert "retry_after_until" not in entry


@pytest.mark.asyncio
async def test_post_connect_initialization_retry_cancels_cleanly(
    tmp_path, monkeypatch
):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)

    class _DesiredCommand:
        def to_dict(self, tree):
            return {
                "name": "status",
                "description": "Show Hermes status",
                "type": 1,
                "options": [],
            }

    adapter._client = SimpleNamespace(
        tree=SimpleNamespace(get_commands=lambda: [_DesiredCommand()]),
        application_id=999,
        user=SimpleNamespace(id=999),
    )

    class _DiscordRateLimit(RuntimeError):
        retry_after = 2.0

    sync = AsyncMock(side_effect=_DiscordRateLimit("rate limited"))
    monkeypatch.setattr(adapter, "_safe_sync_slash_commands", sync)
    entered_backoff = asyncio.Event()

    async def _blocking_sleep(_seconds):
        entered_backoff.set()
        await asyncio.sleep(3600)

    monkeypatch.setattr(adapter, "_sleep_for_command_sync_retry", _blocking_sleep)

    task = asyncio.create_task(adapter._run_post_connect_initialization())
    await asyncio.wait_for(entered_backoff.wait(), timeout=1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert sync.await_count == 1


@pytest.mark.asyncio
async def test_post_connect_initialization_stops_after_three_rate_limits(
    tmp_path, monkeypatch, caplog
):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)

    class _DesiredCommand:
        def to_dict(self, tree):
            return {
                "name": "status",
                "description": "Show Hermes status",
                "type": 1,
                "options": [],
            }

    adapter._client = SimpleNamespace(
        tree=SimpleNamespace(get_commands=lambda: [_DesiredCommand()]),
        application_id=999,
        user=SimpleNamespace(id=999),
    )

    class _DiscordRateLimit(RuntimeError):
        retry_after = 2.0

    sync = AsyncMock(
        side_effect=[
            _DiscordRateLimit("rate limited"),
            _DiscordRateLimit("rate limited"),
            _DiscordRateLimit("rate limited"),
            AssertionError("slash command sync exceeded its retry budget"),
        ]
    )
    monkeypatch.setattr(adapter, "_safe_sync_slash_commands", sync)
    sleep = AsyncMock()
    monkeypatch.setattr(discord_platform.asyncio, "sleep", sleep)

    await adapter._run_post_connect_initialization()

    assert sync.await_count == 3
    assert [call.args[0] for call in sleep.await_args_list] == [2.0, 2.0]
    entry = adapter._read_command_sync_state()["999"]
    assert entry["last_attempt_at"]
    assert entry["retry_after_until"] > entry["last_attempt_at"]
    assert "last_success_at" not in entry
    assert sum("retrying after" in message for message in caplog.messages) == 2
    assert sum("exhausted 3 attempts" in message for message in caplog.messages) == 1


@pytest.mark.asyncio
async def test_post_connect_initialization_waits_out_persisted_cooldown(
    tmp_path, monkeypatch
):
    """A reconnect must wait out a prior 429 and then reconcile."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)

    class _DesiredCommand:
        def to_dict(self, tree):
            return {
                "name": "status",
                "description": "Show Hermes status",
                "type": 1,
                "options": [],
            }

    adapter._client = SimpleNamespace(
        tree=SimpleNamespace(get_commands=lambda: [_DesiredCommand()]),
        application_id=999,
        user=SimpleNamespace(id=999),
    )
    fingerprint = adapter._desired_command_sync_fingerprint()
    adapter._record_command_sync_rate_limit(999, fingerprint, 2.0)

    summary = {
        "total": 1,
        "unchanged": 1,
        "updated": 0,
        "recreated": 0,
        "created": 0,
        "deleted": 0,
    }
    sync = AsyncMock(return_value=summary)
    monkeypatch.setattr(adapter, "_safe_sync_slash_commands", sync)
    sleep = AsyncMock()
    monkeypatch.setattr(discord_platform.asyncio, "sleep", sleep)

    await adapter._run_post_connect_initialization()

    sync.assert_awaited_once()
    sleep.assert_awaited_once()
    assert sleep.await_args_list[0].args[0] > 0
    entry = adapter._read_command_sync_state()["999"]
    assert entry["last_success_at"] >= entry["last_attempt_at"]


# ============================================================================
# #31049: unconfigured platform skips reconnection (non-retryable fatal error)
# ============================================================================

class TestDiscordUnconfiguredNonRetryable:
    """Verify that missing dependency/token sets a non-retryable fatal error
    so the gateway does not queue the platform for background reconnection."""

    @pytest.mark.asyncio
    async def test_no_discord_lib_sets_non_retryable_fatal(self, monkeypatch):
        """connect() with discord.py unavailable → non-retryable fatal error."""
        _ensure_discord_mock()
        adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake"))
        # Simulate discord.py not installed
        monkeypatch.setattr(discord_platform, "DISCORD_AVAILABLE", False)
        result = await adapter.connect()
        assert result is False
        assert adapter.has_fatal_error is True
        assert adapter.fatal_error_retryable is False
        assert adapter.fatal_error_code == "missing_dependency"


# ============================================================================
# #79430: PrivilegedIntentsRequired → actionable, non-retryable fatal
# ============================================================================

class PrivilegedIntentsRequired(Exception):
    """Stand-in for discord.errors.PrivilegedIntentsRequired."""

    def __init__(self, shard_id=None):
        self.shard_id = shard_id
        super().__init__(
            "Shard ID None is requesting privileged intents that have not been "
            "explicitly enabled in the developer portal."
        )


class TestPrivilegedIntentsRequiredFatal:
    """Missing Developer Portal intents must not spin reconnect forever."""

    def test_guidance_lists_message_content_always(self):
        text = discord_platform._format_privileged_intents_guidance(needs_members=False)
        assert "Message Content Intent" in text
        assert "Server Members Intent" not in text
        assert "discord.com/developers/applications" in text

    def test_guidance_lists_members_when_needed(self):
        text = discord_platform._format_privileged_intents_guidance(needs_members=True)
        assert "Message Content Intent" in text
        assert "Server Members Intent" in text

    def test_needs_members_intent_rules(self):
        needs = discord_platform._needs_server_members_intent
        assert needs(set(), set()) is False
        assert needs({"*"}, set()) is False
        assert needs({"769524422783664158"}, set()) is False
        assert needs({"alice"}, set()) is True
        assert needs(set(), {"111"}) is True

    @pytest.mark.asyncio
    async def test_connect_sets_non_retryable_fatal(self, monkeypatch):
        adapter = DiscordAdapter(
            PlatformConfig(enabled=True, token="test-token", extra={"slash_commands": False})
        )
        monkeypatch.setattr(
            "gateway.status.acquire_scoped_lock",
            lambda scope, identity, metadata=None: (True, None),
        )
        monkeypatch.setattr(
            "gateway.status.release_scoped_lock",
            lambda scope, identity: None,
        )

        intents = SimpleNamespace(
            message_content=False,
            dm_messages=False,
            guild_messages=False,
            members=False,
            voice_states=False,
        )
        monkeypatch.setattr(discord_platform.Intents, "default", lambda: intents)

        class BoomBot(FakeBot):
            async def start(self, token):
                raise PrivilegedIntentsRequired(None)

        monkeypatch.setattr(
            discord_platform.commands,
            "Bot",
            lambda **kwargs: BoomBot(
                intents=kwargs["intents"],
                proxy=kwargs.get("proxy"),
                allowed_mentions=kwargs.get("allowed_mentions"),
            ),
        )

        ok = await adapter.connect()

        assert ok is False
        assert adapter.has_fatal_error is True
        assert adapter.fatal_error_retryable is False
        assert adapter.fatal_error_code == "discord_intents_required"
        assert "Message Content Intent" in (adapter.fatal_error_message or "")
        assert "discord.com/developers/applications" in (adapter.fatal_error_message or "")
        assert adapter._bot_task is None

