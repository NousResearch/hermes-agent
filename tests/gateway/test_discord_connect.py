import asyncio
import json
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

import gateway.platforms.discord as discord_platform  # noqa: E402
from gateway.platforms.discord import DiscordAdapter  # noqa: E402


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
    ("allowed_users", "expected_members_intent"),
    [
        ("769524422783664158", False),
        ("abhey-gupta", True),
        ("769524422783664158,abhey-gupta", True),
    ],
)
async def test_connect_only_requests_members_intent_when_needed(monkeypatch, allowed_users, expected_members_intent):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))

    monkeypatch.setenv("DISCORD_ALLOWED_USERS", allowed_users)
    monkeypatch.setattr("gateway.status.acquire_scoped_lock", lambda scope, identity, metadata=None: (True, None))
    monkeypatch.setattr("gateway.status.release_scoped_lock", lambda scope, identity: None)

    intents = SimpleNamespace(message_content=False, dm_messages=False, guild_messages=False, members=False, voice_states=False)
    monkeypatch.setattr(discord_platform.Intents, "default", lambda: intents)

    created = {}

    def fake_bot_factory(*, command_prefix, intents, proxy=None, allowed_mentions=None, **_):
        created["bot"] = FakeBot(intents=intents, allowed_mentions=allowed_mentions)
        return created["bot"]

    monkeypatch.setattr(discord_platform.commands, "Bot", fake_bot_factory)
    monkeypatch.setattr(adapter, "_resolve_allowed_usernames", AsyncMock())

    ok = await adapter.connect()

    assert ok is True
    assert created["bot"].intents.members is expected_members_intent
    # Safe-default AllowedMentions must be applied on every connect so the
    # bot cannot @everyone from LLM output.  Granular overrides live in the
    # dedicated test_discord_allowed_mentions.py module.
    am = created["bot"].allowed_mentions
    assert am is not None, "connect() must pass an AllowedMentions to commands.Bot"
    assert am.everyone is False
    assert am.roles is False

    await adapter.disconnect()


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
async def test_connect_releases_token_lock_on_timeout(monkeypatch):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))

    monkeypatch.setattr("gateway.status.acquire_scoped_lock", lambda scope, identity, metadata=None: (True, None))
    released = []
    monkeypatch.setattr("gateway.status.release_scoped_lock", lambda scope, identity: released.append((scope, identity)))

    intents = SimpleNamespace(message_content=False, dm_messages=False, guild_messages=False, members=False, voice_states=False)
    monkeypatch.setattr(discord_platform.Intents, "default", lambda: intents)

    monkeypatch.setattr(
        discord_platform.commands,
        "Bot",
        lambda **kwargs: FakeBot(
            intents=kwargs["intents"],
            proxy=kwargs.get("proxy"),
            allowed_mentions=kwargs.get("allowed_mentions"),
        ),
    )

    async def fake_wait_for(awaitable, timeout):
        awaitable.close()
        raise asyncio.TimeoutError()

    monkeypatch.setattr(discord_platform.asyncio, "wait_for", fake_wait_for)

    ok = await adapter.connect()

    assert ok is False
    assert released == [("discord-bot-token", "test-token")]
    assert adapter._platform_lock_identity is None


@pytest.mark.asyncio
async def test_connect_does_not_wait_for_slash_sync(monkeypatch):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))

    monkeypatch.setenv("DISCORD_COMMAND_SYNC_POLICY", "bulk")
    monkeypatch.setattr("gateway.status.acquire_scoped_lock", lambda scope, identity, metadata=None: (True, None))
    monkeypatch.setattr("gateway.status.release_scoped_lock", lambda scope, identity: None)

    intents = SimpleNamespace(message_content=False, dm_messages=False, guild_messages=False, members=False, voice_states=False)
    monkeypatch.setattr(discord_platform.Intents, "default", lambda: intents)

    created = {}

    def fake_bot_factory(*, command_prefix, intents, proxy=None, allowed_mentions=None, **_):
        bot = SlowSyncBot(intents=intents, proxy=proxy)
        created["bot"] = bot
        return bot

    monkeypatch.setattr(discord_platform.commands, "Bot", fake_bot_factory)
    monkeypatch.setattr(adapter, "_resolve_allowed_usernames", AsyncMock())

    ok = await asyncio.wait_for(adapter.connect(), timeout=1.0)

    assert ok is True
    assert adapter._ready_event.is_set()

    await asyncio.wait_for(created["bot"].tree.started.wait(), timeout=1.0)
    assert created["bot"].tree.sync.await_count == 1

    created["bot"].tree.allow_finish.set()
    await asyncio.sleep(0)
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_connect_respects_slash_commands_opt_out(monkeypatch):
    adapter = DiscordAdapter(
        PlatformConfig(enabled=True, token="test-token", extra={"slash_commands": False})
    )

    monkeypatch.setenv("DISCORD_COMMAND_SYNC_POLICY", "off")
    monkeypatch.setattr("gateway.status.acquire_scoped_lock", lambda scope, identity, metadata=None: (True, None))
    monkeypatch.setattr("gateway.status.release_scoped_lock", lambda scope, identity: None)

    intents = SimpleNamespace(message_content=False, dm_messages=False, guild_messages=False, members=False, voice_states=False)
    monkeypatch.setattr(discord_platform.Intents, "default", lambda: intents)
    monkeypatch.setattr(
        discord_platform.commands,
        "Bot",
        lambda **kwargs: FakeBot(
            intents=kwargs["intents"],
            proxy=kwargs.get("proxy"),
            allowed_mentions=kwargs.get("allowed_mentions"),
        ),
    )
    register_mock = MagicMock()
    monkeypatch.setattr(adapter, "_register_slash_commands", register_mock)
    monkeypatch.setattr(adapter, "_resolve_allowed_usernames", AsyncMock())

    ok = await adapter.connect()

    assert ok is True
    register_mock.assert_not_called()

    await adapter.disconnect()


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
async def test_safe_sync_slash_commands_recreates_metadata_only_diffs():
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

    desired = {
        "name": "help",
        "description": "Show available commands",
        "type": 1,
        "options": [],
        "nsfw": False,
        "dm_permission": True,
        "default_member_permissions": "8",
    }
    existing = _ExistingCommand(
        12,
        {
            **desired,
            "default_member_permissions": None,
        },
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

    assert summary == {
        "total": 1,
        "unchanged": 0,
        "updated": 0,
        "recreated": 1,
        "created": 0,
        "deleted": 0,
    }
    fake_http.edit_global_command.assert_not_awaited()
    fake_http.delete_global_command.assert_awaited_once_with(999, 12)
    fake_http.upsert_global_command.assert_awaited_once_with(999, desired)


@pytest.mark.asyncio
async def test_post_connect_initialization_skips_sync_when_policy_off(monkeypatch):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    monkeypatch.setenv("DISCORD_COMMAND_SYNC_POLICY", "off")

    fake_tree = SimpleNamespace(sync=AsyncMock())
    adapter._client = SimpleNamespace(tree=fake_tree)

    await adapter._run_post_connect_initialization()

    fake_tree.sync.assert_not_called()


@pytest.mark.asyncio
async def test_post_connect_initialization_skips_same_fingerprint_after_success(tmp_path, monkeypatch):
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

    fake_tree = SimpleNamespace(
        get_commands=lambda: [_DesiredCommand()],
        fetch_commands=AsyncMock(return_value=[]),
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

    await adapter._run_post_connect_initialization()
    await adapter._run_post_connect_initialization()

    fake_tree.fetch_commands.assert_awaited_once()
    fake_http.upsert_global_command.assert_awaited_once()


@pytest.mark.asyncio
async def test_post_connect_initialization_respects_discord_retry_after(tmp_path, monkeypatch):
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
        retry_after = 123.0

    sync = AsyncMock(side_effect=_DiscordRateLimit("discord rate limited"))
    monkeypatch.setattr(adapter, "_safe_sync_slash_commands", sync)

    await adapter._run_post_connect_initialization()
    await adapter._run_post_connect_initialization()

    sync.assert_awaited_once()
    state_path = (
        tmp_path
        / discord_platform._DISCORD_COMMAND_SYNC_STATE_SUBDIR
        / discord_platform._DISCORD_COMMAND_SYNC_STATE_FILENAME
    )
    state = json.loads(state_path.read_text())
    entry = state["999"]
    assert entry["retry_after"] == 123.0
    assert entry["retry_after_until"] > entry["last_attempt_at"]


@pytest.mark.asyncio
async def test_post_connect_initialization_reraises_non_rate_limit_exceptions(tmp_path, monkeypatch):
    """Arbitrary failures during sync must surface, not be swallowed as rate-limits."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)

    class _DesiredCommand:
        def to_dict(self, tree):
            return {"name": "status", "description": "Show Hermes status", "type": 1, "options": []}

    adapter._client = SimpleNamespace(
        tree=SimpleNamespace(get_commands=lambda: [_DesiredCommand()]),
        application_id=4242,
        user=SimpleNamespace(id=4242),
    )

    # Unrelated failure that happens to expose retry_after. Must NOT be
    # caught by the rate-limit handler — it has nothing to do with 429s.
    class _UnrelatedError(RuntimeError):
        retry_after = 999.0

    sync = AsyncMock(side_effect=_UnrelatedError("database is down"))
    monkeypatch.setattr(adapter, "_safe_sync_slash_commands", sync)

    # The outer _run_post_connect_initialization has a broad except Exception
    # that logs defensively — so we assert on state NOT being written.
    await adapter._run_post_connect_initialization()

    sync.assert_awaited_once()
    state_path = (
        tmp_path
        / discord_platform._DISCORD_COMMAND_SYNC_STATE_SUBDIR
        / discord_platform._DISCORD_COMMAND_SYNC_STATE_FILENAME
    )
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    entry = state.get("4242", {})
    # Attempt was recorded before the sync call, but no rate-limit cooldown
    # should have been persisted from the unrelated exception.
    assert "retry_after_until" not in entry
    assert "retry_after" not in entry


@pytest.mark.asyncio
async def test_safe_sync_slash_commands_paces_mutation_writes(monkeypatch):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    monkeypatch.setattr(
        DiscordAdapter,
        "_command_sync_mutation_interval_seconds",
        lambda self: 1.25,
    )
    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(discord_platform.asyncio, "sleep", fake_sleep)

    class _DesiredCommand:
        def __init__(self, payload):
            self._payload = payload

        def to_dict(self, tree):
            assert tree is not None
            return dict(self._payload)

    desired_one = {
        "name": "status",
        "description": "Show Hermes status",
        "type": 1,
        "options": [],
    }
    desired_two = {
        "name": "debug",
        "description": "Generate a debug report",
        "type": 1,
        "options": [],
    }
    fake_tree = SimpleNamespace(
        get_commands=lambda: [_DesiredCommand(desired_one), _DesiredCommand(desired_two)],
        fetch_commands=AsyncMock(return_value=[]),
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

    assert summary["created"] == 2
    assert fake_http.upsert_global_command.await_count == 2
    assert sleeps == [1.25]


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
async def test_safe_sync_detects_contexts_drift():
    """Regression: contexts and integration_types must be canonicalized
    so drift in those fields triggers reconciliation. Without this, the
    diff silently reports 'unchanged' and never reconciles.
    """
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))

    class _DesiredCommand:
        def __init__(self, payload):
            self._payload = payload

        def to_dict(self, tree):
            return dict(self._payload)

    class _ExistingCommand:
        def __init__(self, command_id, payload):
            self.id = command_id
            self.name = payload["name"]
            self.description = payload["description"]
            self.type = SimpleNamespace(value=1)
            self.nsfw = payload.get("nsfw", False)
            self.guild_only = not payload.get("dm_permission", True)
            self.default_member_permissions = None
            self._payload = payload

        def to_dict(self):
            return {
                "id": self.id,
                "type": 1,
                "application_id": 999,
                "name": self.name,
                "description": self.description,
                "name_localizations": {},
                "description_localizations": {},
                "options": [],
                "contexts": self._payload.get("contexts"),
                "integration_types": self._payload.get("integration_types"),
            }

    desired = {
        "name": "help",
        "description": "Show available commands",
        "type": 1,
        "options": [],
        "nsfw": False,
        "dm_permission": True,
        "default_member_permissions": None,
        "contexts": [0, 1, 2],
        "integration_types": [0, 1],
    }
    existing = _ExistingCommand(
        77,
        {
            **desired,
            "contexts": [0],  # server-side only
            "integration_types": [0],
        },
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

    # contexts and integration_types are not patchable by
    # edit_global_command, so the command must be recreated.
    assert summary["unchanged"] == 0
    assert summary["recreated"] == 1
    assert summary["updated"] == 0
    fake_http.edit_global_command.assert_not_awaited()
    fake_http.delete_global_command.assert_awaited_once_with(999, 77)
    fake_http.upsert_global_command.assert_awaited_once_with(999, desired)
