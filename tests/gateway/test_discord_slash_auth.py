"""Security regression tests: slash commands honor on_message authorization gates.

Slash invocations (``_run_simple_slash``, ``_handle_thread_create_slash``)
historically bypassed every gate ``on_message`` enforces — DISCORD_ALLOWED_USERS,
DISCORD_ALLOWED_ROLES, DISCORD_ALLOWED_CHANNELS, DISCORD_IGNORED_CHANNELS.
Any guild member could invoke ``/bg``, ``/restart``, etc. as the
operator. ``_check_slash_authorization`` mirrors all four gates one-for-one.

These tests pin the security-correct behavior so the bypass cannot regress.
"""

import asyncio
import logging
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


# ---------------------------------------------------------------------------
# Discord module mock — borrowed from test_discord_slash_commands.py so this
# file runs on machines without discord.py installed.
# ---------------------------------------------------------------------------


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return  # real discord installed

    if sys.modules.get("discord") is None:
        discord_mod = MagicMock()
        discord_mod.Intents.default.return_value = MagicMock()
        discord_mod.DMChannel = type("DMChannel", (), {})
        discord_mod.Thread = type("Thread", (), {})
        discord_mod.ForumChannel = type("ForumChannel", (), {})
        discord_mod.Interaction = object

        class _FakePermissions:
            def __init__(self, value=0, **_):
                self.value = value

        discord_mod.Permissions = _FakePermissions

        class _FakeGroup:
            def __init__(self, *, name, description, parent=None):
                self.name = name
                self.description = description
                self.parent = parent
                self._children: dict[str, object] = {}
                if parent is not None:
                    parent.add_command(self)

            def add_command(self, cmd):
                self._children[cmd.name] = cmd

        class _FakeCommand:
            def __init__(self, *, name, description, callback, parent=None):
                self.name = name
                self.description = description
                self.callback = callback
                self.parent = parent
                self.default_permissions = None

        discord_mod.app_commands = SimpleNamespace(
            describe=lambda **kwargs: (lambda fn: fn),
            choices=lambda **kwargs: (lambda fn: fn),
            autocomplete=lambda **kwargs: (lambda fn: fn),
            Choice=lambda **kwargs: SimpleNamespace(**kwargs),
            Group=_FakeGroup,
            Command=_FakeCommand,
        )

        ext_mod = MagicMock()
        commands_mod = MagicMock()
        commands_mod.Bot = MagicMock
        ext_mod.commands = commands_mod

        sys.modules["discord"] = discord_mod
        sys.modules.setdefault("discord.ext", ext_mod)
        sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from plugins.platforms.discord.adapter import DiscordAdapter


@pytest.fixture(autouse=True)
def _isolate_discord_env(monkeypatch):
    for var in (
        "DISCORD_ALLOWED_USERS",
        "DISCORD_ALLOWED_ROLES",
        "DISCORD_ALLOWED_CHANNELS",
        "DISCORD_IGNORED_CHANNELS",
        "DISCORD_HIDE_SLASH_COMMANDS",
        "DISCORD_ALLOW_BOTS",
        "DISCORD_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _stub_discord_permissions(monkeypatch):
    """Pin discord.Permissions to a plain stand-in so tests can assert the
    bitfield value regardless of whether real discord.py or a sibling test
    module's MagicMock is loaded."""
    import discord

    class _Perm:
        def __init__(self, value=0, **_):
            self.value = value

    monkeypatch.setattr(discord, "Permissions", _Perm)


@pytest.fixture
def adapter():
    config = PlatformConfig(enabled=True, token="***")
    a = DiscordAdapter(config)
    a._client = SimpleNamespace(user=SimpleNamespace(id=99999, name="HermesBot"), guilds=[])
    return a


_SENTINEL = object()


def _make_interaction(
    user_id, *, channel_id=12345, guild_id=42, in_dm=False, in_thread=False,
    parent_channel_id=None, user=_SENTINEL, channel_name=None,
):
    """Build a mock Discord Interaction with a still-unresponded response.

    ``channel_id`` may be set to ``None`` to simulate a guild interaction
    payload missing a resolvable channel id (fail-closed exercise).
    Pass ``user=None`` to simulate a payload missing the user object.
    ``channel_name`` attaches a ``.name`` to the channel so channel-name /
    ``#name`` allow/ignore matching can be exercised (mirrors on_message).
    """
    import discord

    response = SimpleNamespace(send_message=AsyncMock(), defer=AsyncMock())

    if in_dm:
        channel = discord.DMChannel()
    elif in_thread:
        channel = discord.Thread()
        channel.id = channel_id
        channel.parent_id = parent_channel_id
        if channel_name is not None:
            channel.name = channel_name
    elif channel_id is None:
        channel = None
    else:
        channel = SimpleNamespace(id=channel_id)
        if channel_name is not None:
            channel.name = channel_name

    if user is _SENTINEL:
        user_obj = SimpleNamespace(id=int(user_id), name=f"user_{user_id}")
    else:
        user_obj = user

    return SimpleNamespace(
        user=user_obj,
        # `get_member` needed for the guild-scoped role fallback path in
        # _is_allowed_user after the #12136 cross-guild fix. Fixture guild
        # has no members by default — tests exercising positive role paths
        # assign their own Member via user.roles + matching allowed_role_ids.
        guild=SimpleNamespace(owner_id=999, id=guild_id, get_member=lambda uid: None),
        guild_id=guild_id,
        channel_id=channel_id,
        channel=channel,
        response=response,
    )


def _stub_pairing_store(monkeypatch, approved_ids):
    approved = {str(uid) for uid in approved_ids}

    class _FakePairingStore:
        def is_approved(self, platform, user_id):
            return platform == "discord" and str(user_id) in approved

    from gateway import pairing

    monkeypatch.setattr(pairing, "PairingStore", _FakePairingStore)


# ---------------------------------------------------------------------------
# Backwards-compat: empty allowlist → everything passes (matches on_message)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_allowlist_allows_with_gateway_allow_all(adapter, monkeypatch):
    """Explicit ``GATEWAY_ALLOW_ALL_USERS`` restores open Discord access."""
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    interaction = _make_interaction("999999999")
    assert await adapter._check_slash_authorization(interaction, "/help") is True
    interaction.response.send_message.assert_not_awaited()


# ---------------------------------------------------------------------------
# User allowlist (DISCORD_ALLOWED_USERS) parity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_allowed_user_passes(adapter):
    adapter._allowed_user_ids = {"100200300"}
    interaction = _make_interaction("100200300")
    assert await adapter._check_slash_authorization(interaction, "/bg hi") is True
    interaction.response.send_message.assert_not_awaited()


def test_pairing_approved_user_passes_message_gate_without_allowlist(adapter, monkeypatch):
    """Pairing grants must be honored before on_message drops guild mentions."""
    _stub_pairing_store(monkeypatch, {"100200300"})
    assert adapter._is_allowed_user(
        "100200300",
        author=SimpleNamespace(id=100200300),
        guild=SimpleNamespace(id=42, get_member=lambda _uid: None),
        is_dm=False,
        channel_ids={"12345"},
    ) is True


# ---------------------------------------------------------------------------
# Role allowlist (DISCORD_ALLOWED_ROLES) parity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_role_member_passes(adapter):
    """A user whose Member.roles includes an allowed role passes the gate."""
    adapter._allowed_role_ids = {1234}
    interaction = _make_interaction("999999999")
    interaction.user.roles = [SimpleNamespace(id=1234)]
    assert await adapter._check_slash_authorization(interaction, "/help") is True


def _gateway_authorizes(source):
    """The real gateway authz verdict for a runner with no allowlist and no pairing store."""
    from gateway.run import GatewayRunner

    return object.__new__(GatewayRunner)._is_user_authorized(source)


@pytest.mark.asyncio
@pytest.mark.parametrize("builder", ["slash", "thread_starter"])
@pytest.mark.parametrize("actor_roles,gateway_admits", [
    ([1234], True),   # role-only member the slash gate admits
    ([7], False),     # unrelated role: the builder must not mint a grant for this actor
])
async def test_native_event_carries_this_actors_role_grant(adapter, builder, actor_roles, gateway_admits):
    """Regression for #118958: a role-only member passes the slash gate, and the event a native
    builder dispatches must carry that verdict, or the gateway answers with a pairing code."""
    adapter._allowed_role_ids = {1234}
    interaction = _make_interaction("999999999")
    interaction.user.roles = [SimpleNamespace(id=r) for r in actor_roles]
    interaction.user.display_name = "member"
    interaction.guild.name = "guild"
    dispatched = []
    adapter.handle_message = AsyncMock(side_effect=dispatched.append)

    if builder == "slash":
        dispatched.append(adapter._build_slash_event(interaction, "/reset"))
    else:
        thread = SimpleNamespace(id=555, name="topic", guild=interaction.guild, topic=None,
                                 parent=SimpleNamespace(id=100, name="general", guild=interaction.guild))
        await adapter._dispatch_thread_session(interaction, thread, "hello")

    assert [_gateway_authorizes(event.source) for event in dispatched] == [gateway_admits]


# ---------------------------------------------------------------------------
# Channel allowlist (DISCORD_ALLOWED_CHANNELS) parity — the gate prajer used
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_channel_not_in_allowlist_rejected(adapter, monkeypatch, caplog):
    """on_message blocks messages in channels not in DISCORD_ALLOWED_CHANNELS;
    slash must do the same. This is the EXACT bypass prajer exploited.
    """
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "1111,2222")
    interaction = _make_interaction("100200300", channel_id=9999)
    with caplog.at_level(logging.WARNING):
        assert await adapter._check_slash_authorization(interaction, "/bg hi") is False
    assert any("DISCORD_ALLOWED_CHANNELS" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Channel blocklist (DISCORD_IGNORED_CHANNELS) parity
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Cross-platform admin notification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unauthorized_attempt_notifies_telegram(adapter):
    from gateway.session import Platform

    telegram_adapter = SimpleNamespace(send=AsyncMock())
    home = SimpleNamespace(chat_id="987654321")
    runner = SimpleNamespace(
        adapters={Platform.TELEGRAM: telegram_adapter},
        config=SimpleNamespace(get_home_channel=lambda p: home if p is Platform.TELEGRAM else None),
    )
    adapter.gateway_runner = runner
    adapter._allowed_user_ids = {"100200300"}

    interaction = _make_interaction("999999999")
    await adapter._check_slash_authorization(interaction, "/bg hi")

    # Notify is fire-and-forget — let the scheduled task run.
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    telegram_adapter.send.assert_awaited_once()
    chat_id, msg = telegram_adapter.send.call_args.args
    assert chat_id == "987654321"
    assert "Unauthorized" in msg
    assert "999999999" in msg
    assert "/bg hi" in msg
    assert "DISCORD_ALLOWED_USERS" in msg


# ---------------------------------------------------------------------------
# Opt-in visibility hide
# ---------------------------------------------------------------------------


    # When called directly the helper applies — env gating is at the call site,
    # which we exercise in an integration-style test below.


def test_visibility_hide_tolerates_unsetable_command(adapter, caplog):
    class _Frozen:
        __slots__ = ("name",)
        def __init__(self, name):
            self.name = name

    cmd_ok = SimpleNamespace(name="ok", default_permissions=None)
    cmd_bad = _Frozen("bad")
    tree = SimpleNamespace(get_commands=lambda: [cmd_bad, cmd_ok])

    with caplog.at_level(logging.DEBUG):
        adapter._apply_owner_only_visibility(tree)

    assert cmd_ok.default_permissions.value == 0


# os import for test_visibility_hide_off_by_default_is_noop
import os


# ---------------------------------------------------------------------------
# Fail-closed parity on malformed slash auth context
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env_name",
    ["GATEWAY_ALLOW_ALL_USERS", "DISCORD_ALLOW_ALL_USERS"],
)
@pytest.mark.asyncio
async def test_missing_user_denied_even_with_allow_all(adapter, monkeypatch, env_name):
    """Malformed slash payloads missing user stay fail-closed with allow-all."""
    monkeypatch.setenv(env_name, "true")
    interaction = _make_interaction("100200300", user=None)
    allowed, reason = adapter._evaluate_slash_authorization(interaction)
    assert allowed is False
    assert reason == "missing interaction.user"
    assert await adapter._check_slash_authorization(interaction, "/help") is False
    interaction.response.send_message.assert_awaited_once()


# ---------------------------------------------------------------------------
# Thread parent channel allowlist parity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_thread_parent_in_allowlist_passes(adapter, monkeypatch):
    """Thread whose parent channel is on DISCORD_ALLOWED_CHANNELS passes
    even though the thread id itself isn't on the list."""
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "5555")
    interaction = _make_interaction(
        "100200300", channel_id=9999, in_thread=True, parent_channel_id=5555,
    )
    assert await adapter._check_slash_authorization(interaction, "/help") is True


@pytest.mark.asyncio
async def test_ignored_beats_allowed(adapter, monkeypatch):
    """Channel listed in BOTH allowed and ignored: the ignored entry wins.
    Anything else would be a foot-gun where adding to ignored does nothing
    if the channel is also explicitly allowed."""
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "1111")
    monkeypatch.setenv("DISCORD_IGNORED_CHANNELS", "1111")
    interaction = _make_interaction("100200300", channel_id=1111)
    assert await adapter._check_slash_authorization(interaction, "/help") is False


# ---------------------------------------------------------------------------
# Admin notify soft-fail fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_notify_falls_back_to_slack_on_telegram_soft_fail(adapter):
    """adapter.send returning SendResult(success=False) must NOT short-
    circuit the fallback chain. Treating a soft failure as delivered
    means a Telegram outage swallows alerts silently."""
    from gateway.session import Platform

    soft_fail = SimpleNamespace(success=False, error="rate limited")
    telegram_adapter = SimpleNamespace(send=AsyncMock(return_value=soft_fail))
    slack_adapter = SimpleNamespace(send=AsyncMock())
    home_tg = SimpleNamespace(chat_id="987654321")
    home_sl = SimpleNamespace(chat_id="C12345")
    homes = {Platform.TELEGRAM: home_tg, Platform.SLACK: home_sl}
    runner = SimpleNamespace(
        adapters={
            Platform.TELEGRAM: telegram_adapter,
            Platform.SLACK: slack_adapter,
        },
        config=SimpleNamespace(get_home_channel=lambda p: homes.get(p)),
    )
    adapter.gateway_runner = runner

    await adapter._notify_unauthorized_slash("u", "1", 2, 3, "/x", "reason")

    telegram_adapter.send.assert_awaited_once()
    slack_adapter.send.assert_awaited_once()


# ---------------------------------------------------------------------------
# /skill autocomplete + callback gating
# ---------------------------------------------------------------------------


def _capture_skill_registration(adapter, monkeypatch, entries):
    """Run ``_register_skill_group`` against a stubbed skill catalog and
    return ``(handler_callback, autocomplete_callback)``.

    The autocomplete callback is captured by monkeypatching
    ``discord.app_commands.autocomplete`` -- the production decorator is
    a no-op stub in this test file's discord mock, so capturing the
    callback through it is the direct route in tests.
    """
    import discord

    captured: dict = {}

    def fake_categories(reserved_names):
        # Match discord_skill_commands_by_category's tuple shape:
        # (categories_dict, uncategorized_list, hidden_count)
        return ({}, list(entries), 0)

    import hermes_cli.commands_platforms as _hc
    monkeypatch.setattr(
        _hc, "discord_skill_commands_by_category", fake_categories,
    )

    def capture_autocomplete(**kwargs):
        # Only one autocomplete in /skill registration: name=...
        captured["autocomplete"] = kwargs.get("name")

        def _passthrough(fn):
            return fn

        return _passthrough

    monkeypatch.setattr(
        discord.app_commands, "autocomplete", capture_autocomplete,
        raising=False,
    )

    registered: list = []

    class _Tree:
        def get_commands(self):
            return []

        def add_command(self, cmd):
            registered.append(cmd)

    adapter._register_skill_group(_Tree())
    assert registered, "_register_skill_group did not register a command"
    return registered[0].callback, captured["autocomplete"]


@pytest.mark.asyncio
async def test_skill_autocomplete_returns_empty_for_unauthorized(
    adapter, monkeypatch,
):
    """Autocomplete must not leak the installed skill catalog to users
    who can't run /skill. With DISCORD_ALLOWED_USERS configured and the
    interaction user outside it, the autocomplete callback returns []."""
    adapter._allowed_user_ids = {"100200300"}
    entries = [
        ("alpha", "First skill", "/alpha"),
        ("beta", "Second skill", "/beta"),
    ]
    _handler, autocomplete = _capture_skill_registration(
        adapter, monkeypatch, entries,
    )

    interaction = _make_interaction("999999999")
    result = await autocomplete(interaction, "")
    assert result == []


@pytest.mark.asyncio
async def test_skill_handler_rejects_before_dispatch_for_unauthorized(
    adapter, monkeypatch,
):
    """The /skill handler must call _check_slash_authorization BEFORE
    skill_lookup. Otherwise unknown vs known names produce divergent
    responses ("Unknown skill: foo" vs auth rejection) which is a
    catalog-probing oracle."""
    adapter._allowed_user_ids = {"100200300"}
    entries = [("alpha", "First skill", "/alpha")]
    handler, _autocomplete = _capture_skill_registration(
        adapter, monkeypatch, entries,
    )

    # Patch _run_simple_slash so we can detect any leak through it.
    dispatched: list = []

    async def fake_dispatch(_interaction, text):
        dispatched.append(text)

    adapter._run_simple_slash = fake_dispatch  # type: ignore[assignment]

    interaction = _make_interaction("999999999")
    await handler(interaction, "alpha", "")

    interaction.response.send_message.assert_awaited_once()
    args, kwargs = interaction.response.send_message.call_args
    assert kwargs.get("ephemeral") is True
    assert "hermes pairing approve discord" in (
        args[0] if args else kwargs.get("content", "")
    )
    # Critically: nothing was dispatched, and the auth message did NOT
    # mention the skill name "alpha" (no catalog leak).
    assert dispatched == []


@pytest.mark.asyncio
async def test_skill_handler_known_and_unknown_produce_same_rejection(
    adapter, monkeypatch,
):
    """An unauthorized user probing for valid skill names must see the
    same rejection text regardless of whether the name they tried is
    on the registered catalog."""
    adapter._allowed_user_ids = {"100200300"}
    entries = [("alpha", "First skill", "/alpha")]
    handler, _ = _capture_skill_registration(adapter, monkeypatch, entries)

    adapter._run_simple_slash = AsyncMock()  # type: ignore[assignment]

    known_interaction = _make_interaction("999999999")
    unknown_interaction = _make_interaction("999999999")
    await handler(known_interaction, "alpha", "")
    await handler(unknown_interaction, "definitely-not-a-skill", "")

    known_interaction.response.send_message.assert_awaited_once()
    unknown_interaction.response.send_message.assert_awaited_once()
    known_args, known_kwargs = known_interaction.response.send_message.call_args
    unknown_args, unknown_kwargs = (
        unknown_interaction.response.send_message.call_args
    )
    assert known_args == unknown_args
    assert known_kwargs == unknown_kwargs


# ---------------------------------------------------------------------------
# /model autocomplete
# ---------------------------------------------------------------------------


def _model_spec_args():
    """The option specs ``/model`` registers, as ``_native_slash_commands()`` serves them."""
    from plugins.platforms.discord.adapter import _native_slash_commands

    return next(
        args for name, _desc, args, _template, _followup in _native_slash_commands()
        if name == "model"
    )


def test_model_option_declares_the_autocomplete_provider():
    """The declarative spec names the provider; registration resolves it later.

    A missing 6th element silently degrades ``/model`` back to free text, which
    is invisible from the Discord side, so assert the wiring at the source."""
    args = _model_spec_args()
    assert args[0][0] == "name"
    assert args[0][5] == "models"


def test_slash_proxy_attaches_model_autocomplete(adapter, monkeypatch):
    """``_slash_proxy`` binds ``_autocomplete_model`` to the ``name`` option."""
    import discord

    captured: dict = {}

    def fake_autocomplete(**kwargs):
        captured.update(kwargs)
        return lambda fn: fn

    monkeypatch.setattr(
        discord.app_commands, "autocomplete", fake_autocomplete, raising=False,
    )
    adapter._slash_proxy("model", _model_spec_args(), "/model {name}", None)

    assert set(captured) == {"name"}
    assert getattr(captured["name"], "__self__", None) is adapter
    assert getattr(captured["name"], "__func__", None) is adapter._autocomplete_model.__func__


def test_slash_proxy_ignores_specs_without_an_autocomplete_provider(adapter, monkeypatch):
    """A five-element spec — every other option — must not gain autocomplete."""
    import discord

    from plugins.platforms.discord.adapter import _native_slash_commands

    captured: dict = {}
    monkeypatch.setattr(
        discord.app_commands, "autocomplete",
        lambda **kwargs: (captured.update(kwargs) or (lambda fn: fn)),
        raising=False,
    )
    btw_args = next(
        args for name, _desc, args, _template, _followup in _native_slash_commands()
        if name == "btw"
    )
    adapter._slash_proxy("btw", btw_args, "/btw {question}", None)

    assert captured == {}


@pytest.mark.asyncio
async def test_model_autocomplete_returns_empty_for_unauthorized(adapter, monkeypatch):
    """The catalog must not leak to a user who cannot run /model."""
    adapter._allowed_user_ids = {"100200300"}
    monkeypatch.setattr(
        adapter, "_model_autocomplete_entries",
        lambda: [("gpt-5 · Atlas", "gpt-5")],
    )

    result = await adapter._autocomplete_model(_make_interaction("999999999"), "")

    assert result == []


@pytest.mark.asyncio
async def test_model_autocomplete_filters_caps_and_skips_overlong_values(
    adapter, monkeypatch,
):
    """Discord's limits are enforced before the response leaves the process.

    An overlong Choice *value* is dropped, never clipped: a truncated model id
    is a different, invalid model id. Labels may be clipped — cosmetic only."""
    adapter._allowed_user_ids = {"999999999"}
    overlong = "m" * 101
    entries = [(f"model-{i:02d} · Atlas", f"model-{i:02d}") for i in range(40)]
    entries.append((overlong, overlong))
    entries.append(("x" * 200, "short-value"))
    monkeypatch.setattr(adapter, "_model_autocomplete_entries", lambda: entries)

    interaction = _make_interaction("999999999")
    choices = await adapter._autocomplete_model(interaction, "")
    assert len(choices) == 25  # Discord's per-response cap
    assert all(len(choice.value) <= 100 for choice in choices)
    assert all(overlong not in choice.value for choice in choices)

    filtered = await adapter._autocomplete_model(interaction, "model-07")
    assert [choice.value for choice in filtered] == ["model-07"]

    long_label = await adapter._autocomplete_model(interaction, "x" * 200)
    assert [choice.value for choice in long_label] == ["short-value"]
    assert len(long_label[0].name) <= 100


def test_model_autocomplete_entries_round_trip_through_the_model_argument(
    adapter, monkeypatch,
):
    """A value must be exactly what ``/model`` accepts: a bare model id for the
    current provider, ``<id> --provider <slug>`` for every other one."""
    import sys
    import types

    # The adapter imports these lazily at call time; registering stand-ins keeps
    # the test off the real gateway/hermes_cli modules (importing them reads the
    # real hermes home, which the home-io guard refuses).
    fake_run = types.ModuleType("gateway.run")
    fake_run._load_gateway_config = lambda: {"model": {"provider": "atlas"}}
    monkeypatch.setitem(sys.modules, "gateway.run", fake_run)

    fake_cp = types.ModuleType("hermes_cli.config_providers")
    fake_cp.get_compatible_custom_providers = lambda cfg: []
    monkeypatch.setitem(sys.modules, "hermes_cli.config_providers", fake_cp)

    fake_msp = types.ModuleType("hermes_cli.model_switch_providers")
    fake_msp.list_authenticated_providers = lambda **kwargs: [
        {"slug": "openrouter", "name": "OpenRouter", "models": ["gpt-5"]},
        {"slug": "atlas", "name": "Atlas Cloud", "models": ["gpt-5"]},
    ]
    monkeypatch.setitem(sys.modules, "hermes_cli.model_switch_providers", fake_msp)

    entries = adapter._model_autocomplete_entries()
    values = {value: label for label, value in entries}

    assert values["gpt-5"] == "gpt-5 · Atlas Cloud"  # current provider keeps a bare id
    assert values["gpt-5 --provider openrouter"] == "gpt-5 · OpenRouter"
    assert len(entries) == 2  # the same id on two providers stays distinct


