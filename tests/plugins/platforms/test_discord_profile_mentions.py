"""Discord ``$<profile>`` mention dispatch: each served profile opts itself in.

Feature request #159776: one shared Discord bot under ``gateway.multiplex_profiles`` hands a message
that names a profile to that profile's own session for one normal turn. A profile is reachable only
when its OWN ``config.yaml`` sets ``discord.mentionable: true`` (``discord.mention_handle`` picks a
handle other than its name) and this gateway serves it. The six acceptance cases from the issue are
the matrix below; the identity tests run the copy through the gateway's real ``resolve_identity``;
the live tests flip the flag in a served profile's config.yaml and prove the running gateway's
hot-serve rescan (and, for the bot's own profile, its live config refresh) applies it.
"""

import itertools
import os
import sys
import time
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import hermes_yaml as yaml
import pytest


def _ensure_discord_mock():
    """Install a mock discord module when discord.py isn't available."""
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return
    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
    discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, secondary=2, danger=3, green=1, grey=2, blurple=2, red=3)
    discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4, purple=lambda: 5)
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.Object = lambda *, id: SimpleNamespace(id=id)
    discord_mod.Message = type("Message", (), {})
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod
    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

import plugins.platforms.discord.adapter as discord_platform
import plugins.platforms.discord.adapter_profile_mentions as profile_mentions
from gateway.config import GatewayConfig, Platform, PlatformConfig, load_gateway_config
from gateway.platforms.base import BasePlatformAdapter
from plugins.platforms.discord.adapter import DiscordAdapter

_WRITES = itertools.count()


def _write_config(home, doc):
    """Write *home*'s config.yaml with a distinct, settled mtime (two rewrites never share a stat)."""
    home.mkdir(parents=True, exist_ok=True)
    path = home / "config.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    stamp = time.time_ns() - 10_000_000_000 + next(_WRITES) * 1_000_000
    os.utime(path, ns=(stamp, stamp))


def _profile_config(monkeypatch, home, discord=None):
    """*home*'s GatewayConfig through the real loader, from a config.yaml whose own ``discord:``
    block is *discord* (None: no block at all)."""
    _write_config(home, {"model": {"default": "m"}, **({"discord": discord} if discord is not None else {})})
    with monkeypatch.context() as m:
        m.setenv("HERMES_HOME", str(home))
        return load_gateway_config()


class _Runner:
    """The bookkeeping a multiplexed runner keeps, as the plugin reads it."""

    def __init__(self, served, configs, primary="default"):
        self._served = list(served)
        self._profile_configs = dict(configs)
        self._primary_profile_name = primary
        self.adapters = {}
        self._profile_adapters = {}
        self.config = GatewayConfig(multiplex_profiles=True)

    def served_profile_names(self):
        return list(self._served)


class FakeDMChannel:
    def __init__(self, channel_id: int = 1):
        self.id = channel_id
        self.name = "dm"


class FakeThread:
    pass


class FakeTextChannel:
    def __init__(self, channel_id: int = 555):
        self.id = channel_id
        self.name = "general"
        self.guild = SimpleNamespace(name="Hermes Server", id=777)
        self.topic = None

    def history(self, *, limit, before, after=None, oldest_first=None):
        async def _iter():
            return
            yield
        return _iter()


@pytest.fixture
def served(tmp_path, monkeypatch):
    """Served profiles ``agent-one`` and ``agent-two`` opted in through their own config.yaml;
    ``quiet`` is served but did not opt in. The bot belongs to ``default``, which did not either."""
    profiles = tmp_path / "hh" / "profiles"
    configs = {
        "agent-one": _profile_config(monkeypatch, profiles / "agent-one", {"mentionable": True}),
        "agent-two": _profile_config(monkeypatch, profiles / "agent-two", {"mentionable": True}),
        "quiet": _profile_config(monkeypatch, profiles / "quiet"),
    }
    return _Runner(["default", "agent-one", "agent-two", "quiet"], configs)


@pytest.fixture
def adapter(monkeypatch, served):
    monkeypatch.setattr(discord_platform.discord, "DMChannel", FakeDMChannel, raising=False)
    monkeypatch.setattr(discord_platform.discord, "Thread", FakeThread, raising=False)
    for var in ("DISCORD_REQUIRE_MENTION", "DISCORD_FREE_RESPONSE_CHANNELS", "DISCORD_AUTO_THREAD",
                "DISCORD_ALLOWED_CHANNELS", "DISCORD_IGNORED_CHANNELS", "DISCORD_HISTORY_BACKFILL"):
        monkeypatch.delenv(var, raising=False)
    # A require_mention server channel: a profile handle must be enough to address the bot.
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    monkeypatch.setenv("DISCORD_HISTORY_BACKFILL", "false")
    monkeypatch.setattr(profile_mentions, "is_multiplex_active", lambda: True)
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake-token", extra={}))
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=999))
    adapter._text_batch_delay_seconds = 0
    adapter.handle_message = AsyncMock()
    adapter.gateway_runner = served
    served.adapters[Platform.DISCORD] = adapter
    return adapter


def _message(content: str, *, attachments=()):
    return SimpleNamespace(
        id=123, content=content, mentions=[], attachments=list(attachments), reference=None,
        created_at=datetime.now(UTC), channel=FakeTextChannel(),
        author=SimpleNamespace(id=42, display_name="Jezza", name="Jezza"),
        guild=SimpleNamespace(id=777, name="Hermes Server"),
        type=discord_platform.discord.MessageType.default,
    )


async def _dispatched(adapter, content: str):
    adapter.handle_message.reset_mock()
    await adapter._handle_message(_message(content))
    return [call.args[0] for call in adapter.handle_message.await_args_list]


# --- the issue's acceptance matrix -------------------------------------------------------------


@pytest.mark.asyncio
async def test_leading_mention_runs_one_turn_in_that_profile(adapter):
    """Case 1: ``$agent-one what's the status?`` -> only agent-one, handle stripped, labelled."""
    events = await _dispatched(adapter, "$agent-one what's the status?")

    assert [e.source.profile for e in events] == ["agent-one"]
    assert events[0].text == "what's the status?"
    assert events[0].source.profile_mention is True
    assert events[0].source.chat_id == "555" and events[0].source.user_id == "42"


@pytest.mark.asyncio
async def test_mention_mid_message_still_routes(adapter):
    """Case 2: detection is not anchored to the start of the message."""
    events = await _dispatched(adapter, "hey $agent-one, any update?")

    assert [e.source.profile for e in events] == ["agent-one"]
    assert events[0].text == "hey, any update?"


@pytest.mark.asyncio
async def test_several_mentions_each_get_an_independent_turn(adapter):
    """Case 3: both profiles get the same prompt as separate events; nothing is shared between
    the copies (a batch merge into one must never edit the other)."""
    events = await _dispatched(adapter, "$agent-one $agent-two compare your dependency staleness")

    assert [e.source.profile for e in events] == ["agent-one", "agent-two"]
    assert {e.text for e in events} == {"compare your dependency staleness"}
    one, two = events
    assert one.source is not two.source and one.media_urls is not two.media_urls
    assert one.metadata is not two.metadata


@pytest.mark.asyncio
async def test_message_without_a_mention_is_untouched(adapter, monkeypatch):
    """Case 4: no handle -> the normal require_mention gate (dropped here), and in a free-response
    channel the event goes through exactly as before, under the channel's own profile."""
    assert await _dispatched(adapter, "anyone know if CI is green?") == []

    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    events = await _dispatched(adapter, "anyone know if CI is green?")
    assert len(events) == 1
    assert events[0].text == "anyone know if CI is green?"
    assert events[0].source.profile is None and events[0].source.profile_mention is False


@pytest.mark.asyncio
async def test_unknown_profile_fails_closed_with_a_log_line(adapter, caplog):
    """Case 5: a handle no served profile opted in to is never forwarded as a profile; one INFO
    line says so (counts only, no message content) and the message follows the normal rules."""
    with caplog.at_level("INFO", logger=profile_mentions.__name__):
        events = await _dispatched(adapter, "$not-a-real-profile ping")

    assert events == []  # require_mention channel, bot not addressed
    lines = [r for r in caplog.records if r.name == profile_mentions.__name__ and r.levelname == "INFO"]
    assert len(lines) == 1 and "unknown" in lines[0].getMessage()
    assert "not-a-real-profile" not in lines[0].getMessage()


@pytest.mark.asyncio
async def test_mention_with_no_prompt_is_a_no_op(adapter):
    """Case 6: ``$agent-one`` alone dispatches nothing and does not error."""
    assert await adapter._handle_message(_message("$agent-one")) is True
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_recovered_message_passes_the_same_gate_as_a_live_one(adapter, monkeypatch):
    """Missed-message recovery re-applies the require_mention gate itself: a ``$<profile>``
    message the live path would dispatch must not be skipped there."""
    monkeypatch.setattr(adapter, "_discord_message_admission", lambda message, claim: (True, False))
    monkeypatch.setattr(adapter, "_in_bot_thread", lambda message: False)

    assert await adapter._dispatch_recovered_message(_message("$agent-two rerun the audit")) is True
    (event,) = [call.args[0] for call in adapter.handle_message.await_args_list]
    assert (event.source.profile, event.text) == ("agent-two", "rerun the audit")


# --- the opt-in: each profile's own config, served profiles only ------------------------------


@pytest.mark.asyncio
async def test_feature_is_off_without_multiplexing(adapter, monkeypatch):
    """Without ``gateway.multiplex_profiles`` there is no per-profile session to hand to: the
    handle is ordinary text and the require_mention gate applies."""
    monkeypatch.setattr(profile_mentions, "is_multiplex_active", lambda: False)
    assert await _dispatched(adapter, "$agent-one what's the status?") == []


@pytest.mark.asyncio
async def test_a_served_profile_that_did_not_opt_in_is_unreachable(adapter, served, monkeypatch, tmp_path):
    """Being served is not enough: ``quiet`` has no flag, and a flag that is not a YAML boolean
    (``"true"``, ``"yes"``, ``1``) is not an opt-in. The bot's own config listing other profiles
    grants nothing either: another profile's opt-in is never read from the host's config."""
    adapter.config.extra["profile_mentions"] = ["quiet"]
    adapter.config.extra["mentionable"] = False
    assert await _dispatched(adapter, "$quiet hello") == []
    assert await _dispatched(adapter, "$default hello") == []
    for value in ("true", "yes", 1):
        served._profile_configs["quiet"] = _profile_config(
            monkeypatch, tmp_path / "hh" / "profiles" / "quiet", {"mentionable": value})
        assert await _dispatched(adapter, "$quiet hello") == [], value


@pytest.mark.asyncio
async def test_an_opted_in_profile_this_gateway_does_not_serve_is_unreachable(adapter, served, monkeypatch, tmp_path):
    """A profile whose config opts in but that this gateway does not serve (parked, running its
    own gateway, deleted) is never dispatched; dropping a served profile withdraws its handle."""
    served._profile_configs["ghost"] = _profile_config(
        monkeypatch, tmp_path / "hh" / "profiles" / "ghost", {"mentionable": True})
    assert await _dispatched(adapter, "$ghost status?") == []

    served._served.remove("agent-two")
    assert await _dispatched(adapter, "$agent-two status?") == []
    assert [e.source.profile for e in await _dispatched(adapter, "$agent-one status?")] == ["agent-one"]


def test_custom_handles_collisions_and_invalid_handles(served, monkeypatch, tmp_path):
    """``mention_handle`` replaces the profile name as the handle; matching is case-insensitive;
    a handle two profiles claim reaches neither; an invalid handle makes the profile unreachable;
    ``$5`` and ``a$a1`` are not handles."""
    profiles = tmp_path / "hh" / "profiles"
    served._profile_configs.update({
        "agent-one": _profile_config(monkeypatch, profiles / "agent-one", {"mentionable": True, "mention_handle": "A1"}),
        "agent-two": _profile_config(monkeypatch, profiles / "agent-two", {"mentionable": True, "mention_handle": "dup"}),
        "quiet": _profile_config(monkeypatch, profiles / "quiet", {"mentionable": True, "mention_handle": "dup"}),
        "bad": _profile_config(monkeypatch, profiles / "bad", {"mentionable": True, "mention_handle": "../evil"}),
    })
    served._served.append("bad")
    bot = SimpleNamespace(platform=Platform.DISCORD, config=PlatformConfig(extra={}), gateway_runner=served)

    handles = profile_mentions.mentionable_handles(bot)
    assert handles == {"a1": "agent-one"}

    plan = profile_mentions.plan_mentions("$A1: costs $5, ask a$a1", handles)
    assert plan.profiles == ("agent-one",) and plan.prompt == "costs $5, ask a$a1"


# --- live: flipping the flag in a profile's own config.yaml, no restart ------------------------


@pytest.mark.asyncio
async def test_the_hot_serve_rescan_rereads_a_served_profiles_flag(tmp_path, monkeypatch):
    """End to end through the real reconcile (``reconcile_served_profiles`` ->
    ``_start_one_profile_adapters`` -> ``_load_secondary_profile_config``): ``gamma`` is served
    with no bot of its own. Writing ``discord.mentionable: true`` into ITS config.yaml makes
    ``$gamma`` reachable from the default profile's bot after the next rescan, and turning it off
    makes it unreachable again; nothing is restarted and the bot is never rebuilt."""
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner

    home = tmp_path / ".hermes"
    (home / "profiles").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr(profile_mentions, "is_multiplex_active", lambda: True)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.session_authorities = None
    runner._running = True
    runner._primary_profile_name = "default"
    runner._profile_adapters, runner._profile_failed_platforms, runner._failed_platforms = {}, {}, {}
    runner._agent_cache, runner._agent_cache_lock = {}, None
    runner.pairing_store, runner.pairing_stores = MagicMock(), {}
    runner._adapter_disconnect_timeout_secs = lambda: 0.5
    runner._register_config_hooks = lambda *a, **kw: None
    runner._busy_text_modes_by_profile, runner._busy_input_modes_by_profile = {}, {}
    runner.session_runtime_descriptor = {"state": "ready", "served_profiles": [], "capabilities": []}
    runner.session_ticket_store = None

    async def _no_added(profiles):
        pass

    runner._after_profiles_added = _no_added
    bot = DiscordAdapter(PlatformConfig(enabled=True, token="fake-token", extra={}))
    bot.gateway_runner = runner
    runner.adapters = {Platform.DISCORD: bot}

    def _mentions(text):
        source = bot.build_source(chat_id="555", chat_type="group", user_id="42")
        event = MessageEvent(text=text, source=source, message_id="1")
        return [(e.source.profile, e.text) for e in profile_mentions.mention_events(bot, event)]

    gamma = home / "profiles" / "gamma"
    _write_config(gamma, {"model": {"default": "m"}})
    (gamma / ".env").write_text("", encoding="utf-8")
    with patch("hermes_cli.profiles.get_active_profile_name", return_value="default"):
        await runner._start_secondary_profile_adapters()
        assert runner.served_profile_names() == ["default", "gamma"]
        assert _mentions("$gamma status?") == [(None, "$gamma status?")]

        _write_config(gamma, {"model": {"default": "m"}, "discord": {"mentionable": True}})
        # Not before the rescan: the flag reaches the bot through the reconcile, nothing else.
        assert _mentions("$gamma status?") == [(None, "$gamma status?")]
        result = await runner.reconcile_served_profiles(reason="watcher")
        assert result["rescanned"] == ["gamma"]
        assert _mentions("$gamma status?") == [("gamma", "status?")]

        _write_config(gamma, {"model": {"default": "m"}, "discord": {"mentionable": False}})
        assert (await runner.reconcile_served_profiles(reason="watcher"))["rescanned"] == ["gamma"]
        assert _mentions("$gamma status?") == [(None, "$gamma status?")]
    assert runner.adapters[Platform.DISCORD] is bot


def test_the_bots_own_profile_flag_follows_its_config_yaml(tmp_path, monkeypatch):
    """The profile that owns the bot is never rebuilt by the rescan; its own flag rides the live
    config refresh the allowlists use, so editing its config.yaml applies on the next message."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(profile_mentions, "is_multiplex_active", lambda: True)
    _write_config(tmp_path, {"platforms": {"discord": {"enabled": True}}})
    bot = DiscordAdapter(load_gateway_config().platforms[Platform.DISCORD])
    bot._snapshot_gate_env()
    bot._mark_live_gate_baseline()
    runner = _Runner(["default"], {})
    runner.adapters[Platform.DISCORD] = bot
    bot.gateway_runner = runner
    assert profile_mentions.addresses_profile(bot, "$host hi") is False

    _write_config(tmp_path, {"platforms": {"discord": {"enabled": True}},
                             "discord": {"mentionable": True, "mention_handle": "host"}})
    assert profile_mentions.mentionable_handles(bot) == {"host": "default"}
    assert profile_mentions.addresses_profile(bot, "$host hi") is True

    _write_config(tmp_path, {"platforms": {"discord": {"enabled": True}}})
    assert profile_mentions.addresses_profile(bot, "$host hi") is False


# --- the re-homed copy through the gateway's identity seam --------------------------------------


class _Stub(BasePlatformAdapter):
    pass


_Stub.__abstractmethods__ = frozenset()


@pytest.fixture
def mux(tmp_path, monkeypatch):
    """A multiplexed runner whose default profile owns the Discord bot; ``agent-one`` is served
    (no bot of its own) and opted in through its own config."""
    from gateway.pairing import PairingStore
    from gateway.run import GatewayRunner

    home = tmp_path / "hh"
    agent_one = home / "profiles" / "agent-one"
    agent_one_cfg = _profile_config(monkeypatch, agent_one, {"mentionable": True})
    monkeypatch.setenv("HERMES_HOME", str(home))
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.config.platforms = {Platform.DISCORD: PlatformConfig(enabled=True, extra={})}
    runner.config.profile_routes = []
    runner.pairing_store = PairingStore(profile="default")
    runner.pairing_stores = {}
    runner._primary_profile_name = "default"
    primary = _Stub.__new__(_Stub)
    primary.platform, primary.gateway_runner = Platform.DISCORD, runner
    primary.config = PlatformConfig(enabled=True, extra={})
    primary._pending_messages, primary._active_sessions = {}, {}
    runner.adapters = {Platform.DISCORD: primary}
    runner._profile_adapters = {"agent-one": {}}
    runner._profile_configs = {"agent-one": agent_one_cfg}
    served = [("default", home), ("agent-one", agent_one)]
    runner._served_profile_homes = dict(served)
    monkeypatch.setattr(profile_mentions, "is_multiplex_active", lambda: True)
    with patch("hermes_cli.profiles.profiles_to_serve", return_value=served), \
            patch("hermes_cli.profiles.get_profile_dir",
                  side_effect=lambda n: home if n == "default" else home / "profiles" / n), \
            patch("hermes_cli.profiles.profile_exists", side_effect=lambda n: n in {"default", "agent-one"}):
        yield SimpleNamespace(runner=runner, primary=primary, home=home)


def _copies(mux, text):
    from gateway.platforms.event import MessageEvent

    source = mux.primary.build_source(chat_id="555", chat_type="group", user_id="42", guild_id="777")
    event = MessageEvent(text=text, source=source, message_id="123")
    return profile_mentions.mention_events(mux.primary, event)


def test_the_copy_runs_in_the_named_profile_and_authorizes_as_the_receiving_bot(mux):
    from gateway.session_identity import resolve_identity

    (event,) = _copies(mux, "$agent-one status?")
    identity = resolve_identity(event.source, runner=mux.runner)

    assert identity.runtime_profile == "agent-one"
    assert identity.runtime_home == mux.home / "profiles" / "agent-one"
    assert identity.transport_profile == "default" and identity.authorization_home == mux.home
    assert identity.adapter() is mux.primary
    assert identity.namespace == "agent:agent-one"


def test_reply_label_names_the_profile(mux, monkeypatch):
    """The turn's footer line leads with the answering profile's name (empty for ordinary turns),
    and keeps a configured runtime footer after it."""
    import gateway.runtime_footer as footer
    from gateway.run_turn_persistence import GatewayTurnPersistenceMixin
    from gateway.session_identity import resolve_identity

    (event,) = _copies(mux, "$agent-one status?")
    resolve_identity(event.source, runner=mux.runner)
    plain = mux.primary.build_source(chat_id="555", chat_type="group", user_id="42")
    harness = GatewayTurnPersistenceMixin()
    monkeypatch.setattr(footer, "build_footer_line", lambda **kw: "")

    assert harness._hmwa_runtime_footer_line({"model": "m"}, event.source, 1.0) == "— agent-one"
    assert harness._hmwa_runtime_footer_line({"model": "m"}, plain, 1.0) == ""

    monkeypatch.setattr(footer, "build_footer_line", lambda **kw: "m · 3s")
    assert harness._hmwa_runtime_footer_line({"model": "m"}, event.source, 1.0) == "— agent-one · m · 3s"


def test_rehomed_source_stays_off_the_wire(mux):
    """``profile_mention`` is a transport-local marker: never serialized, so a restored row or a
    relay peer cannot carry or forge it."""
    from gateway.session import SessionSource

    (event,) = _copies(mux, "$agent-one status?")
    wire = event.source.to_dict()
    assert "profile_mention" not in wire
    assert SessionSource.from_dict(wire).profile_mention is False
    assert wire["profile"] == "agent-one"
