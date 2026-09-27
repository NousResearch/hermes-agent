"""Matching and real profile-scope contracts for channel-name routing (#109676)."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.channel_directory import build_gateway_channel_directories
from gateway.channel_matching import ChannelNames, get_channel_override
from gateway.channel_names import ChannelNameDirectory, clear_adapter_directory, name_resolver_for_source
from gateway.config import ChannelOverride, GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.profile_routing import match_profile_route, parse_profile_routes
from gateway.session import SessionSource, build_session_key
from gateway.session_identity import IdentityUnresolved, resolve_identity


@pytest.mark.parametrize("case", [
    "parent-order", "route-order", "id-fast-path", "thread-scope", "scope-specificity",
    "name-before-pattern", "pattern-order", "invalid-pattern", "literal-metacharacters",
    "guild-and-channel", "unknown", "same-thread-other-chat", "same-chat-other-scope",
    "override-id-fast-path", "anchored-pattern", "dot-star-literal", "whatsapp-id",
    "discord-forum", "slack-composite", "telegram-topic",
    "unknown-scope", "unscoped-alias",
])
def test_channel_name_matching_contract(case, caplog):
    if case == "whatsapp-id":
        routes = parse_profile_routes([{"platform": "whatsapp", "profile": "phone", "chat_id": "+15551234567"}])
        def no_names():
            pytest.fail("WhatsApp identity aliases must match in the ID phase")
        assert match_profile_route(routes, "whatsapp", chat_id="15551234567@s.whatsapp.net",
                                   name_resolver=no_names) is routes[0]
        return
    snapshot = ChannelNameDirectory([
        {"id": "111", "name": "parent"},
        {"id": "222", "name": "thread"},
        {"id": "333:7", "chat_id": "333", "thread_id": "7", "thread_name": "topic"},
        {"id": "444:7", "chat_id": "444", "thread_id": "7", "thread_name": "other"},
        {"id": "222", "scope_id": "other-guild", "name": "elsewhere"},
    ])
    source = SessionSource(Platform.DISCORD, "222", thread_id="222", parent_chat_id="111")
    resolve = lambda: snapshot.resolve(source)
    if case in {"route-order", "id-fast-path", "scope-specificity", "guild-and-channel"}:
        rules = [
            {"profile": "generic", "chat_id": "^thr"},
            {"profile": "specific", "chat_id": "222"},
        ]
        if case == "id-fast-path":
            rules[0]["chat_id"] = "999"
            def resolve():
                pytest.fail("An ID route hit must not resolve names")
        elif case == "scope-specificity":
            rules = [{"profile": "generic", "guild_id": "guild"}, {"profile": "specific", "thread_id": "thread"}]
            resolve = lambda: ChannelNames(thread=("thread",))
        elif case == "guild-and-channel":
            rules = [{"profile": "generic", "guild_id": "Wrong", "chat_id": "thread"},
                     {"profile": "specific", "guild_id": "Right", "chat_id": "thread"}]
            resolve = lambda: ChannelNames(chat=("thread",), guild=("Right",))
        routes = parse_profile_routes([{"platform": "discord", **rule} for rule in rules])
        matched = match_profile_route(routes, "discord", chat_id="222", thread_id="222", guild_id="guild",
                                      name_resolver=resolve)
        assert matched and matched.profile == "specific"
        return

    keys, expected = ["parent", "thread", "topic"], "thread"
    if case == "discord-forum":
        from gateway.channel_directory import _build_discord
        parent = SimpleNamespace(id=111, name="parent")
        thread = SimpleNamespace(id=222, name="thread", parent_id=111, parent=parent)
        guild = SimpleNamespace(id=10, name="Guild", text_channels=[], forum_channels=[parent], threads=[thread])
        snapshot = ChannelNameDirectory(_build_discord(SimpleNamespace(_client=SimpleNamespace(guilds=[guild]))))
        source.scope_id = "10"
        assert snapshot.resolve(source).parent == (parent.name,)
        assert snapshot.resolve(source).guild == (guild.name,)
    elif case == "slack-composite":
        from gateway.channel_directory import _entries_from_origins, resolve_channel_name
        origin = {"chat_id": "opaque:chat", "thread_id": "ts:thread", "scope_id": "workspace",
                  "chat_name": "DO NOT PARSE / display", "channel_name": "channel", "thread_name": "topic"}
        entries = _entries_from_origins("slack", "test", lambda: [(origin, "group")])
        snapshot = ChannelNameDirectory(entries)
        with patch("gateway.channel_directory.load_directory", return_value={"platforms": {"slack": entries}}):
            assert resolve_channel_name("slack", "unknown/room") is None
        source = SessionSource(Platform.SLACK, origin["chat_id"], thread_id=origin["thread_id"], scope_id="workspace")
        expected = "topic"
    elif case == "telegram-topic":
        from gateway.platforms.base import MessageType
        from plugins.platforms.telegram.adapter import TelegramAdapter
        adapter = TelegramAdapter(PlatformConfig(extra={"group_topics": [
            {"chat_id": -100100, "topics": [{"thread_id": 7, "name": "topic"}]}]}))
        message = SimpleNamespace(message_id=1, date=datetime.now(timezone.utc),
                                  chat=SimpleNamespace(id=-100100, type="supergroup", title="channel", is_forum=True),
                                  from_user=SimpleNamespace(id=42, full_name="User", is_bot=False),
                                  message_thread_id=7, is_topic_message=True, text="DO NOT INFER A TITLE FROM THIS",
                                  entities=[], caption=None, caption_entities=[], reply_to_message=None,
                                  quote=None, forum_topic_created=None)
        source = adapter._build_message_event(message, MessageType.TEXT).source
        assert source.thread_name == "topic" and source.channel_name == "channel"
        resolve, expected = name_resolver_for_source(source), "topic"
    elif case == "thread-scope":
        source.chat_id, source.thread_id, source.parent_chat_id = "333", "7", None
        expected = "topic"
    elif case == "same-thread-other-chat":
        source.chat_id, source.thread_id, source.parent_chat_id = "444", "7", None
        expected = None
    elif case == "same-chat-other-scope":
        source.scope_id, source.parent_chat_id = "other-guild", None
        expected = None
    elif case == "unknown-scope":
        source.scope_id, source.parent_chat_id = "unknown-guild", None
        expected = None
    elif case == "unscoped-alias":
        snapshot = ChannelNameDirectory([
            {"id": "222:7", "scope_id": "workspace", "chat_id": "222", "thread_id": "7", "chat_name": "channel"},
            {"id": "222", "name": "alias", "alias": "alias"},
        ])
        source.scope_id = "workspace"
        keys, expected = ["alias"], "alias"
    elif case == "name-before-pattern":
        keys = ["^thr", "thread"]
    elif case == "pattern-order":
        keys, expected = ["^th", "^thread$"], "^th"
    elif case == "invalid-pattern":
        keys, expected = ["^thread-("], None
    elif case == "override-id-fast-path":
        keys, expected = ["^th", "parent", "111"], "111"
        def resolve():
            pytest.fail("Even a parent ID hit must not resolve names")
    elif case == "anchored-pattern":
        keys, expected = ["hread$"], None
    elif case == "dot-star-literal":
        keys, expected = ["thr.*"], None
    elif case == "literal-metacharacters":
        keys, expected = ["^work", "work+(literal)"], "work+(literal)"
        resolve = lambda: ChannelNames(chat=("work+(literal)",))
    elif case == "unknown":
        source.chat_id, source.thread_id, source.parent_chat_id = "missing", None, None
        expected = None
    config = GatewayConfig(platforms={source.platform: PlatformConfig(
        channel_overrides={key: ChannelOverride(model=key) for key in keys},
    )})
    result = get_channel_override(config, source.platform, source.chat_id, thread_id=source.thread_id,
                                  parent_id=source.parent_chat_id, name_resolver=resolve)
    assert (result.model if result else None) == expected
    if case == "invalid-pattern":
        assert sum("not a valid regex" in record.message for record in caplog.records) == 1
        assert get_channel_override(config, Platform.DISCORD, "other",
                                    name_resolver=lambda: ChannelNames(chat=(keys[0],))).model == keys[0]


class _Transport(BasePlatformAdapter):
    async def list_channels(self):
        return self.entries


_Transport.__abstractmethods__ = frozenset()


@pytest.mark.parametrize("platform", [Platform.DISCORD, Platform.SLACK, Platform.TELEGRAM,
                                     Platform.WHATSAPP, Platform.MATTERMOST])
def test_receiving_bot_names_survive_profile_switches(platform, tmp_path, monkeypatch):
    """Real config, builders, ingress identity, model and prompt resolution across A→B→A."""
    from gateway.run import GatewayRunner, _profile_runtime_scope

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    secondary = home / "profiles" / "other"
    secondary.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    for directory, alias in ((home, "alpha"), (secondary, "beta")):
        (directory / "config.yaml").write_text("{}\n", encoding="utf-8")
        (directory / ".env").write_text("", encoding="utf-8")
        (directory / "channel_aliases.json").write_text(
            json.dumps({platform.value: {"same-id": alias}}), encoding="utf-8")
    config = GatewayConfig.from_dict({
        "multiplex_profiles": True,
        "profile_routes": [
            {"platform": platform.value, "chat_id": "alpha", "profile": "other", "user_id": "allowed"},
            {"platform": platform.value, "chat_id": "beta", "profile": "default", "bot_profile": "other"},
            {"platform": platform.value, "chat_id": "^ghost", "profile": "missing"},
        ],
        "platforms": {platform.value: {"enabled": True, "channel_overrides": {
            "alpha": {"model": "alpha-model", "provider": "openrouter", "system_prompt": "alpha-prompt"},
            "beta": {"model": "beta-model", "provider": "anthropic", "system_prompt": "beta-prompt"},
        }}},
    })
    runner = object.__new__(GatewayRunner)
    runner.config, runner._primary_profile_name = config, "default"
    primary, other = (_Transport.__new__(_Transport) for _ in range(2))
    for adapter, owner, label in ((primary, None, "first"), (other, "other", "second")):
        adapter.platform, adapter.gateway_runner = platform, runner
        adapter.config = config.platforms[platform]
        adapter.set_owner_profile(owner)
        adapter.entries = [{"id": "same-id", "chat_id": "same-id", "name": label, "scope_id": "guild"}]
    runner.adapters, runner._profile_adapters = {platform: primary}, {"other": {platform: other}}
    asyncio.run(build_gateway_channel_directories(runner))
    assert (home / "channel_directory.json").exists()
    assert (secondary / "channel_directory.json").exists()

    def provider_runtime(provider, **kwargs):
        return {"provider": provider, "api_key": "test-only", "api_mode": "chat_completions"}

    for adapter, alias, target, auth_home in ((primary, "alpha", "other", home),
                                            (other, "beta", "default", secondary),
                                            (primary, "alpha", "other", home)):
        source = adapter.build_source(chat_id="same-id", user_id="allowed", scope_id="guild")
        identity = resolve_identity(source, runner=runner)
        assert identity.runtime_profile == target
        assert identity.authorization_home == auth_home
        assert identity.adapter() is adapter
        assert runner._delivery_adapter_for(source) is adapter
        with _profile_runtime_scope(identity.runtime_home), \
                patch("gateway.run._resolve_runtime_agent_kwargs", return_value=provider_runtime("global")), \
                patch("gateway.run._resolve_runtime_agent_kwargs_for_provider", side_effect=provider_runtime):
            model, runtime = runner._resolve_session_agent_runtime(source=source, user_config={"model": {"default": "global"}})
            assert model == f"{alias}-model"
            assert runtime["provider"] == ("openrouter" if alias == "alpha" else "anthropic")
            assert runner._get_system_prompt_for_channel(platform, "same-id", source=source) == f"{alias}-prompt"
            assert runner._channel_override_for(source).model == model
            session_key = build_session_key(source)
            conversation = runner._session_state(session_key).conversation
            conversation.model_override = {"model": "session-model", "provider": "chosen",
                                           "api_key": "test-only", "credential_pool": object()}
            try:
                selected, runtime = runner._resolve_session_agent_runtime(source=source, session_key=session_key,
                                                                          user_config={"model": {"default": "global"}})
                assert (selected, runtime["provider"]) == ("session-model", "chosen")
            finally:
                conversation.model_override = None
        assert build_session_key(SessionSource.from_dict(source.to_dict())) == build_session_key(source)

    denied = primary.build_source(chat_id="same-id", user_id="someone-else", scope_id="guild")
    assert denied.profile is None
    ghost = primary.build_source(chat_id="new-id", chat_name="ghost-room", user_id="allowed")
    with pytest.raises(IdentityUnresolved):
        resolve_identity(ghost, runner=runner)

    primary.entries[0]["name"] = "renamed"
    (home / "channel_aliases.json").write_text("{}", encoding="utf-8")
    asyncio.run(build_gateway_channel_directories(runner))
    source = primary.build_source(chat_id="same-id", scope_id="guild")
    assert name_resolver_for_source(source)().chat == ("renamed",)
    primary._routing_channel_directory.expires_at = 0
    # A turn keeps the same names, while the next message observes expiration.
    assert name_resolver_for_source(source)().chat == ("renamed",)
    assert name_resolver_for_source(primary.build_source(chat_id="same-id", scope_id="guild"))().chat == ()
    asyncio.run(build_gateway_channel_directories(runner))
    clear_adapter_directory(primary)
    assert name_resolver_for_source(primary.build_source(chat_id="same-id", scope_id="guild"))().chat == ()

    async def disconnect_during_refresh():
        started, release = asyncio.Event(), asyncio.Event()
        async def slow_channels():
            started.set()
            await release.wait()
            return primary.entries
        with patch.object(primary, "list_channels", slow_channels):
            task = asyncio.create_task(build_gateway_channel_directories(runner))
            await started.wait()
            clear_adapter_directory(primary)
            release.set()
            await task
        assert primary._routing_channel_directory is None

    asyncio.run(disconnect_during_refresh())
