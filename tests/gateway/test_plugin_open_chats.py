"""``plugins.entries.<id>.open_chats``: an operator lets one plugin open a chat with
``inject_message(origin=...)`` before any human message there (#148065).

Without the key, the session authority refuses an origin with no committed human turn (PLG-6). With
the chat listed, the plugin turn opens the chat's route and joins its FIFO under the connector
binding a human message there would record. Real GatewayRunner, DiscordAdapter and SessionAuthority.
"""
import asyncio

import pytest
import hermes_yaml as yaml

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from tests.waiting import async_wait_until

PLUGIN = "desk"


def _context(home, open_chats) -> tuple[PluginContext, PluginManager]:
    (home / "config.yaml").write_text(yaml.safe_dump({
        "plugins": {"entries": {PLUGIN: {"allow_gateway_injection": True, "open_chats": open_chats}}},
    }), encoding="utf-8")
    manager = PluginManager(scope_key=str(home))
    return PluginContext(PluginManifest(name=PLUGIN, key=PLUGIN, source="user"), manager), manager


def _publish(runner, manager) -> None:
    runner._gateway_loop = asyncio.get_running_loop()
    runner._running = True
    manager.set_gateway_message_injector(runner, runner._schedule_plugin_message_injection)


def _rows(authority, session_id):
    from hermes_state_runtime import list_session_admissions
    return list_session_admissions(authority.db, session_id=session_id, pending_only=False)


async def _gateway(monkeypatch, *, allowed="42"):
    """A single-profile gateway under the session authority; ``executed`` records each turn's text."""
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_results import execution_result
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setenv("DISCORD_ALLOWED_USERS", allowed)
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fixture-token", typing_indicator=False))

    async def send(chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="sent")
    monkeypatch.setattr(adapter, "send", send)
    adapter.gateway_runner = runner
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="owner")
    runner._wire_adapter_handlers(adapter)
    executed = []

    async def turn(event, source, quick_key, run_generation):
        executed.append(event.text)
        captured = execution_result.get()
        if captured is not None:
            captured["result"] = {"final_response": "reply", "messages": []}
        return "reply"
    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)
    return runner, adapter, authority, executed


def _origin(chat_id, user_id="42"):
    return {"platform": "discord", "chat_id": chat_id, "chat_type": "dm", "user_id": user_id}


@pytest.mark.asyncio
async def test_a_listed_chat_opens_its_session_and_a_human_reply_continues_it(monkeypatch):
    from gateway.platforms.event import MessageEvent
    from hermes_constants import get_hermes_home

    runner, adapter, authority, executed = await _gateway(monkeypatch)
    ctx, manager = _context(get_hermes_home(), ["discord:42"])
    _publish(runner, manager)
    source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")
    key = runner.session_store._generate_session_key(source)

    # Nobody has written in chat 42: the listed chat opens and the plugin turn runs as its first input.
    assert await asyncio.to_thread(ctx.inject_message, "kickoff", origin=_origin("42")) is True
    await async_wait_until(lambda: "kickoff" in executed, timeout=10, what="the opening plugin turn")
    owner = authority.logical_owner(runner.session_store.lookup_by_session_key(key).session_id)
    await async_wait_until(lambda: [row["outcome"] for row in _rows(authority, owner)] == ["completed"],
                           timeout=10, what="the opening turn to settle")
    (opened,) = _rows(authority, owner)
    assert opened["payload"]["native_text_v1"]["automation"]["opened"] is True

    # A human reply in the same chat keys into the session the plugin opened.
    await adapter.handle_message(MessageEvent(text="thanks", source=source, message_id="human-1"))
    await async_wait_until(lambda: "thanks" in executed, timeout=10, what="the human follow-up")
    assert authority.logical_owner(runner.session_store.lookup_by_session_key(key).session_id) == owner
    assert [row["payload"]["text"] for row in _rows(authority, owner)] == ["kickoff", "thanks"]


@pytest.mark.asyncio
async def test_an_unlisted_chat_or_an_unauthorized_user_still_opens_nothing(monkeypatch):
    from hermes_constants import get_hermes_home

    runner, adapter, _authority, executed = await _gateway(monkeypatch)
    ctx, manager = _context(get_hermes_home(), ["discord:42"])
    _publish(runner, manager)

    # Chat 43 is not listed: today's up-front refusal.
    assert await asyncio.to_thread(ctx.inject_message, "elsewhere", origin=_origin("43", user_id="43")) is False
    # Chat 42 is listed but user 99 fails the allowlist: dropped at dispatch, before any route is reserved.
    assert await asyncio.to_thread(ctx.inject_message, "stranger", origin=_origin("42", user_id="99")) is True
    await async_wait_until(lambda: not runner._background_tasks, timeout=10, what="the dispatch to finish")
    assert executed == []
    for chat_id, user_id in (("43", "43"), ("42", "99")):
        source = adapter.build_source(chat_id=chat_id, chat_type="dm", user_id=user_id)
        assert runner.session_store.lookup_by_session_key(runner.session_store._generate_session_key(source)) is None


@pytest.mark.asyncio
async def test_a_listed_chat_whose_connector_cannot_bind_is_refused_before_anything_is_reserved(monkeypatch):
    from hermes_constants import get_hermes_home

    runner, adapter, _authority, executed = await _gateway(monkeypatch)
    ctx, manager = _context(get_hermes_home(), ["discord:42"])
    _publish(runner, manager)
    runner._native_transport_homes.clear()  # the connector's home is gone: no binding to record

    assert await asyncio.to_thread(ctx.inject_message, "kickoff", origin=_origin("42")) is False
    assert await runner._dispatch_plugin_origin_injection(  # dispatch rechecks
        source=runner._plugin_origin_source(_origin("42"), plugin_home=get_hermes_home(), plugin_id=PLUGIN),
        content="kickoff", plugin_id=PLUGIN, plugin_home=get_hermes_home()) is False
    source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")
    assert runner.session_store.lookup_by_session_key(runner.session_store._generate_session_key(source)) is None
    assert executed == []


@pytest.mark.asyncio
async def test_a_queued_opening_turn_replays_only_against_the_same_live_connector(monkeypatch):
    """A restart recovers the queued opening turn like any native row: its binding is checked against
    the live connector again, and a changed credential leaves it queued instead of running it."""
    from gateway.session_authority import SessionAuthority, initialize_session_authority
    from hermes_constants import get_hermes_home

    runner, adapter, authority, executed = await _gateway(monkeypatch)
    ctx, manager = _context(get_hermes_home(), ["discord:42"])
    _publish(runner, manager)
    schedule = SessionAuthority._schedule
    monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: None)  # the owner dies before the drain
    assert await asyncio.to_thread(ctx.inject_message, "kickoff", origin=_origin("42")) is True
    source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")
    key = runner.session_store._generate_session_key(source)
    await async_wait_until(lambda: runner.session_store.lookup_by_session_key(key) is not None
                           and _rows(authority, runner.session_store.lookup_by_session_key(key).session_id),
                           timeout=10, what="the queued opening admission")
    sid = authority.logical_owner(runner.session_store.lookup_by_session_key(key).session_id)
    monkeypatch.setattr(SessionAuthority, "_schedule", schedule)

    restarted = await initialize_session_authority(runner, profile_id="default", instance_id="next-owner",
                                                   db=authority.db)
    adapter.config.token = "rotated-token"
    assert (await restarted.recover_native_sessions([(sid, source, adapter)]))[sid] != "ready"
    assert [row["status"] for row in _rows(restarted, sid)] == ["queued"] and executed == []

    adapter.config.token = "fixture-token"
    assert (await restarted.recover_native_sessions([(sid, source, adapter)]))[sid] == "ready"
    await async_wait_until(lambda: executed == ["kickoff"], timeout=10, what="the recovered opening turn")


@pytest.mark.parametrize(("entries", "origin", "listed"), [
    (["telegram:-100123"], ("telegram", "-100123", None, None), True),
    (["telegram:-100123"], ("telegram", "-100123", "17", None), True),      # a chat entry covers its threads
    (["telegram:-100123:17"], ("telegram", "-100123", "17", None), True),
    (["telegram:-100123:17"], ("telegram", "-100123", "18", None), False),  # a thread entry covers that one only
    (["telegram:-100123:17"], ("telegram", "-100123", None, None), False),
    (["Telegram:-100123"], ("telegram", "-100123", None, None), True),      # platform names ignore case
    (["discord:-100123"], ("telegram", "-100123", None, None), False),      # another platform's chat
    (["discord:900"], ("discord", "555", "555", "900"), True),              # a forum post under a listed forum
    (["discord:900:555"], ("discord", "555", "555", "900"), True),
    (["matrix:!room:example.org"], ("matrix", "!room:example.org", None, None), True),  # ids with ':'
    (["telegram", "origin", "nope:-100123", "telegram:-1", 7], ("telegram", "-100123", None, None), False),
    ("telegram:-100123", ("telegram", "-100123", None, None), False),       # a bare string is not a list
])
def test_open_chats_entries_name_a_chat_or_one_of_its_threads(entries, origin, listed):
    from gateway.config import Platform
    from gateway.session import SessionSource
    from hermes_cli.plugins_injection import open_chat_listed

    platform, chat_id, thread_id, parent_chat_id = origin
    source = SessionSource(platform=Platform(platform), chat_id=chat_id, chat_type="group", thread_id=thread_id,
                           parent_chat_id=parent_chat_id)
    assert open_chat_listed(entries, source) is listed


@pytest.mark.asyncio
async def test_multiplexed_profiles_open_only_their_own_listed_chats_on_their_own_bots(tmp_path, monkeypatch):
    """``alpha`` and ``beta`` each own a bot and list one chat. A -> B -> A: each plugin opens its chat in
    its own profile's store and FIFO, under its own bot's binding, and a human reply on that bot
    commits the same provenance into the same session. ``beta``'s list grants ``alpha`` nothing."""
    from unittest.mock import patch
    import gateway.run as gateway_run
    from gateway.config import GatewayConfig, Platform
    from gateway.platforms.event import MessageEvent, MessageType
    from gateway.run import GatewayRunner
    from gateway.session_authorities import SessionAuthorities
    from gateway.session_authority import initialize_session_authority
    from gateway.session_results import execution_result
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB
    from tests.gateway.test_plugin_origin_injection import _RecordingAdapter

    home = tmp_path / "hh"
    homes = {"default": home, "alpha": home / "profiles" / "alpha", "beta": home / "profiles" / "beta"}
    for profile_home in homes.values():
        profile_home.mkdir(parents=True, exist_ok=True)
        (profile_home / ".env").write_text("TELEGRAM_ALLOWED_USERS=u1\n", encoding="utf-8")
    contexts = {"alpha": _context(homes["alpha"], ["telegram:100"]), "beta": _context(homes["beta"], ["telegram:200"])}
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(gateway_run, "_hermes_home", home)
    runner = GatewayRunner(GatewayConfig(multiplex_profiles=True, sessions_dir=home / "sessions"))
    runner.config._runtime_profile_homes = list(homes.items())
    adapters = {name: _RecordingAdapter(name) for name in homes}
    runner.adapters = {Platform.TELEGRAM: adapters["default"]}
    runner._profile_adapters = {name: {Platform.TELEGRAM: adapters[name]} for name in ("alpha", "beta")}
    runner.session_authorities = SessionAuthorities(home, multiplexed=True)
    authorities = {}
    for name, profile_home in homes.items():
        authorities[name] = await initialize_session_authority(
            runner, profile_id=str(profile_home), instance_id="owner", db=SessionDB(profile_home / "state.db"),
            register=name == "default")
        runner.session_authorities.add(profile_home, authorities[name], name=name)
    for name, adapter in adapters.items():
        adapter.gateway_runner = runner
        if name != "default":
            adapter.set_owner_profile(name)
        adapter.set_message_handler(runner._make_profile_message_handler(name) if name != "default"
                                    else runner._make_default_profile_message_handler())
    executed = []

    async def turn(event, source, quick_key, run_generation):
        executed.append((event.text, get_hermes_home()))
        captured = execution_result.get()
        if captured is not None:
            captured["result"] = {"final_response": "reply", "messages": []}
        return "reply"
    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)

    def origin(chat_id, thread_id):
        return {"platform": "telegram", "chat_id": chat_id, "chat_type": "group", "thread_id": thread_id,
                "user_id": "u1"}

    async def opened(name, text, chat_id, thread_id):
        """Inject as *name*'s plugin, wait for the turn to settle, return the session's logical owner."""
        assert await asyncio.to_thread(contexts[name][0].inject_message, text, origin=origin(chat_id, thread_id))
        await async_wait_until(lambda: (text, homes[name]) in executed, timeout=10, what=f"{name}'s opening turn")
        async with gateway_run._async_profile_runtime_scope(homes[name]):
            entry = runner.session_store.lookup_by_session_key(f"agent:{name}:telegram:group:{chat_id}:{thread_id}")
        owner = authorities[name].logical_owner(entry.session_id)
        await async_wait_until(lambda: [row["outcome"] for row in _rows(authorities[name], owner)] == ["completed"],
                               timeout=10, what=f"{name}'s opening turn to settle")
        return owner

    with patch("hermes_cli.profiles.profiles_to_serve", return_value=list(homes.items())), \
            patch("hermes_cli.profiles.get_profile_dir", side_effect=lambda n: homes[n]), \
            patch("hermes_cli.profiles.profile_exists", side_effect=lambda n: n in homes):
        for _ctx, manager in contexts.values():
            _publish(runner, manager)
        try:
            alpha_owner = await opened("alpha", "alpha kickoff", "100", "7")
            beta_owner = await opened("beta", "beta kickoff", "200", "7")
            # Chat 200 on alpha's bot is alpha's chat, but only beta listed it.
            assert await asyncio.to_thread(contexts["alpha"][0].inject_message, "borrowed", origin=origin("200", "8")) is False
            await opened("alpha", "alpha again", "100", "8")
            assert [text for text, _home in executed] == ["alpha kickoff", "beta kickoff", "alpha again"]

            for name, owner, chat_id in (("alpha", alpha_owner, "100"), ("beta", beta_owner, "200")):
                reply = adapters[name].build_source(chat_id=chat_id, chat_type="group", thread_id="7", user_id="u1")
                async with gateway_run._async_profile_runtime_scope(homes[name]):
                    await adapters[name].handle_message(MessageEvent(
                        text=f"{name} thanks", message_type=MessageType.TEXT, source=reply, message_id=f"{name}-1"))
                await async_wait_until(lambda: len(_rows(authorities[name], owner)) == 2, timeout=10,
                                       what=f"the human reply on {name}'s bot")
                first, human = (row["payload"]["native_text_v1"] for row in _rows(authorities[name], owner))
                assert first["automation"]["opened"] is True and "automation" not in human
                assert first["provenance"] == human["provenance"]
                assert first["provenance"]["transport_profile"] == name
            await async_wait_until(lambda: len(executed) == 5, timeout=10, what="both human replies to run")
            assert {text: run_home for text, run_home in executed} == {
                "alpha kickoff": homes["alpha"], "beta kickoff": homes["beta"], "alpha again": homes["alpha"],
                "alpha thanks": homes["alpha"], "beta thanks": homes["beta"]}
            assert adapters["default"].sent == []
        finally:
            for authority in authorities.values():
                authority.db.close()
