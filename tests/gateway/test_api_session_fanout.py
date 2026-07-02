import pytest
from collections import OrderedDict
from typing import Any, cast

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource, SessionStore
from gateway.run import GatewayRunner


CANONICAL_KEY = "agent:main:shared:mobile-main"
DISCORD_KEY = "agent:main:discord:group:200000000000000001:900000000000000001"
MOBILE_SURFACE_KEYS = {
    "telegram-main": ("agent:main:telegram:dm:1000000001", Platform.TELEGRAM, "1000000001"),
    "discord-main": ("agent:main:discord:group:200000000000000002:900000000000000001", Platform.DISCORD, "200000000000000002"),
    "discord-life": ("agent:main:discord:group:200000000000000003:900000000000000001", Platform.DISCORD, "200000000000000003"),
    "discord-culture": ("agent:main:discord:group:200000000000000004:900000000000000001", Platform.DISCORD, "200000000000000004"),
}


class FakeAdapter:
    def __init__(self):
        self.sent = []
        self._pending_messages = {}
        self.started = []

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        return object()

    async def _send_with_retry(self, chat_id, content, reply_to=None, metadata=None):
        return await self.send(chat_id, content, reply_to=reply_to, metadata=metadata)

    def _start_session_processing(self, event, session_key):
        self.started.append((session_key, event.text))
        return True


def _runner(config):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = config
    runner.adapters = cast(Any, {})
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._draining = False
    runner._busy_input_mode = "interrupt"
    runner._busy_text_mode = "interrupt"
    runner._busy_ack_ts = {}
    runner._queued_events = {}
    runner._agent_cache = OrderedDict()
    runner._session_model_overrides = {}
    runner._session_reasoning_overrides = {}
    runner._pending_native_image_paths_by_session = {}
    runner._session_run_generation = {}
    runner._session_sources = OrderedDict()
    runner._session_sources_max = 512
    runner._is_user_authorized = lambda source: True
    return runner


class FakeAgent:
    def __init__(self):
        self.interrupts = []

    def interrupt(self, reason):
        self.interrupts.append(reason)


def _discord_event(text="new Discord instruction", chat_id="200000000000000001"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id=chat_id,
            chat_type="group",
            user_id="900000000000000001",
            user_name="User",
        ),
        message_id="m1",
    )


def _telegram_event(text="new Telegram instruction"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="1000000001",
            chat_type="dm",
            user_id="1000000001",
            user_name="User",
        ),
        message_id="m1",
    )


@pytest.mark.asyncio
async def test_api_originated_shared_session_fans_out_to_discord_alias():
    config = GatewayConfig(
        session_key_aliases={DISCORD_KEY: CANONICAL_KEY},
        session_id_overrides={CANONICAL_KEY: "mobile-main"},
    )
    runner = _runner(config)
    discord = FakeAdapter()
    runner.adapters[Platform.DISCORD] = cast(Any, discord)

    await runner._fanout_api_session_turn(
        origin_platform="api_server",
        session_key=CANONICAL_KEY,
        session_id="mobile-main",
        user_message="hello from the phone",
        assistant_response="hello back",
        run_id="run_test",
    )

    assert discord.sent == [
        {
            "chat_id": "200000000000000001",
            "content": "📱 Mobile: hello from the phone\n\nhello back",
            "reply_to": None,
            "metadata": None,
        }
    ]


@pytest.mark.asyncio
async def test_api_originated_daylight_user_only_fanout_is_dropped():
    config = GatewayConfig(
        session_key_aliases={DISCORD_KEY: CANONICAL_KEY},
        session_id_overrides={CANONICAL_KEY: "mobile-main"},
    )
    runner = _runner(config)
    discord = FakeAdapter()
    runner.adapters[Platform.DISCORD] = cast(Any, discord)

    await runner._fanout_api_session_turn(
        origin_platform="api_server",
        session_key=CANONICAL_KEY,
        session_id="mobile-main",
        user_message="hello from Mobile",
        assistant_response="",
        run_id="run_test",
        message_kind="user",
    )
    await runner._fanout_api_session_turn(
        origin_platform="api_server",
        session_key=CANONICAL_KEY,
        session_id="mobile-main",
        user_message="hello from Mobile",
        assistant_response="hello back",
        run_id="run_test",
        message_kind="turn",
    )

    assert discord.sent == [
        {
            "chat_id": "200000000000000001",
            "content": "📱 Mobile: hello from Mobile\n\nhello back",
            "reply_to": None,
            "metadata": None,
        },
    ]


@pytest.mark.asyncio
async def test_api_fanout_can_resolve_canonical_key_from_session_id_override():
    config = GatewayConfig(
        session_key_aliases={DISCORD_KEY: CANONICAL_KEY},
        session_id_overrides={CANONICAL_KEY: "mobile-main"},
    )
    runner = _runner(config)
    discord = FakeAdapter()
    runner.adapters[Platform.DISCORD] = cast(Any, discord)

    await runner._fanout_api_session_turn(
        origin_platform="api_server",
        session_key="",
        session_id="mobile-main",
        user_message="session id only",
        assistant_response="still mapped",
    )

    assert len(discord.sent) == 1
    assert discord.sent[0]["chat_id"] == "200000000000000001"
    assert discord.sent[0]["content"] == "📱 Mobile: session id only\n\nstill mapped"
    assert discord.sent[0]["metadata"] is None


@pytest.mark.asyncio
async def test_api_fanout_can_resolve_session_key_from_session_store_when_header_missing(tmp_path):
    config = GatewayConfig(session_key_aliases={DISCORD_KEY: DISCORD_KEY})
    runner = _runner(config)
    runner.session_store = SessionStore(tmp_path, config)
    runner.session_store._db = None
    with runner.session_store._lock:
        runner.session_store._ensure_loaded_locked()
        from datetime import datetime
        from gateway.session import SessionEntry

        now = datetime.now()
        runner.session_store._entries[DISCORD_KEY] = SessionEntry(
            session_key=DISCORD_KEY,
            session_id="mobile-session-id",
            created_at=now,
            updated_at=now,
        )
    discord = FakeAdapter()
    runner.adapters[Platform.DISCORD] = cast(Any, discord)

    await runner._fanout_api_session_turn(
        origin_platform="api_server",
        session_key="",
        session_id="mobile-session-id",
        user_message="missing header still resolves",
        assistant_response="resolved reply",
        message_kind="turn",
    )

    assert discord.sent == [
        {
            "chat_id": "200000000000000001",
            "content": "📱 Mobile: missing header still resolves\n\nresolved reply",
            "reply_to": None,
            "metadata": None,
        }
    ]


@pytest.mark.asyncio
async def test_non_api_origin_does_not_fan_out_back_to_discord():
    config = GatewayConfig(session_key_aliases={DISCORD_KEY: CANONICAL_KEY})
    runner = _runner(config)
    discord = FakeAdapter()
    runner.adapters[Platform.DISCORD] = cast(Any, discord)

    await runner._fanout_api_session_turn(
        origin_platform="discord",
        session_key=CANONICAL_KEY,
        session_id="mobile-main",
        user_message="from discord",
        assistant_response="do not loop",
    )

    assert discord.sent == []


@pytest.mark.asyncio
async def test_api_fanout_is_gated_by_alias_mapping_and_connected_adapter():
    runner = _runner(GatewayConfig())
    discord = FakeAdapter()
    runner.adapters[Platform.DISCORD] = cast(Any, discord)

    await runner._fanout_api_session_turn(
        origin_platform="api_server",
        session_key=CANONICAL_KEY,
        session_id="mobile-main",
        user_message="unmapped",
        assistant_response="not visible",
    )
    assert discord.sent == []

    mapped_without_adapter = _runner(GatewayConfig(session_key_aliases={DISCORD_KEY: CANONICAL_KEY}))
    await mapped_without_adapter._fanout_api_session_turn(
        origin_platform="api_server",
        session_key=CANONICAL_KEY,
        session_id="mobile-main",
        user_message="mapped",
        assistant_response="but no adapter",
    )
    # No exception and no implicit adapter creation.
    assert mapped_without_adapter.adapters == {}


def test_api_fanout_target_resolution_dedupes_config_aliases():
    duplicate_key = "agent:main:discord:group:200000000000000001:900000000000000001"
    runner = _runner(
        GatewayConfig(
            session_key_aliases={
                DISCORD_KEY: CANONICAL_KEY,
                duplicate_key: CANONICAL_KEY,
            }
        )
    )

    assert runner._fanout_targets_for_session_key(CANONICAL_KEY) == [
        (Platform.DISCORD, "200000000000000001", "")
    ]


def test_api_fanout_target_resolution_routes_direct_concrete_session_key_without_alias():
    runner = _runner(GatewayConfig())

    assert runner._fanout_targets_for_session_key(DISCORD_KEY) == [
        (Platform.DISCORD, "200000000000000001", "")
    ]



def test_api_fanout_target_resolution_preserves_real_discord_thread_keys():
    thread_key = "agent:main:discord:thread:200000000000000005:200000000000000005"
    runner = _runner(GatewayConfig(session_key_aliases={thread_key: thread_key}))

    assert runner._fanout_targets_for_session_key(thread_key) == [
        (Platform.DISCORD, "200000000000000005", "200000000000000005")
    ]

def test_pinned_mobile_surface_targets_resolve_from_session_id_overrides():
    """mobile surface switcher targets must survive Hermes updates/refactors."""
    aliases = {key: key for key, _, _ in MOBILE_SURFACE_KEYS.values()}
    overrides = {key: session_id for session_id, (key, _, _) in MOBILE_SURFACE_KEYS.items()}
    runner = _runner(
        GatewayConfig(
            session_key_aliases=aliases,
            session_id_overrides=overrides,
        )
    )

    for session_id, (key, platform, chat_id) in MOBILE_SURFACE_KEYS.items():
        expected_thread_id = ""
        assert runner._fanout_targets_for_session_key("", session_id) == [
            (platform, chat_id, expected_thread_id)
        ]
        assert runner._fanout_targets_for_session_key(key, session_id) == [
            (platform, chat_id, expected_thread_id)
        ]


def test_pinned_surface_platform_turns_use_same_session_ids_as_mobile_fetches(tmp_path):
    """Platform-origin turns and mobile app history fetch must address one SessionDB lane."""
    aliases = {key: key for key, _, _ in MOBILE_SURFACE_KEYS.values()}
    overrides = {key: session_id for session_id, (key, _, _) in MOBILE_SURFACE_KEYS.items()}
    config = GatewayConfig(
        session_key_aliases=aliases,
        session_id_overrides=overrides,
    )
    store = SessionStore(tmp_path, config)

    telegram_entry = store.get_or_create_session(_telegram_event().source)
    assert telegram_entry.session_key == MOBILE_SURFACE_KEYS["telegram-main"][0]
    assert telegram_entry.session_id == "telegram-main"

    for session_id, (key, platform, chat_id) in MOBILE_SURFACE_KEYS.items():
        if platform != Platform.DISCORD:
            continue
        entry = store.get_or_create_session(_discord_event(chat_id=chat_id).source)
        assert entry.session_key == key
        assert entry.session_id == session_id


def test_api_session_key_control_rotates_session_id_not_channel_key(tmp_path):
    key, _, _ = MOBILE_SURFACE_KEYS["discord-life"]
    config = GatewayConfig(
        session_key_aliases={key: key},
        session_id_overrides={key: "discord-life"},
    )
    runner = _runner(config)
    runner.session_store = SessionStore(tmp_path, config)

    current = runner._api_session_key_control(action="current", session_key=key)
    assert current is not None
    assert current["session_key"] == key
    assert current["session_id"] == "discord-life"

    reset = runner._api_session_key_control(action="reset", session_key=key)
    assert reset is not None
    assert reset["session_key"] == key
    assert reset["session_id"] != "discord-life"

    current_again = runner._api_session_key_control(action="current", session_key=key)
    assert current_again is not None
    assert current_again["session_key"] == key
    assert current_again["session_id"] == reset["session_id"]


@pytest.mark.asyncio
async def test_pinned_surface_platform_followups_interrupt_api_owned_runs():
    aliases = {key: key for key, _, _ in MOBILE_SURFACE_KEYS.values()}
    overrides = {key: session_id for session_id, (key, _, _) in MOBILE_SURFACE_KEYS.items()}

    for session_id, (key, platform, chat_id) in MOBILE_SURFACE_KEYS.items():
        runner = _runner(
            GatewayConfig(
                session_key_aliases=aliases,
                session_id_overrides=overrides,
            )
        )
        adapter = FakeAdapter()
        runner.adapters[platform] = cast(Any, adapter)
        agent = FakeAgent()
        event = (
            _telegram_event(f"interrupt {session_id}")
            if platform == Platform.TELEGRAM
            else _discord_event(f"interrupt {session_id}", chat_id=chat_id)
        )

        runner._handle_api_session_run_lifecycle(
            event="started",
            origin_platform="api_server",
            session_key=key,
            session_id=session_id,
            run_id=f"run_{session_id}",
            agent=agent,
        )

        handled = await runner._handle_active_session_busy_message(event, key)

        assert handled is True
        assert agent.interrupts == [f"interrupt {session_id}"]
        assert adapter._pending_messages[key].text == f"interrupt {session_id}"

        runner._handle_api_session_run_lifecycle(
            event="completed",
            origin_platform="api_server",
            session_key=key,
            session_id=session_id,
            run_id=f"run_{session_id}",
            agent=agent,
        )

        assert key not in runner._running_agents
        assert adapter._pending_messages == {}
        assert adapter.started == [(key, f"interrupt {session_id}")]


def test_api_session_run_lifecycle_registers_and_clears_canonical_agent():
    runner = _runner(
        GatewayConfig(
            session_key_aliases={DISCORD_KEY: CANONICAL_KEY},
            session_id_overrides={CANONICAL_KEY: "mobile-main"},
        )
    )
    agent = FakeAgent()

    runner._handle_api_session_run_lifecycle(
        event="started",
        origin_platform="api_server",
        session_key="",
        session_id="mobile-main",
        run_id="run_api",
        agent=agent,
    )

    assert runner._running_agents[CANONICAL_KEY] is agent

    runner._handle_api_session_run_lifecycle(
        event="completed",
        origin_platform="api_server",
        session_key="",
        session_id="mobile-main",
        run_id="run_api",
        agent=agent,
    )

    assert CANONICAL_KEY not in runner._running_agents


@pytest.mark.asyncio
async def test_discord_followup_interrupts_api_owned_shared_session_and_drains_after_completion():
    runner = _runner(
        GatewayConfig(
            session_key_aliases={DISCORD_KEY: CANONICAL_KEY},
            session_id_overrides={CANONICAL_KEY: "mobile-main"},
        )
    )
    discord = FakeAdapter()
    runner.adapters[Platform.DISCORD] = cast(Any, discord)
    agent = FakeAgent()
    event = _discord_event("interrupt from discord")

    runner._handle_api_session_run_lifecycle(
        event="started",
        origin_platform="api_server",
        session_key=CANONICAL_KEY,
        session_id="mobile-main",
        run_id="run_api",
        agent=agent,
    )

    handled = await runner._handle_active_session_busy_message(event, CANONICAL_KEY)

    assert handled is True
    assert agent.interrupts == ["interrupt from discord"]
    assert discord._pending_messages[CANONICAL_KEY].text == "interrupt from discord"

    runner._handle_api_session_run_lifecycle(
        event="completed",
        origin_platform="api_server",
        session_key=CANONICAL_KEY,
        session_id="mobile-main",
        run_id="run_api",
        agent=agent,
    )

    assert CANONICAL_KEY not in runner._running_agents
    assert discord._pending_messages == {}
    assert discord.started == [(CANONICAL_KEY, "interrupt from discord")]


@pytest.mark.asyncio
async def test_api_server_adapter_invokes_session_fanout_hook():
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    calls = []

    async def handler(**kwargs):
        calls.append(kwargs)

    adapter.set_session_fanout_handler(handler)
    await adapter._notify_session_fanout(
        gateway_session_key=CANONICAL_KEY,
        session_id="mobile-main",
        user_message=[{"type": "text", "text": "phone text"}],
        assistant_response="assistant text",
        run_id="run_1",
    )

    assert calls == [
        {
            "origin_platform": "api_server",
            "session_key": CANONICAL_KEY,
            "session_id": "mobile-main",
            "user_message": "phone text",
            "assistant_response": "assistant text",
            "run_id": "run_1",
            "message_kind": "turn",
        }
    ]


def test_api_server_adapter_invokes_session_run_lifecycle_hook():
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    calls = []
    agent = object()

    def handler(**kwargs):
        calls.append(kwargs)

    adapter.set_session_run_handler(handler)
    adapter._notify_session_run_lifecycle(
        "started",
        gateway_session_key=CANONICAL_KEY,
        session_id="mobile-main",
        run_id="run_1",
        agent=agent,
    )

    assert calls == [
        {
            "event": "started",
            "origin_platform": "api_server",
            "session_key": CANONICAL_KEY,
            "session_id": "mobile-main",
            "run_id": "run_1",
            "agent": agent,
        }
    ]


def test_api_server_session_key_resolves_platform_context_for_mobile_surfaces():
    user_config = {
        "discord": {
            "channel_prompts": {
                "200000000000000003": "Life workspace prompt",
            }
        }
    }
    ctx = APIServerAdapter._surface_context_from_session_key(
        "agent:main:discord:group:200000000000000003:900000000000000001",
        user_config,
    )
    assert ctx["platform_key"] == "discord"
    assert ctx["platform"] == "discord"
    assert ctx["chat_id"] == "200000000000000003"
    assert ctx["channel_prompt"] == "Life workspace prompt"

    tg = APIServerAdapter._surface_context_from_session_key(
        "agent:main:telegram:dm:1000000001",
        {"telegram": {}},
    )
    assert tg["platform_key"] == "telegram"
    assert tg["platform"] == "telegram"
    assert tg["chat_id"] == "1000000001"

    plain = APIServerAdapter._surface_context_from_session_key("mobile-main", user_config)
    assert plain["platform_key"] == "api_server"
