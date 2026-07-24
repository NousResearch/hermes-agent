import asyncio
import pytest
import sys
import types


def test_scrub_voice_turn_id_preserves_other_completed_segments():
    """Failed-stream cleanup must only drop the failed segment's turn id.

    Multi-segment streamed turns (e.g. preamble that completed +
    final that failed) used to lose every voice_turn_id because the
    failed-stream cleanup path blanket-popped voice_turn_id /
    voice_turn_ids after delegating to scrub. The fix in gateway/run.py
    is to trust scrub's per-id semantics; this regression test pins
    that semantics so a future revert is caught.
    """
    from gateway.voice_stream import scrub_voice_turn_id_from_result

    result = {
        "voice_turn_id": "final-stream-turn",
        "voice_turn_ids": ["preamble-turn", "final-stream-turn"],
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "preamble", "voice_turn_id": "preamble-turn"},
            {"role": "assistant", "content": "final", "voice_turn_id": "final-stream-turn"},
        ],
    }

    changed = scrub_voice_turn_id_from_result(result, "final-stream-turn")

    assert changed is True
    assert result["voice_turn_ids"] == ["preamble-turn"], \
        "completed preamble turn id must be preserved"
    assert "voice_turn_id" not in result, \
        "result-level final turn id must be cleared after failure"
    assert result["messages"][1]["voice_turn_id"] == "preamble-turn", \
        "preamble row must keep its voice_turn_id"
    assert "voice_turn_id" not in result["messages"][2], \
        "failed final row's voice_turn_id must be stripped"


def test_gateway_config_accepts_voice_server_platform():
    from gateway.config import GatewayConfig, Platform

    cfg = GatewayConfig.from_dict(
        {
            "platforms": {
                "voice_server": {
                    "enabled": True,
                    "extra": {
                        "url": "ws://127.0.0.1:7860/events",
                        "room_id": "personal-room",
                    },
                }
            }
        }
    )

    assert Platform.VOICE_SERVER in cfg.platforms
    assert cfg.platforms[Platform.VOICE_SERVER].extra["room_id"] == "personal-room"
    assert cfg.get_connected_platforms() == [Platform.VOICE_SERVER]


def test_env_overrides_enable_voice_server_platform(monkeypatch):
    from gateway.config import GatewayConfig, Platform, _apply_env_overrides

    monkeypatch.setenv("VOICE_SERVER_ENABLED", "true")
    monkeypatch.setenv("VOICE_SERVER_ROOM_URL", "ws://127.0.0.1:7860/events")
    monkeypatch.setenv("VOICE_SERVER_ROOM_ID", "personal-room")

    cfg = GatewayConfig()
    _apply_env_overrides(cfg)

    assert cfg.platforms[Platform.VOICE_SERVER].enabled is True
    assert cfg.platforms[Platform.VOICE_SERVER].extra == {
        "url": "ws://127.0.0.1:7860/events",
        "room_id": "personal-room",
    }


def test_gateway_runner_creates_voice_server_adapter():
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.run import GatewayRunner
    from gateway.platforms.voice_server import VoiceServerAdapter

    cfg = GatewayConfig(
        platforms={
            Platform.VOICE_SERVER: PlatformConfig(
                enabled=True,
                extra={"url": "ws://127.0.0.1:7860/events"},
            )
        }
    )
    runner = GatewayRunner(config=cfg)

    adapter = runner._create_adapter(Platform.VOICE_SERVER, cfg.platforms[Platform.VOICE_SERVER])

    assert isinstance(adapter, VoiceServerAdapter)
    assert cfg.platforms[Platform.VOICE_SERVER].extra["group_sessions_per_user"] is True
    assert cfg.platforms[Platform.VOICE_SERVER].extra["thread_sessions_per_user"] is True
    assert adapter.SUPPORTS_MESSAGE_EDITING is False


def test_gateway_runner_voice_server_grouping_default_overrides_global_false():
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.run import GatewayRunner

    cfg = GatewayConfig(
        group_sessions_per_user=False,
        platforms={
            Platform.VOICE_SERVER: PlatformConfig(
                enabled=True,
                extra={"url": "ws://127.0.0.1:7860/events"},
            )
        },
    )
    runner = GatewayRunner(config=cfg)

    runner._create_adapter(Platform.VOICE_SERVER, cfg.platforms[Platform.VOICE_SERVER])

    assert cfg.platforms[Platform.VOICE_SERVER].extra["group_sessions_per_user"] is True


def test_gateway_runner_skips_home_channel_notice_for_voice_server():
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    cfg = GatewayConfig(
        platforms={
            Platform.VOICE_SERVER: PlatformConfig(
                enabled=True,
                extra={"url": "ws://127.0.0.1:7860/events"},
            )
        }
    )
    runner = GatewayRunner(config=cfg)
    source = SessionSource(
        platform=Platform.VOICE_SERVER,
        chat_id="default",
        chat_type="channel",
        user_id="caller",
    )

    assert runner._should_send_home_channel_notice(source, history=[]) is False


def test_voice_server_adapter_normalizes_auto_client_url_to_events_websocket():
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "http://127.0.0.1:7860/auto-client/"},
        )
    )

    assert adapter._events_url() == "ws://127.0.0.1:7860/events"


def test_voice_server_adapter_maps_transcript_event_to_voice_message():
    from gateway.config import PlatformConfig, Platform
    from gateway.platforms.base import MessageType
    from gateway.platforms.voice_server import VoiceServerAdapter

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )

    event = adapter.event_to_message_event(
        {
            "type": "transcript",
            "room_id": "personal-room",
            "room_name": "Personal Room",
            "participant_id": "whatsapp:+49123",
            "participant_name": "Lev",
            "text": "Hallo Hermes",
            "turn_id": "turn-1",
        }
    )

    assert event is not None
    assert event.message_type == MessageType.VOICE
    assert event.text == "Hallo Hermes"
    assert event.message_id == "turn-1"
    assert event.source.platform == Platform.VOICE_SERVER
    assert event.source.chat_id == "personal-room"
    assert event.source.chat_name == "Personal Room"
    assert event.source.chat_type == "channel"
    assert event.source.user_id == "whatsapp:+49123"
    assert event.source.user_name == "Lev"
    assert event.auto_skill == "talk"


@pytest.mark.asyncio
async def test_voice_server_adapter_pre_authorizes_outbound_call_started():
    """Outbound ``call_started`` confirmations for Hermes-initiated calls
    must bypass the inbound allowlist.

    ``start_outbound_call()`` is invoked by an already-authorized agent
    turn, so the voice runtime's confirmation may legitimately arrive
    without caller identity. The adapter tracks pending outbound
    ``call_id``s and lets matching ``call_started`` events through even
    when the authorizer rejects the synthetic ``"caller"`` user_id that
    ``event_to_session_source()`` falls back to.
    """
    from gateway.config import PlatformConfig
    from gateway.platforms.base import build_session_key
    from gateway.platforms.voice_server import VoiceServerAdapter

    sessions: list = []

    class CapturingSessionStore:
        def get_or_create_session(self, source, force_new=False):
            sessions.append((source, force_new))
            return types.SimpleNamespace(
                session_id=f"session-{len(sessions)}",
                session_key=build_session_key(source),
            )

    sent: list = []

    class FakeWebSocket:
        closed = False

        async def send_json(self, payload):
            sent.append(payload)

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter.set_session_store(CapturingSessionStore())
    # Authorizer rejects the synthetic "caller" identity that outbound
    # call_started events fall back to. Pre-authorization must still allow
    # the Hermes-initiated call through.
    adapter.set_authorizer(lambda source: False)
    adapter._ws = FakeWebSocket()

    result = await adapter.start_outbound_call(target="+15551234567", room_id="personal-room")
    assert result.success, "outbound call request must send successfully"
    outbound_payload = next(p for p in sent if p["type"] == "start_outbound_call")
    expected_call_id = outbound_payload["call_id"]
    sent.clear()

    await adapter.handle_room_event(
        {
            "type": "call_started",
            "room_id": "personal-room",
            "call_id": expected_call_id,
        }
    )

    assert len(sessions) == 1, "pre-authorized outbound call_started must create a session"
    assert any(p["type"] == "session_bound" for p in sent), \
        "session_bound must be emitted for pre-authorized outbound call"

    # An unrelated outbound call_id (not initiated by Hermes) must still
    # be rejected by the authorizer.
    sessions.clear()
    sent.clear()
    await adapter.handle_room_event(
        {
            "type": "call_started",
            "room_id": "personal-room",
            "call_id": "unknown-outbound-id",
        }
    )
    assert sessions == [], "outbound call_started with unknown call_id must be authorized"
    assert sent == [], "no session_bound for unknown-outbound call_started"

    # Pending entries are consumed on bind: a second call_started reusing
    # the same id must not bypass auth a second time.
    sessions.clear()
    sent.clear()
    await adapter.handle_room_event(
        {
            "type": "call_started",
            "room_id": "personal-room",
            "call_id": expected_call_id,
        }
    )
    assert sessions == [], "reused outbound call_id must not bypass auth twice"
    assert sent == [], "no session_bound on second use of consumed outbound id"


@pytest.mark.asyncio
async def test_voice_server_adapter_outbound_pending_is_scoped_to_room():
    """An outbound call_id registered for room A must not bypass auth in room B."""
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    sent: list = []

    class FakeWebSocket:
        closed = False

        async def send_json(self, payload):
            sent.append(payload)

    class FailIfTouchedSessionStore:
        def get_or_create_session(self, source, force_new=False):
            raise AssertionError("session store must not be touched in this test")

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "url": "ws://127.0.0.1:7860/events",
                "room_id": "room-a",
                "multi_room": True,
            },
        )
    )
    adapter.set_session_store(FailIfTouchedSessionStore())
    adapter.set_authorizer(lambda source: False)
    adapter._ws = FakeWebSocket()

    result = await adapter.start_outbound_call(target="+15551234567", room_id="room-a")
    assert result.success
    outbound_payload = next(p for p in sent if p["type"] == "start_outbound_call")
    foreign_call_id = outbound_payload["call_id"]
    sent.clear()

    # Different room, same call_id: must NOT pre-authorize.
    await adapter.handle_room_event(
        {
            "type": "call_started",
            "room_id": "room-b",
            "call_id": foreign_call_id,
        }
    )
    assert sent == [], "cross-room outbound id reuse must not bypass auth"


@pytest.mark.asyncio
async def test_voice_server_adapter_outbound_pending_expires_after_ttl(monkeypatch):
    """A stale unconfirmed outbound id must not pre-authorize indefinitely.

    Without a TTL, a `start_outbound_call` whose confirmation never
    arrives leaves its (room_id, call_id) entry in the ledger until
    1024 newer entries evict it. That lets a much-later unrelated
    `call_started` reusing the same id bypass the authorizer.
    """
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    sent: list = []

    class FakeWebSocket:
        closed = False

        async def send_json(self, payload):
            sent.append(payload)

    class FailIfTouchedSessionStore:
        def get_or_create_session(self, source, force_new=False):
            raise AssertionError(
                "session store must not be touched for expired pending outbound"
            )

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter.set_session_store(FailIfTouchedSessionStore())
    adapter.set_authorizer(lambda source: False)
    adapter._ws = FakeWebSocket()
    # Shorten the TTL for the test so we don't need to actually wait.
    adapter._pending_outbound_ttl_seconds = 0.5

    # Drive a monotonic clock we can fast-forward.
    fake_now = {"t": 1_000_000.0}
    monkeypatch.setattr(
        "gateway.platforms.voice_server.time.monotonic", lambda: fake_now["t"]
    )

    result = await adapter.start_outbound_call(target="+15551234567", room_id="personal-room")
    assert result.success
    outbound_payload = next(p for p in sent if p["type"] == "start_outbound_call")
    expected_call_id = outbound_payload["call_id"]
    sent.clear()

    # Fast-forward past the TTL window with no confirmation arriving.
    fake_now["t"] += 10.0

    await adapter.handle_room_event(
        {
            "type": "call_started",
            "room_id": "personal-room",
            "call_id": expected_call_id,
        }
    )

    assert sent == [], "expired pending outbound id must not bypass authorizer"


@pytest.mark.asyncio
async def test_voice_server_adapter_outbound_send_failure_clears_pending():
    """If start_outbound_call's send fails, the pre-registered id is discarded.

    Without cleanup, a spurious ``call_started`` for that id could bypass
    auth even though Hermes never actually placed the call.
    """
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    class ExplodingWebSocket:
        closed = False

        async def send_json(self, payload):
            raise RuntimeError("transport down")

    class FailIfTouchedSessionStore:
        def get_or_create_session(self, source, force_new=False):
            raise AssertionError("session store must not be touched")

    sent: list = []

    class CapturingWebSocket:
        closed = False

        async def send_json(self, payload):
            sent.append(payload)

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter.set_session_store(FailIfTouchedSessionStore())
    adapter.set_authorizer(lambda source: False)
    adapter._ws = ExplodingWebSocket()

    result = await adapter.start_outbound_call(target="+15551234567", room_id="personal-room")
    assert not result.success, "send failure must propagate"

    # Re-attach a working ws and try to use the (now discarded) id. Auth must reject.
    adapter._ws = CapturingWebSocket()
    # Reconstruct the call_id from the exploding attempt is not possible, but
    # we can prove the ledger is empty by inspecting it directly.
    assert len(adapter._pending_outbound_calls) == 0, \
        "send failure must clear the pre-registered outbound entry"


@pytest.mark.asyncio
async def test_voice_server_adapter_rejects_unauthorized_call_lifecycle_events():
    """Unauthorized callers must not provision sessions or receive session_bound.

    Transcript events go through ``handle_message`` which the runner
    authorizes; ``inbound_call`` / ``call_started`` are adapter-side and so
    must call the authorizer set by the runner explicitly. Without the gate,
    any room participant could mint ``session_id`` / ``session_key`` by
    pressing the New Call button.
    """
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    sessions: list = []

    class FailIfTouchedSessionStore:
        def get_or_create_session(self, source, force_new=False):
            sessions.append(source)
            raise AssertionError(
                "session store must not be touched for unauthorized callers"
            )

    sent: list = []

    class FakeWebSocket:
        closed = False

        async def send_json(self, payload):
            sent.append(payload)

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter.set_session_store(FailIfTouchedSessionStore())
    adapter.set_authorizer(lambda source: source.user_id == "allowed-caller")
    adapter._ws = FakeWebSocket()

    await adapter.handle_room_event(
        {
            "type": "inbound_call",
            "room_id": "personal-room",
            "session_call_id": "call-evil-1",
            "caller": {"id": "stranger", "name": "Stranger"},
        }
    )

    await adapter.handle_room_event(
        {
            "type": "call_started",
            "room_id": "personal-room",
            "call_id": "call-evil-2",
            "caller": {"id": "stranger", "name": "Stranger"},
        }
    )

    assert sessions == [], "session store was touched for unauthorized caller"
    assert sent == [], "session_bound must not be sent to unauthorized caller"


@pytest.mark.asyncio
async def test_voice_server_call_lifecycle_uses_gateway_allowlist(monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter
    from gateway.run import GatewayRunner

    monkeypatch.setenv("VOICE_SERVER_ALLOWED_USERS", "allowed-caller")
    monkeypatch.delenv("VOICE_SERVER_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)

    class PairingStore:
        def is_approved(self, platform_name, user_id):
            return False

    runner = object.__new__(GatewayRunner)
    runner.pairing_store = PairingStore()

    sessions: list = []

    class FailIfTouchedSessionStore:
        def get_or_create_session(self, source, force_new=False):
            sessions.append(source)
            raise AssertionError(
                "session store must not be touched for callers outside VOICE_SERVER_ALLOWED_USERS"
            )

    sent: list = []

    class FakeWebSocket:
        closed = False

        async def send_json(self, payload):
            sent.append(payload)

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter.set_session_store(FailIfTouchedSessionStore())
    adapter.set_authorizer(runner._is_user_authorized)
    adapter._ws = FakeWebSocket()

    await adapter.handle_room_event(
        {
            "type": "inbound_call",
            "room_id": "personal-room",
            "session_call_id": "call-evil-1",
            "caller": {"id": "stranger", "name": "Stranger"},
        }
    )

    assert sessions == []
    assert sent == []


@pytest.mark.asyncio
async def test_voice_server_adapter_starts_inbound_call_without_message_dispatch():
    from gateway.config import PlatformConfig
    from gateway.platforms.base import build_session_key
    from gateway.platforms.voice_server import VoiceServerAdapter

    sessions = []
    handled = []

    class FakeSessionStore:
        def get_or_create_session(self, source, force_new=False):
            sessions.append((source, force_new))
            return types.SimpleNamespace(
                session_id=f"session-{len(sessions)}",
                session_key=build_session_key(source, thread_sessions_per_user=True),
            )

    sent = []

    class FakeWebSocket:
        closed = False

        async def send_json(self, payload):
            sent.append(payload)

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter.set_session_store(FakeSessionStore())
    adapter.set_message_handler(lambda event: handled.append(event))
    adapter._ws = FakeWebSocket()

    await adapter.handle_room_event(
        {
            "type": "inbound_call",
            "room_id": "personal-room",
            "session_call_id": "call-1",
            "caller": {"id": "browser-caller", "name": "Browser caller"},
        }
    )
    await adapter.handle_room_event(
        {
            "type": "inbound_call",
            "room_id": "personal-room",
            "session_call_id": "call-1",
            "caller": {"id": "browser-caller", "name": "Browser caller"},
        }
    )

    assert handled == []
    assert len(sessions) == 2
    first_source, first_force_new = sessions[0]
    duplicate_source, duplicate_force_new = sessions[1]
    assert first_force_new is False
    assert duplicate_force_new is False
    assert first_source.user_id == "browser-caller"
    assert first_source.user_name == "Browser caller"
    assert first_source.thread_id == "call-1"
    assert duplicate_source.thread_id == "call-1"
    assert sent == [
        {
            "type": "session_bound",
            "room_id": "personal-room",
            "session_call_id": "call-1",
            "call_id": "call-1",
            "direction": "inbound",
            "session_id": "session-1",
            "session_key": build_session_key(first_source, thread_sessions_per_user=True),
            "caller": {"id": "browser-caller", "name": "Browser caller"},
        },
        {
            "type": "session_bound",
            "room_id": "personal-room",
            "session_call_id": "call-1",
            "call_id": "call-1",
            "direction": "inbound",
            "session_id": "session-2",
            "session_key": build_session_key(duplicate_source, thread_sessions_per_user=True),
            "caller": {"id": "browser-caller", "name": "Browser caller"},
        },
    ]

    transcript = adapter.event_to_message_event(
        {
            "type": "transcript",
            "room_id": "personal-room",
            "session_call_id": "call-1",
            "caller": {"id": "browser-caller", "name": "Browser caller"},
            "text": "hello",
            "turn_id": "turn-1",
        }
    )

    assert transcript is not None
    assert transcript.text == "hello"
    assert transcript.source.thread_id == "call-1"
    assert build_session_key(transcript.source) == build_session_key(first_source)

    second_source = adapter.event_to_session_source(
        {
            "type": "inbound_call",
            "room_id": "personal-room",
            "call_id": "call-2",
            "caller": {"id": "browser-caller", "name": "Browser caller"},
        }
    )
    assert build_session_key(first_source, thread_sessions_per_user=True) != build_session_key(
        second_source,
        thread_sessions_per_user=True,
    )
    other_caller_same_call = adapter.event_to_session_source(
        {
            "type": "call_started",
            "room_id": "personal-room",
            "call_id": "call-2",
            "caller": {"id": "other-caller"},
        }
    )
    assert build_session_key(second_source, thread_sessions_per_user=True) != build_session_key(
        other_caller_same_call,
        thread_sessions_per_user=True,
    )


@pytest.mark.asyncio
async def test_voice_server_adapter_ignores_room_status_without_creating_session():
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    handled = []
    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter.set_message_handler(lambda event: handled.append(event))

    await adapter.handle_room_event({"type": "room_status", "room_id": "personal-room"})

    assert handled == []


def test_voice_server_adapter_session_grouping_matches_discord_voice_mode():
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.base import SessionSource, build_session_key
    from gateway.platforms.voice_server import VoiceServerAdapter

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    source = SessionSource(
        platform=Platform.VOICE_SERVER,
        chat_id="personal-room",
        chat_type="channel",
        user_id="caller",
    )

    assert adapter._session_key_for_source(source) == build_session_key(source)

    adapter.config.extra["group_sessions_per_user"] = "false"

    assert adapter._session_key_for_source(source) == build_session_key(
        source,
        group_sessions_per_user=False,
    )


@pytest.mark.asyncio
async def test_voice_server_adapter_normal_response_includes_participant_id():
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    sent = []

    class FakeWebSocket:
        closed = False

        async def send_json(self, payload):
            sent.append(payload)

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter._ws = FakeWebSocket()

    async def handler(_event):
        return "Hallo"

    adapter.set_message_handler(handler)

    await adapter.handle_room_event(
        {
            "type": "transcript",
            "room_id": "personal-room",
            "participant_id": "caller",
            "text": "hi",
            "turn_id": "voice-turn-1",
        }
    )

    for _ in range(50):
        if sent:
            break
        await asyncio.sleep(0.01)
    await adapter.cancel_background_tasks()

    assert sent
    assert sent[-1]["type"] == "assistant_reply"
    assert sent[-1]["participant_id"] == "caller"


@pytest.mark.asyncio
async def test_voice_server_adapter_connect_starts_room_bot(monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    sent = []

    class FakeWebSocket:
        closed = False

        async def send_json(self, payload):
            sent.append(payload)

    class FakeSession:
        async def ws_connect(self, url):
            assert url == "ws://127.0.0.1:7860/events"
            return FakeWebSocket()

        async def close(self):
            pass

    class DummyTask:
        def cancel(self):
            pass

    def create_task(coro):
        coro.close()
        return DummyTask()

    fake_aiohttp = types.SimpleNamespace(ClientSession=lambda: FakeSession())
    monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)
    monkeypatch.setattr("asyncio.create_task", create_task)

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )

    assert await adapter.connect() is True
    assert sent == [{"type": "start_bot", "room_id": "personal-room"}]


@pytest.mark.asyncio
async def test_voice_server_adapter_send_writes_assistant_reply_to_websocket():
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    sent = []

    class FakeWebSocket:
        closed = False

        async def send_json(self, payload):
            sent.append(payload)

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter._ws = FakeWebSocket()

    result = await adapter.send(
        "personal-room",
        "Hallo, ich bin da.",
        metadata={
            "turn_id": "assistant-turn-1",
            "participant_id": "whatsapp:+49123",
            "thread_id": "call-1",
        },
    )

    assert result.success is True
    assert sent == [
        {
            "type": "assistant_reply",
            "room_id": "personal-room",
            "text": "Hallo, ich bin da.",
            "turn_id": "assistant-turn-1",
            "participant_id": "whatsapp:+49123",
            "call_id": "call-1",
        }
    ]


@pytest.mark.asyncio
async def test_voice_server_adapter_sends_append_only_llm_stream_events():
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    sent = []

    class FakeWebSocket:
        closed = False

        async def send_json(self, payload):
            sent.append(payload)

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter._ws = FakeWebSocket()
    metadata = {
        "turn_id": "assistant-turn-1",
        "participant_id": "whatsapp:+49123",
        "thread_id": "call-1",
        "_hermes_session_key": "agent:main:voice_server:channel:personal-room",
        "_hermes_session_id": "session-1",
    }

    start = await adapter.start_assistant_stream("personal-room", metadata=metadata)
    first = await adapter.push_assistant_delta("personal-room", "Hallo", metadata=metadata)
    second = await adapter.push_assistant_delta("personal-room", ", Welt.", metadata=metadata)
    end = await adapter.end_assistant_stream("personal-room", metadata=metadata)

    assert start.success is True
    assert first.success is True
    assert second.success is True
    assert end.success is True
    assert sent == [
        {
            "type": "assistant_llm_start",
            "room_id": "personal-room",
            "turn_id": "assistant-turn-1",
            "seq": 0,
            "participant_id": "whatsapp:+49123",
            "call_id": "call-1",
        },
        {
            "type": "assistant_llm_text",
            "room_id": "personal-room",
            "turn_id": "assistant-turn-1",
            "seq": 1,
            "text": "Hallo",
            "participant_id": "whatsapp:+49123",
            "call_id": "call-1",
        },
        {
            "type": "assistant_llm_text",
            "room_id": "personal-room",
            "turn_id": "assistant-turn-1",
            "seq": 2,
            "text": ", Welt.",
            "participant_id": "whatsapp:+49123",
            "call_id": "call-1",
        },
        {
            "type": "assistant_llm_end",
            "room_id": "personal-room",
            "turn_id": "assistant-turn-1",
            "seq": 3,
            "participant_id": "whatsapp:+49123",
            "call_id": "call-1",
        },
    ]
    assert adapter._turns["assistant-turn-1"]["planned_text"] == "Hallo, Welt."
    assert adapter._turns["assistant-turn-1"]["call_id"] == "call-1"


@pytest.mark.asyncio
async def test_voice_server_adapter_sends_abort_for_open_llm_stream():
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    sent = []

    class FakeWebSocket:
        closed = False

        async def send_json(self, payload):
            sent.append(payload)

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter._ws = FakeWebSocket()
    metadata = {
        "turn_id": "assistant-turn-1",
        "participant_id": "whatsapp:+49123",
        "thread_id": "call-1",
    }

    assert (await adapter.start_assistant_stream("personal-room", metadata=metadata)).success is True
    assert (await adapter.push_assistant_delta("personal-room", "Hallo", metadata=metadata)).success is True
    abort = await adapter.abort_assistant_stream("personal-room", metadata=metadata)

    assert abort.success is True
    assert sent[-1] == {
        "type": "assistant_llm_abort",
        "room_id": "personal-room",
        "turn_id": "assistant-turn-1",
        "seq": 2,
        "participant_id": "whatsapp:+49123",
        "call_id": "call-1",
    }
    assert "assistant-turn-1" not in adapter._stream_turns
    assert adapter._turns["assistant-turn-1"]["planned_text"] == "Hallo"


@pytest.mark.asyncio
async def test_voice_server_adapter_abort_after_failed_delta_preserves_sequence():
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    sent = []

    class FakeWebSocket:
        closed = False

        async def send_json(self, payload):
            if payload["type"] == "assistant_llm_text" and payload["seq"] == 2:
                raise OSError("send failed")
            sent.append(payload)

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter._ws = FakeWebSocket()
    metadata = {"turn_id": "assistant-turn-1", "participant_id": "caller"}

    assert (await adapter.start_assistant_stream("personal-room", metadata=metadata)).success is True
    assert (await adapter.push_assistant_delta("personal-room", "first", metadata=metadata)).success is True
    failed = await adapter.push_assistant_delta("personal-room", "second", metadata=metadata)
    abort = await adapter.abort_assistant_stream("personal-room", metadata=metadata)

    assert failed.success is False
    assert abort.success is True
    assert sent[-1] == {
        "type": "assistant_llm_abort",
        "room_id": "personal-room",
        "turn_id": "assistant-turn-1",
        "seq": 3,
        "participant_id": "caller",
    }
    assert "assistant-turn-1" not in adapter._stream_turns
    assert adapter._turns["assistant-turn-1"]["planned_text"] == "first"


@pytest.mark.asyncio
async def test_voice_server_adapter_reconciles_streamed_turn_after_transcript_persists():
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    messages = [
        {"role": "user", "content": "Tell me a long story"},
    ]
    rewrites = []

    class FakeWebSocket:
        closed = False

        async def send_json(self, _payload):
            pass

    class FakeSessionStore:
        def load_transcript(self, session_id):
            assert session_id == "session-1"
            return list(messages if not rewrites else rewrites[-1])

        def rewrite_transcript(self, session_id, new_messages):
            assert session_id == "session-1"
            rewrites.append(new_messages)

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter._ws = FakeWebSocket()
    adapter.set_session_store(FakeSessionStore())
    metadata = {"turn_id": "assistant-turn-1", "_hermes_session_id": "session-1"}

    assert (await adapter.start_assistant_stream("personal-room", metadata=metadata)).success is True
    assert (await adapter.push_assistant_delta("personal-room", "Hallo", metadata=metadata)).success is True
    assert (await adapter.push_assistant_delta("personal-room", ", Welt.", metadata=metadata)).success is True
    assert (await adapter.end_assistant_stream("personal-room", metadata=metadata)).success is True
    rewrites.clear()
    messages.append(
        {"role": "assistant", "content": "Hallo, Welt.", "voice_turn_id": "assistant-turn-1"}
    )

    await adapter.handle_room_event(
        {
            "type": "assistant_spoken",
            "room_id": "personal-room",
            "turn_id": "assistant-turn-1",
            "spoken_text": "Hallo",
            "interrupted": True,
        }
    )

    assert rewrites[-1][-1] == {
        "role": "assistant",
        "content": "Hallo",
        "voice_turn_id": "assistant-turn-1",
        "voice_interrupted": True,
        "voice_planned_content": "Hallo, Welt.",
        "voice_spoken_content": "Hallo",
    }


@pytest.mark.asyncio
async def test_voice_server_adapter_reconciles_spoken_event_received_before_stream_end(monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    monkeypatch.setattr("gateway.platforms.voice_server._SPOKEN_RECONCILE_RETRY_DELAYS", (0,))

    messages = [
        {"role": "user", "content": "Tell me a long story"},
    ]
    rewrites = []

    class FakeWebSocket:
        closed = False

        async def send_json(self, _payload):
            pass

    class FakeSessionStore:
        def load_transcript(self, session_id):
            assert session_id == "session-1"
            return list(messages if not rewrites else rewrites[-1])

        def rewrite_transcript(self, session_id, new_messages):
            assert session_id == "session-1"
            rewrites.append(new_messages)

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter._ws = FakeWebSocket()
    adapter.set_session_store(FakeSessionStore())
    metadata = {"turn_id": "assistant-turn-1", "_hermes_session_id": "session-1"}

    assert (await adapter.start_assistant_stream("personal-room", metadata=metadata)).success is True
    assert (await adapter.push_assistant_delta("personal-room", "Hallo", metadata=metadata)).success is True
    await adapter.handle_room_event(
        {
            "type": "assistant_spoken",
            "room_id": "personal-room",
            "turn_id": "assistant-turn-1",
            "spoken_text": "Hallo",
            "interrupted": True,
        }
    )

    assert rewrites == []

    assert (await adapter.push_assistant_delta("personal-room", ", Welt.", metadata=metadata)).success is True
    assert (await adapter.end_assistant_stream("personal-room", metadata=metadata)).success is True
    assert rewrites == []

    messages.append(
        {"role": "assistant", "content": "Hallo, Welt.", "voice_turn_id": "assistant-turn-1"}
    )
    await adapter._spoken_reconcile_tasks["assistant-turn-1"]

    assert rewrites[-1][-1]["content"] == "Hallo"
    assert rewrites[-1][-1]["voice_turn_id"] == "assistant-turn-1"


@pytest.mark.asyncio
async def test_voice_server_adapter_pending_spoken_waits_for_current_streamed_turn(monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    monkeypatch.setattr("gateway.platforms.voice_server._SPOKEN_RECONCILE_RETRY_DELAYS", (0,))

    messages = [
        {"role": "user", "content": "previous"},
        {"role": "assistant", "content": "Repeated answer"},
    ]
    rewrites = []

    class FakeWebSocket:
        closed = False

        async def send_json(self, _payload):
            pass

    class FakeSessionStore:
        def load_transcript(self, session_id):
            assert session_id == "session-1"
            return list(messages if not rewrites else rewrites[-1])

        def rewrite_transcript(self, session_id, new_messages):
            assert session_id == "session-1"
            rewrites.append(new_messages)

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter._ws = FakeWebSocket()
    adapter.set_session_store(FakeSessionStore())
    metadata = {"turn_id": "assistant-turn-2", "_hermes_session_id": "session-1"}

    assert (await adapter.start_assistant_stream("personal-room", metadata=metadata)).success is True
    assert (await adapter.push_assistant_delta("personal-room", "Repeated answer", metadata=metadata)).success is True
    await adapter.handle_room_event(
        {
            "type": "assistant_spoken",
            "room_id": "personal-room",
            "turn_id": "assistant-turn-2",
            "spoken_text": "Repeated",
            "interrupted": True,
        }
    )
    assert (await adapter.end_assistant_stream("personal-room", metadata=metadata)).success is True

    assert rewrites == []
    assert messages[-1].get("voice_turn_id") is None

    messages.append(
        {"role": "assistant", "content": "Repeated answer", "voice_turn_id": "assistant-turn-2"}
    )
    await adapter._spoken_reconcile_tasks["assistant-turn-2"]

    assert rewrites[-1][1] == {"role": "assistant", "content": "Repeated answer"}
    assert rewrites[-1][2]["content"] == "Repeated"
    assert rewrites[-1][2]["voice_turn_id"] == "assistant-turn-2"


@pytest.mark.asyncio
async def test_voice_server_adapter_does_not_reconcile_streamed_turn_by_text(monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    monkeypatch.setattr("gateway.platforms.voice_server._SPOKEN_RECONCILE_RETRY_DELAYS", (0,))

    rewrites = []

    class FakeWebSocket:
        closed = False

        async def send_json(self, _payload):
            pass

    class FakeSessionStore:
        def load_transcript(self, session_id):
            assert session_id == "session-1"
            return [
                {"role": "user", "content": "previous"},
                {"role": "assistant", "content": "Repeated answer"},
                {"role": "user", "content": "current"},
                {"role": "assistant", "content": "Repeated answer"},
            ]

        def rewrite_transcript(self, session_id, new_messages):
            assert session_id == "session-1"
            rewrites.append(new_messages)

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter._ws = FakeWebSocket()
    adapter.set_session_store(FakeSessionStore())
    metadata = {"turn_id": "assistant-turn-2", "_hermes_session_id": "session-1"}

    assert (await adapter.start_assistant_stream("personal-room", metadata=metadata)).success is True
    assert (await adapter.push_assistant_delta("personal-room", "Repeated answer", metadata=metadata)).success is True
    await adapter.handle_room_event(
        {
            "type": "assistant_spoken",
            "room_id": "personal-room",
            "turn_id": "assistant-turn-2",
            "spoken_text": "Repeated",
            "interrupted": True,
        }
    )
    assert (await adapter.end_assistant_stream("personal-room", metadata=metadata)).success is True
    await adapter._spoken_reconcile_tasks["assistant-turn-2"]

    assert rewrites == []


@pytest.mark.asyncio
async def test_voice_server_adapter_reconciles_streamed_turn_after_session_split():
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    messages_by_session = {
        "session-old": [{"role": "assistant", "content": "Old answer"}],
        "session-new": [
            {"role": "assistant", "content": "New answer", "voice_turn_id": "assistant-turn-1"}
        ],
    }
    rewrites = []

    class FakeWebSocket:
        closed = False

        async def send_json(self, _payload):
            pass

    class FakeSessionStore:
        def __init__(self):
            self._entries = {
                "voice-session-key": types.SimpleNamespace(session_id="session-old"),
        }

        def load_transcript(self, session_id):
            if rewrites and rewrites[-1][0] == session_id:
                return list(rewrites[-1][1])
            return list(messages_by_session[session_id])

        def rewrite_transcript(self, session_id, new_messages):
            rewrites.append((session_id, new_messages))

    store = FakeSessionStore()
    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter._ws = FakeWebSocket()
    adapter.set_session_store(store)
    metadata = {
        "turn_id": "assistant-turn-1",
        "_hermes_session_key": "voice-session-key",
        "_hermes_session_id": "session-old",
    }

    assert (await adapter.start_assistant_stream("personal-room", metadata=metadata)).success is True
    assert (await adapter.push_assistant_delta("personal-room", "New answer", metadata=metadata)).success is True
    assert (await adapter.end_assistant_stream("personal-room", metadata=metadata)).success is True
    store._entries["voice-session-key"].session_id = "session-new"

    await adapter.handle_room_event(
        {
            "type": "assistant_spoken",
            "room_id": "personal-room",
            "turn_id": "assistant-turn-1",
            "spoken_text": "New",
            "interrupted": True,
        }
    )

    assert rewrites[-1][0] == "session-new"
    assert rewrites[-1][1][-1]["content"] == "New"
    assert rewrites[-1][1][-1]["voice_turn_id"] == "assistant-turn-1"


@pytest.mark.asyncio
async def test_voice_server_adapter_reconciles_late_spoken_turn_against_original_session():
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    messages_by_session = {
        "session-old": [
            {"role": "assistant", "content": "Old answer", "voice_turn_id": "assistant-turn-1"}
        ],
        "session-new": [{"role": "assistant", "content": "Different answer"}],
    }
    rewrites = []

    class FakeWebSocket:
        closed = False

        async def send_json(self, _payload):
            pass

    class FakeSessionStore:
        def __init__(self):
            self._entries = {
                "voice-session-key": types.SimpleNamespace(session_id="session-new"),
            }

        def load_transcript(self, session_id):
            if rewrites and rewrites[-1][0] == session_id:
                return list(rewrites[-1][1])
            return list(messages_by_session[session_id])

        def rewrite_transcript(self, session_id, new_messages):
            rewrites.append((session_id, new_messages))

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter._ws = FakeWebSocket()
    adapter.set_session_store(FakeSessionStore())
    adapter._turns["assistant-turn-1"] = {
        "room_id": "personal-room",
        "planned_text": "Old answer",
        "session_key": "voice-session-key",
        "session_id": "session-old",
    }

    await adapter.handle_room_event(
        {
            "type": "assistant_spoken",
            "room_id": "personal-room",
            "turn_id": "assistant-turn-1",
            "spoken_text": "Old",
            "interrupted": True,
        }
    )

    assert rewrites[-1][0] == "session-old"
    assert rewrites[-1][1][-1]["content"] == "Old"
    assert rewrites[-1][1][-1]["voice_turn_id"] == "assistant-turn-1"


@pytest.mark.asyncio
async def test_voice_server_adapter_stream_end_does_not_stamp_prior_assistant_turn():
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    rewrites = []

    class FakeWebSocket:
        closed = False

        async def send_json(self, _payload):
            pass

    class FakeSessionStore:
        def load_transcript(self, session_id):
            assert session_id == "session-1"
            return [
                {"role": "user", "content": "previous"},
                {"role": "assistant", "content": "Previous answer"},
            ]

        def rewrite_transcript(self, session_id, new_messages):
            assert session_id == "session-1"
            rewrites.append(new_messages)

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter._ws = FakeWebSocket()
    adapter.set_session_store(FakeSessionStore())
    metadata = {"turn_id": "assistant-turn-2", "_hermes_session_id": "session-1"}

    assert (await adapter.start_assistant_stream("personal-room", metadata=metadata)).success is True
    assert (await adapter.push_assistant_delta("personal-room", "New answer", metadata=metadata)).success is True
    assert (await adapter.end_assistant_stream("personal-room", metadata=metadata)).success is True

    assert rewrites == []


@pytest.mark.asyncio
async def test_voice_server_adapter_spoken_after_stream_end_does_not_stamp_prior_mismatched_turn():
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    rewrites = []

    class FakeWebSocket:
        closed = False

        async def send_json(self, _payload):
            pass

    class FakeSessionStore:
        def load_transcript(self, session_id):
            assert session_id == "session-1"
            return [
                {"role": "user", "content": "previous"},
                {"role": "assistant", "content": "Previous answer"},
            ]

        def rewrite_transcript(self, session_id, new_messages):
            assert session_id == "session-1"
            rewrites.append(new_messages)

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter._ws = FakeWebSocket()
    adapter.set_session_store(FakeSessionStore())
    metadata = {"turn_id": "assistant-turn-2", "_hermes_session_id": "session-1"}

    assert (await adapter.start_assistant_stream("personal-room", metadata=metadata)).success is True
    assert (await adapter.push_assistant_delta("personal-room", "New answer", metadata=metadata)).success is True
    assert (await adapter.end_assistant_stream("personal-room", metadata=metadata)).success is True
    await adapter.handle_room_event(
        {
            "type": "assistant_spoken",
            "room_id": "personal-room",
            "turn_id": "assistant-turn-2",
            "spoken_text": "New",
            "interrupted": True,
        }
    )

    assert rewrites == []


@pytest.mark.asyncio
async def test_voice_server_adapter_push_delta_without_metadata_auto_starts_stream():
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    sent = []

    class FakeWebSocket:
        closed = False

        async def send_json(self, payload):
            sent.append(payload)

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter._ws = FakeWebSocket()

    result = await adapter.push_assistant_delta("personal-room", "Hallo")

    assert result.success is True
    assert [payload["type"] for payload in sent] == [
        "assistant_llm_start",
        "assistant_llm_text",
    ]
    assert sent[0]["turn_id"] == sent[1]["turn_id"]
    assert list(adapter._stream_turns) == [sent[0]["turn_id"]]


@pytest.mark.asyncio
async def test_voice_server_adapter_preserves_stream_turn_until_abort_on_text_failure():
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    sent = []

    class FakeWebSocket:
        closed = False

        async def send_json(self, payload):
            sent.append(payload)
            if payload["type"] == "assistant_llm_text":
                raise OSError("connection lost")

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter._ws = FakeWebSocket()
    metadata = {"turn_id": "assistant-turn-1"}

    assert (await adapter.start_assistant_stream("personal-room", metadata=metadata)).success is True
    result = await adapter.push_assistant_delta("personal-room", "Hallo", metadata=metadata)
    abort = await adapter.abort_assistant_stream("personal-room", metadata=metadata)

    assert result.success is False
    assert abort.success is True
    assert sent[-1] == {
        "type": "assistant_llm_abort",
        "room_id": "personal-room",
        "turn_id": "assistant-turn-1",
        "seq": 2,
    }
    assert "assistant-turn-1" not in adapter._stream_turns
    assert "assistant-turn-1" not in adapter._turns


@pytest.mark.asyncio
async def test_voice_server_adapter_preserves_accepted_stream_turn_on_end_failure():
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    class FakeWebSocket:
        closed = False

        async def send_json(self, payload):
            if payload["type"] == "assistant_llm_end":
                raise OSError("connection lost")

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter._ws = FakeWebSocket()
    metadata = {"turn_id": "assistant-turn-1"}

    assert (await adapter.start_assistant_stream("personal-room", metadata=metadata)).success is True
    assert (await adapter.push_assistant_delta("personal-room", "Hallo", metadata=metadata)).success is True
    result = await adapter.end_assistant_stream("personal-room", metadata=metadata)

    assert result.success is False
    assert "assistant-turn-1" not in adapter._stream_turns
    assert adapter._turns["assistant-turn-1"]["planned_text"] == "Hallo"


@pytest.mark.asyncio
async def test_voice_server_adapter_rewrites_history_from_interrupted_spoken_event():
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    messages = [
        {"role": "user", "content": "Tell me a long story"},
        {"role": "assistant", "content": "Full answer", "voice_turn_id": "voice-turn-1"},
    ]
    rewrites = []

    class FakeWebSocket:
        closed = False

        async def send_json(self, _payload):
            pass

    class FakeSessionStore:
        def load_transcript(self, session_id):
            assert session_id == "session-1"
            return list(messages if not rewrites else rewrites[-1])

        def rewrite_transcript(self, session_id, new_messages):
            assert session_id == "session-1"
            rewrites.append(new_messages)

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter._ws = FakeWebSocket()
    adapter.set_session_store(FakeSessionStore())

    await adapter.send(
        "personal-room",
        "Full answer",
        metadata={"turn_id": "voice-turn-1", "_hermes_session_id": "session-1"},
    )
    await adapter.handle_room_event(
        {
            "type": "assistant_spoken",
            "room_id": "personal-room",
            "turn_id": "voice-turn-1",
            "spoken_text": "Full",
            "interrupted": True,
        }
    )

    assert rewrites[0][-1]["voice_turn_id"] == "voice-turn-1"
    assert rewrites[-1][-1] == {
        "role": "assistant",
        "content": "Full",
        "voice_turn_id": "voice-turn-1",
        "voice_interrupted": True,
        "voice_planned_content": "Full answer",
        "voice_spoken_content": "Full",
    }


@pytest.mark.asyncio
async def test_voice_server_adapter_deletes_unspoken_interrupted_turn():
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    rewrites = []

    class FakeWebSocket:
        closed = False

        async def send_json(self, _payload):
            pass

    class FakeSessionStore:
        def load_transcript(self, _session_id):
            if rewrites:
                return list(rewrites[-1])
            return [
                {"role": "user", "content": "Tell me a long story"},
                {"role": "assistant", "content": "Full answer", "voice_turn_id": "voice-turn-1"},
            ]

        def rewrite_transcript(self, _session_id, new_messages):
            rewrites.append(new_messages)

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter._ws = FakeWebSocket()
    adapter.set_session_store(FakeSessionStore())

    await adapter.send(
        "personal-room",
        "Full answer",
        metadata={"turn_id": "voice-turn-1", "_hermes_session_id": "session-1"},
    )
    await adapter.handle_room_event(
        {
            "type": "assistant_spoken",
            "room_id": "personal-room",
            "turn_id": "voice-turn-1",
            "spoken_text": "",
            "interrupted": True,
        }
    )

    assert rewrites[-1] == [{"role": "user", "content": "Tell me a long story"}]


@pytest.mark.asyncio
async def test_voice_server_adapter_handles_spoken_event_during_send():
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )

    class FakeWebSocket:
        closed = False

        async def send_json(self, _payload):
            await adapter.handle_room_event(
                {
                    "type": "assistant_spoken",
                    "room_id": "personal-room",
                    "turn_id": "voice-turn-1",
                    "spoken_text": "Full",
                    "interrupted": True,
                }
            )

    adapter._ws = FakeWebSocket()

    result = await adapter.send(
        "personal-room",
        "Full answer",
        metadata={"turn_id": "voice-turn-1", "_hermes_session_id": "session-1"},
    )

    assert result.success is True


@pytest.mark.asyncio
async def test_voice_server_adapter_uses_gateway_reply_turn_id_for_full_send():
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    payloads = []

    class FakeWebSocket:
        closed = False

        async def send_json(self, payload):
            payloads.append(payload)

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter._ws = FakeWebSocket()
    adapter.set_next_reply_turn_id("voice-session-key", "voice-turn-1")

    result = await adapter._send_with_retry(
        "personal-room",
        "Full answer",
        metadata={
            "_hermes_session_key": "voice-session-key",
            "_hermes_session_id": "session-1",
        },
    )

    assert result.success is True
    assert payloads[-1]["type"] == "assistant_reply"
    assert payloads[-1]["turn_id"] == "voice-turn-1"
    assert adapter._turns["voice-turn-1"]["session_id"] == "session-1"


@pytest.mark.asyncio
async def test_voice_server_adapter_clears_unused_gateway_reply_turn_id():
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    payloads = []

    class FakeWebSocket:
        closed = False

        async def send_json(self, payload):
            payloads.append(payload)

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter._ws = FakeWebSocket()
    adapter.set_next_reply_turn_id("voice-session-key", "stale-turn")
    adapter.clear_next_reply_turn_id("voice-session-key")

    result = await adapter._send_with_retry(
        "personal-room",
        "Next answer",
        metadata={
            "_hermes_session_key": "voice-session-key",
            "_hermes_session_id": "session-1",
        },
    )

    assert result.success is True
    assert payloads[-1]["turn_id"] != "stale-turn"


@pytest.mark.asyncio
async def test_voice_server_adapter_retries_interrupted_spoken_reconciliation(monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    monkeypatch.setattr("gateway.platforms.voice_server._SPOKEN_RECONCILE_RETRY_DELAYS", (0, 0))

    current_messages = [
        {"role": "user", "content": "Tell me a long story"},
        {"role": "assistant", "content": "Full answer"},
    ]

    class FakeWebSocket:
        closed = False

        async def send_json(self, _payload):
            pass

    class FakeSessionStore:
        def load_transcript(self, _session_id):
            return list(current_messages)

        def rewrite_transcript(self, _session_id, new_messages):
            current_messages[:] = new_messages

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter._ws = FakeWebSocket()
    adapter.set_session_store(FakeSessionStore())

    await adapter.send(
        "personal-room",
        "Full answer",
        metadata={"turn_id": "voice-turn-1", "_hermes_session_id": "session-1"},
    )
    await adapter.handle_room_event(
        {
            "type": "assistant_spoken",
            "room_id": "personal-room",
            "turn_id": "voice-turn-1",
            "spoken_text": "Full",
            "interrupted": True,
        }
    )
    assert current_messages[-1]["content"] == "Full answer"

    current_messages[-1] = {
        "role": "assistant",
        "content": "Full answer",
        "voice_turn_id": "voice-turn-1",
    }
    await adapter._spoken_reconcile_tasks["voice-turn-1"]

    assert current_messages[-1]["content"] == "Full"
    assert current_messages[-1]["voice_interrupted"] is True
    assert current_messages[-1]["voice_planned_content"] == "Full answer"
    assert current_messages[-1]["voice_spoken_content"] == "Full"
    assert "voice-turn-1" not in adapter._turns


@pytest.mark.asyncio
async def test_voice_server_adapter_expires_unmatched_spoken_retry(monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    monkeypatch.setattr("gateway.platforms.voice_server._SPOKEN_RECONCILE_RETRY_DELAYS", (0,))

    class FakeSessionStore:
        def load_transcript(self, _session_id):
            return [{"role": "assistant", "content": "Full answer"}]

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter.set_session_store(FakeSessionStore())
    adapter._turns["voice-turn-1"] = {"session_id": "session-1"}

    await adapter.handle_room_event(
        {
            "type": "assistant_spoken",
            "room_id": "personal-room",
            "turn_id": "voice-turn-1",
            "spoken_text": "Full",
            "interrupted": True,
        }
    )
    await adapter._spoken_reconcile_tasks["voice-turn-1"]

    assert "voice-turn-1" not in adapter._turns
    assert "voice-turn-1" not in adapter._pending_stream_spoken


@pytest.mark.asyncio
async def test_voice_server_adapter_held_spoken_retry_survives_until_release(monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    monkeypatch.setattr("gateway.platforms.voice_server._SPOKEN_RECONCILE_RETRY_DELAYS", (0,))

    class FakeSessionStore:
        def load_transcript(self, _session_id):
            return [{"role": "assistant", "content": "Full answer"}]

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter.set_session_store(FakeSessionStore())
    adapter._turns["voice-turn-1"] = {"session_id": "session-1"}
    adapter.hold_pending_spoken_turn("voice-turn-1")

    await adapter.handle_room_event(
        {
            "type": "assistant_spoken",
            "room_id": "personal-room",
            "turn_id": "voice-turn-1",
            "spoken_text": "Full",
            "interrupted": True,
        }
    )
    await adapter._spoken_reconcile_tasks["voice-turn-1"]

    assert "voice-turn-1" in adapter._turns
    assert "voice-turn-1" in adapter._pending_stream_spoken

    adapter.release_pending_spoken_turn("voice-turn-1")
    await adapter._spoken_reconcile_tasks["voice-turn-1"]

    assert "voice-turn-1" not in adapter._turns
    assert "voice-turn-1" not in adapter._pending_stream_spoken


@pytest.mark.asyncio
async def test_voice_server_adapter_falls_back_to_jsonl_for_voice_turn_metadata(tmp_path, monkeypatch):
    import json

    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    monkeypatch.setattr("gateway.platforms.voice_server._SPOKEN_RECONCILE_RETRY_DELAYS", ())

    transcript_path = tmp_path / "session-1.jsonl"
    transcript_path.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "Tell me a long story"}),
                json.dumps(
                    {
                        "role": "assistant",
                        "content": "Full answer",
                        "voice_turn_id": "voice-turn-1",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rewrites = []

    class FakeSessionStore:
        def load_transcript(self, _session_id):
            return [
                {"role": "user", "content": "Tell me a long story"},
                {"role": "assistant", "content": "Full answer"},
            ]

        def get_transcript_path(self, _session_id):
            return transcript_path

        def rewrite_transcript(self, _session_id, new_messages):
            rewrites.append(new_messages)

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter.set_session_store(FakeSessionStore())
    adapter._turns["voice-turn-1"] = {"session_id": "session-1"}

    await adapter.handle_room_event(
        {
            "type": "assistant_spoken",
            "room_id": "personal-room",
            "turn_id": "voice-turn-1",
            "spoken_text": "Full",
            "interrupted": True,
        }
    )
    await adapter._spoken_reconcile_tasks["voice-turn-1"]

    assert rewrites[-1][-1]["content"] == "Full"
    assert rewrites[-1][-1]["voice_interrupted"] is True
    assert rewrites[-1][-1]["voice_planned_content"] == "Full answer"


@pytest.mark.asyncio
async def test_voice_server_adapter_stale_retry_task_does_not_clear_replacement(monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    monkeypatch.setattr("gateway.platforms.voice_server._SPOKEN_RECONCILE_RETRY_DELAYS", (3600,))

    payload = {
        "type": "assistant_spoken",
        "room_id": "personal-room",
        "turn_id": "voice-turn-1",
        "spoken_text": "Full",
        "interrupted": True,
    }
    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter._turns["voice-turn-1"] = {"session_id": "session-1"}

    await adapter.handle_room_event(payload)
    first_task = adapter._spoken_reconcile_tasks["voice-turn-1"]
    await adapter.handle_room_event(payload)
    second_task = adapter._spoken_reconcile_tasks["voice-turn-1"]
    await asyncio.sleep(0)

    assert first_task.cancelled()
    assert adapter._spoken_reconcile_tasks["voice-turn-1"] is second_task
    assert "voice-turn-1" in adapter._turns

    second_task.cancel()
    await asyncio.gather(second_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_voice_server_adapter_reuses_generated_turn_id_across_send_retry(monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr("gateway.platforms.base.asyncio.sleep", no_sleep)
    monkeypatch.setattr("gateway.platforms.base.random.uniform", lambda _start, _end: 0)

    sent = []
    class FakeWebSocket:
        closed = False

        def __init__(self):
            self.calls = 0

        async def send_json(self, payload):
            self.calls += 1
            if self.calls == 1:
                raise OSError("temporary send failure")
            sent.append(payload)

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter._ws = FakeWebSocket()

    metadata = {"_hermes_session_id": "session-1"}
    result = await adapter._send_with_retry(
        "personal-room",
        "Full answer",
        metadata=metadata,
        max_retries=1,
        base_delay=0,
    )

    assert result.success is True
    assert len(sent) == 1
    turn_id = sent[0]["turn_id"]
    assert turn_id.startswith("voice_server-")
    assert metadata["_voice_server_turn_id"] == turn_id

    await adapter.handle_room_event(
        {
            "type": "assistant_spoken",
            "room_id": "personal-room",
            "turn_id": turn_id,
            "spoken_text": "Full",
            "interrupted": True,
        }
    )


@pytest.mark.asyncio
async def test_voice_server_adapter_send_failure_returns_error_without_turn_state():
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    class FakeWebSocket:
        closed = False

        async def send_json(self, _payload):
            raise RuntimeError("send failed")

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter._ws = FakeWebSocket()

    result = await adapter.send(
        "personal-room",
        "Full answer",
        metadata={"turn_id": "voice-turn-1", "_hermes_session_id": "session-1"},
    )

    assert result.success is False
    assert "voice-turn-1" not in adapter._turns


@pytest.mark.asyncio
async def test_voice_server_adapter_rejects_unconfigured_send_room():
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    sent = []

    class FakeWebSocket:
        closed = False

        async def send_json(self, payload):
            sent.append(payload)

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter._ws = FakeWebSocket()

    result = await adapter.send("other-room", "Hallo")

    assert result.success is False
    assert "other-room" in result.error
    assert sent == []


@pytest.mark.asyncio
async def test_voice_server_adapter_treats_string_false_multi_room_as_false():
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    sent = []

    class FakeWebSocket:
        closed = False

        async def send_json(self, payload):
            sent.append(payload)

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "url": "ws://127.0.0.1:7860/events",
                "room_id": "personal-room",
                "multi_room": "false",
            },
        )
    )
    adapter._ws = FakeWebSocket()

    result = await adapter.send("other-room", "Hallo")

    assert adapter.multi_room is False
    assert result.success is False
    assert sent == []


@pytest.mark.asyncio
async def test_voice_server_adapter_sends_outbound_call_command_for_string_target():
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    sent = []

    class FakeWebSocket:
        closed = False

        async def send_json(self, payload):
            sent.append(payload)

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter._ws = FakeWebSocket()

    result = await adapter.start_outbound_call(
        target="+491234567",
        room_id="personal-room",
    )

    assert result.success is True
    assert sent == [
        {
            "type": "start_outbound_call",
            "room_id": "personal-room",
            "call_id": sent[0]["call_id"],
            "target": "+491234567",
        }
    ]
    assert sent[0]["call_id"].startswith("voice-call-")
    assert "session_id" not in sent[0]


@pytest.mark.asyncio
async def test_voice_server_adapter_sends_outbound_call_command_for_object_target():
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    sent = []

    class FakeWebSocket:
        closed = False

        async def send_json(self, payload):
            sent.append(payload)

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter._ws = FakeWebSocket()
    target = {"provider": "example", "address": "+491234567"}

    result = await adapter.start_outbound_call(
        target=target,
        room_id="personal-room",
        context={"preload": [{"role": "system", "content": "Use German."}]},
        metadata={"reason": "scheduled"},
        call_id="call-1",
    )

    assert result.success is True
    assert sent == [
        {
            "type": "start_outbound_call",
            "room_id": "personal-room",
            "call_id": "call-1",
            "target": target,
            "context": {"preload": [{"role": "system", "content": "Use German."}]},
            "metadata": {"reason": "scheduled"},
        }
    ]
    assert "session_id" not in sent[0]


@pytest.mark.asyncio
async def test_voice_server_adapter_binds_outbound_call_started_to_fresh_sessions():
    from gateway.config import PlatformConfig
    from gateway.platforms.base import build_session_key
    from gateway.platforms.voice_server import VoiceServerAdapter

    sessions = []
    handled = []

    class FakeSessionStore:
        def get_or_create_session(self, source, force_new=False):
            sessions.append((source, force_new))
            return types.SimpleNamespace(session_id=f"session-{len(sessions)}")

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter.set_session_store(FakeSessionStore())
    adapter.set_message_handler(lambda event: handled.append(event))

    await adapter.handle_room_event(
        {
            "type": "call_started",
            "room_id": "personal-room",
            "call_id": "call-1",
            "target": "+491234567",
        }
    )
    await adapter.handle_room_event(
        {
            "type": "call_started",
            "room_id": "personal-room",
            "call_id": "call-2",
            "target": {"provider": "example", "address": "+491234567"},
        }
    )
    await adapter.handle_room_event(
        {
            "type": "call_started",
            "room_id": "personal-room",
            "call_id": "call-2",
            "target": {"provider": "example", "address": "+491234567"},
        }
    )

    assert handled == []
    assert len(sessions) == 3
    first_source, first_force_new = sessions[0]
    second_source, second_force_new = sessions[1]
    duplicate_source, duplicate_force_new = sessions[2]
    assert first_force_new is False
    assert second_force_new is False
    assert duplicate_force_new is False
    assert first_source.user_id == "caller"
    assert first_source.thread_id == "call-1"
    assert second_source.user_id == "caller"
    assert second_source.thread_id == "call-2"
    assert duplicate_source.thread_id == "call-2"
    assert build_session_key(first_source) != build_session_key(second_source)

    first_transcript = adapter.event_to_message_event(
        {
            "type": "transcript",
            "room_id": "personal-room",
            "call_id": "call-1",
            "target": "+491234567",
            "text": "hello",
        }
    )
    second_transcript = adapter.event_to_message_event(
        {
            "type": "transcript",
            "room_id": "personal-room",
            "call_id": "call-2",
            "target": {"provider": "example", "address": "+491234567"},
            "text": "hello again",
        }
    )

    assert first_transcript is not None
    assert second_transcript is not None
    assert first_transcript.source.user_id == "caller"
    assert second_transcript.source.user_id == "caller"
    assert build_session_key(
        first_transcript.source,
        thread_sessions_per_user=True,
    ) == build_session_key(first_source, thread_sessions_per_user=True)
    assert build_session_key(
        second_transcript.source,
        thread_sessions_per_user=True,
    ) == build_session_key(second_source, thread_sessions_per_user=True)
    assert build_session_key(
        first_transcript.source,
        thread_sessions_per_user=True,
    ) != build_session_key(second_transcript.source, thread_sessions_per_user=True)


@pytest.mark.asyncio
async def test_voice_server_adapter_disconnect_does_not_cancel_current_listener_task():
    import asyncio

    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    closed = []

    class FakeWebSocket:
        closed = False

        async def close(self):
            closed.append("ws")
            self.closed = True

    class FakeSession:
        closed = False

        async def close(self):
            closed.append("session")
            self.closed = True

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter._listen_task = asyncio.current_task()
    adapter._ws = FakeWebSocket()
    adapter._session = FakeSession()

    await adapter.disconnect()

    assert closed == ["ws", "session"]
    assert adapter._listen_task is None
    assert not asyncio.current_task().cancelled()


@pytest.mark.asyncio
async def test_voice_server_adapter_disables_gateway_auto_tts_and_voice_media():
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter._auto_tts_enabled_chats.add("personal-room")

    assert adapter._should_auto_tts_for_chat("personal-room") is False
    assert (await adapter.play_tts("personal-room", "/tmp/audio.wav")).success is True
    assert (await adapter.send_voice("personal-room", "/tmp/audio.wav")).success is True


@pytest.mark.asyncio
async def test_voice_server_adapter_send_generates_turn_id_for_room():
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    sent = []

    class FakeWebSocket:
        closed = False

        async def send_json(self, payload):
            sent.append(payload)

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter._ws = FakeWebSocket()

    result = await adapter.send("personal-room", "planned response")

    assert result.success is True
    assert sent[0]["turn_id"].startswith("voice_server-")
    assert sent[0]["text"] == "planned response"


@pytest.mark.asyncio
async def test_voice_server_adapter_send_failure_returns_retryable_error():
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    class FakeWebSocket:
        closed = False

        async def send_json(self, payload):
            raise OSError("connection lost")

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter._ws = FakeWebSocket()

    result = await adapter.send("personal-room", "planned response")

    assert result.success is False
    assert result.retryable is True
    assert "connection lost" in result.error


@pytest.mark.asyncio
async def test_voice_server_listener_close_notifies_gateway_for_reconnect():
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    closed = []
    notified = []

    class FakeWebSocket:
        closed = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def close(self):
            closed.append("ws")
            self.closed = True

    class FakeSession:
        closed = False

        async def close(self):
            closed.append("session")
            self.closed = True

    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )
    adapter._ws = FakeWebSocket()
    adapter._session = FakeSession()
    adapter._running = True

    async def on_fatal(failed_adapter):
        notified.append(failed_adapter.fatal_error_code)

    adapter.set_fatal_error_handler(on_fatal)

    await adapter._listen_loop()

    assert adapter.is_connected is False
    assert adapter.fatal_error_code == "connection_closed"
    assert closed == ["ws", "session"]
    assert notified == ["connection_closed"]


@pytest.mark.asyncio
async def test_voice_server_adapter_ignores_unconfigured_room_events():
    from gateway.config import PlatformConfig
    from gateway.platforms.voice_server import VoiceServerAdapter

    handled = []
    adapter = VoiceServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"url": "ws://127.0.0.1:7860/events", "room_id": "personal-room"},
        )
    )

    async def handle(event):
        handled.append(event)

    adapter.set_message_handler(handle)

    await adapter.handle_room_event(
        {"type": "transcript", "room_id": "other-room", "text": "hello"}
    )
    await adapter.handle_room_event(
        {"type": "assistant_spoken", "room_id": "other-room", "turn_id": "t1"}
    )

    assert handled == []
