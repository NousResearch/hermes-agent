import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.session import SessionSource
from plugins.platforms.zalo.adapter import (
    HzcaClient,
    ZaloAdapter,
    ZaloBackendConfig,
    config_from_platform,
    sanitize_sse_event,
    validate_local_backend_url,
)


class FakeClient:
    def __init__(self, events=None):
        self.sent = []
        self.events = events or []

    async def get_health(self):
        return {"status": "ok"}

    async def get_me(self):
        return {"id": "self-1"}

    async def send_text(self, *, thread_id, text, is_group):
        self.sent.append({"threadId": thread_id, "message": text, "isGroup": is_group})
        return {"success": True, "msgId": "out-1"}

    async def send_typing(self, *, thread_id, is_group):
        self.sent.append({"typing": True, "threadId": thread_id, "isGroup": is_group})

    async def iter_events(self, stop_event):
        for event in self.events:
            if stop_event.is_set():
                break
            yield event


def config(extra=None):
    return PlatformConfig(enabled=True, extra=extra or {})


def _clear_gateway_auth_env(monkeypatch):
    for key in (
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "ZALO_ALLOWED_USER_IDS",
        "ZALO_ALLOWED_USERS",
        "ZALO_ALLOWED_GROUP_IDS",
        "ZALO_ALLOWED_GROUPS",
        "ZALO_ALLOW_ALL_USERS",
        "ZALO_GROUP_REQUIRE_MENTION",
        "ZALO_GROUP_PREFIXES",
        "ZALO_FREE_RESPONSE_GROUP_IDS",
        "ZALO_REQUEST_TIMEOUT",
        "ZALO_SSE_TIMEOUT",
        "ZALO_SSE_READ_TIMEOUT",
    ):
        monkeypatch.delenv(key, raising=False)


def _make_bare_gateway_runner(adapter):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform("zalo"): adapter}
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_args, **_kwargs: False)
    return runner


def test_local_only_backend_validation_rejects_remote_hosts():
    assert validate_local_backend_url("http://127.0.0.1:56789") == "http://127.0.0.1:56789"
    assert validate_local_backend_url("http://localhost:56789/") == "http://localhost:56789"
    assert validate_local_backend_url("HTTP://LOCALHOST:56789/") == "http://localhost:56789"
    with pytest.raises(ValueError, match="non-loopback"):
        validate_local_backend_url("http://192.168.1.10:56789")
    with pytest.raises(ValueError, match="non-loopback"):
        validate_local_backend_url("https://example.com")
    assert validate_local_backend_url("http://192.168.1.10:56789", allow_unsafe_remote=True)


def test_backend_validation_rejects_non_origin_components():
    for value in (
        "http://127.0.0.1:56789/api",
        "http://127.0.0.1:56789?token=secret",
        "http://127.0.0.1:56789#fragment",
        "http://user:pass@127.0.0.1:56789",
    ):
        with pytest.raises(ValueError, match="origin"):
            validate_local_backend_url(value, allow_unsafe_remote=True)


def test_policy_config_keeps_sse_timeout_separate_from_rest_request_timeout(monkeypatch):
    _clear_gateway_auth_env(monkeypatch)
    default_backend = config_from_platform(config())
    assert default_backend.request_timeout == 30.0
    assert default_backend.sse_timeout is None

    backend = config_from_platform(config({"request_timeout": 2, "sse_timeout": 3600}))

    assert backend.request_timeout == 2.0
    assert backend.sse_timeout == 3600.0


def test_sanitizer_drops_unneeded_raw_fields():
    frame = Path("plugins/platforms/zalo/fixtures/sse-message.frame").read_text()
    payload = json.loads(next(line[5:].strip() for line in frame.splitlines() if line.startswith("data:")))
    sanitized = sanitize_sse_event(payload)
    assert sanitized["msgId"] == "m-1"
    assert sanitized["cliMsgId"] == "c-1"
    assert sanitized["threadId"] == "g-1"
    assert sanitized["senderId"] == "u-1"
    assert sanitized["isGroup"] is True
    assert sanitized["quote"] == {"msgId": "q-1", "cliMsgId": "qc-1", "senderId": "u-2"}
    assert "sensitive" not in sanitized
    assert "senderName" not in sanitized
    assert "timestamp" not in sanitized
    assert "mentions" not in sanitized
    assert "atAll" not in sanitized


def test_sanitizer_prefers_hzca_v2_canonical_fields():
    sanitized = sanitize_sse_event({
        "type": "message",
        "data": {
            "schemaVersion": 2,
            "messageId": "canonical-message",
            "id": "legacy-id",
            "msgId": "legacy-msg",
            "cliMsgId": "legacy-cli",
            "threadId": "g-1",
            "senderId": "u-1",
            "text": "@Bot canonical hello",
            "content": "legacy content",
            "messageType": "image",
            "type": "text",
            "msgType": "video",
            "mentions": [{"uid": "self-1", "text": "@Bot", "pos": 0, "len": 4}],
            "atAll": False,
            "quote": {"messageId": "canonical-quote", "senderId": "self-1"},
            "raw": {
                "mentionList": [{"uid": "other-1"}],
                "isAtAll": True,
                "quotedMessage": {"msgId": "legacy-quote", "uidFrom": "other-1"},
            },
            "isGroup": True,
        },
    })

    assert sanitized["id"] == "canonical-message"
    assert sanitized["content"] == "@Bot canonical hello"
    assert sanitized["type"] == "image"
    assert sanitized["quote"] == {
        "msgId": None,
        "cliMsgId": None,
        "senderId": "self-1",
        "messageId": "canonical-quote",
    }
    assert "mentions" not in sanitized
    assert "atAll" not in sanitized
    assert "senderName" not in sanitized
    assert "timestamp" not in sanitized


def test_gateway_authorizes_zalo_source_from_adapter_allowed_group(monkeypatch):
    _clear_gateway_auth_env(monkeypatch)
    adapter = ZaloAdapter(config({"allowed_group_ids": ["g-1"]}), client=FakeClient())
    runner = _make_bare_gateway_runner(adapter)

    source = SessionSource(
        platform=Platform("zalo"),
        chat_id="g-1",
        chat_name="g-1",
        chat_type="group",
        user_id="3887207304695409270",
        user_name="Dsphn",
        thread_id="g-1",
    )

    assert runner._is_user_authorized(source) is True


def test_gateway_rejects_zalo_source_from_group_not_allowed_by_adapter(monkeypatch):
    _clear_gateway_auth_env(monkeypatch)
    adapter = ZaloAdapter(config({"allowed_group_ids": ["g-1"]}), client=FakeClient())
    runner = _make_bare_gateway_runner(adapter)

    source = SessionSource(
        platform=Platform("zalo"),
        chat_id="g-2",
        chat_name="g-2",
        chat_type="group",
        user_id="3887207304695409270",
        user_name="Dsphn",
        thread_id="g-2",
    )

    assert runner._is_user_authorized(source) is False


def test_gateway_authorization_docstring_mentions_adapter_hook_order():
    from gateway.run import GatewayRunner

    doc = GatewayRunner._is_user_authorized.__doc__ or ""

    assert "adapter-level is_source_authorized" in doc
    assert "before environment allowlists" in doc


def test_policy_config_supports_legacy_env_fallbacks(monkeypatch):
    _clear_gateway_auth_env(monkeypatch)
    monkeypatch.setenv("ZALO_ALLOWED_USERS", "u-1,u-2")
    monkeypatch.setenv("ZALO_ALLOWED_GROUPS", "g-1,g-2")
    monkeypatch.setenv("ZALO_ALLOW_ALL_USERS", "true")
    monkeypatch.setenv("ZALO_GROUP_REQUIRE_MENTION", "false")
    monkeypatch.setenv("ZALO_GROUP_PREFIXES", "/ngan,/hermes")
    monkeypatch.setenv("ZALO_FREE_RESPONSE_GROUP_IDS", "g-2")

    backend = config_from_platform(config())

    assert backend.allowed_user_ids == ("u-1", "u-2")
    assert backend.allowed_group_ids == ("g-1", "g-2")
    assert backend.allow_all_users is True
    assert backend.require_mention is False
    assert backend.prefixes == ("/ngan", "/hermes")
    assert backend.free_response_group_ids == ("g-2",)


@pytest.mark.asyncio
async def test_authorization_runs_before_handle_message_for_denied_event(monkeypatch):
    adapter = ZaloAdapter(config({"allowed_group_ids": ["allowed-group"]}), client=FakeClient())
    called = False

    async def fake_handle(message):
        nonlocal called
        called = True

    adapter.handle_message = fake_handle
    await adapter._handle_sse_event({
        "type": "message",
        "data": {"msgId": "m-1", "threadId": "blocked-group", "senderId": "u-1", "content": "no", "isGroup": True},
    })
    assert called is False


@pytest.mark.asyncio
async def test_handle_sse_event_uses_single_inbound_gate_for_authorized_events():
    adapter = ZaloAdapter(config({"allowed_group_ids": ["g-1"], "require_mention": False}), client=FakeClient())
    handled = []
    auth_calls = 0
    original_auth = adapter._is_authorized

    def tracked_auth(event):
        nonlocal auth_calls
        auth_calls += 1
        return original_auth(event)

    async def fake_handle(message):
        handled.append(message)

    adapter._is_authorized = tracked_auth
    adapter.handle_message = fake_handle
    await adapter._handle_sse_event({
        "type": "message",
        "data": {"msgId": "m-gate-1", "threadId": "g-1", "senderId": "u-1", "content": "ping", "isGroup": True},
    })

    assert len(handled) == 1
    assert auth_calls == 1


@pytest.mark.asyncio
async def test_allowed_group_requires_trigger_by_default():
    adapter = ZaloAdapter(config({"allowed_group_ids": ["g-1"]}), client=FakeClient())
    adapter._self_user_id = "self-1"
    handled = []

    async def fake_handle(message):
        handled.append(message)

    adapter.handle_message = fake_handle
    await adapter._handle_sse_event({
        "type": "message",
        "data": {"msgId": "m-trigger-1", "threadId": "g-1", "senderId": "u-1", "content": "ping", "isGroup": True},
    })

    assert handled == []


@pytest.mark.asyncio
async def test_allowed_group_native_mention_of_bot_triggers_handle_message():
    adapter = ZaloAdapter(config({"allowed_group_ids": ["g-1"]}), client=FakeClient())
    adapter._self_user_id = "self-1"
    handled = []

    async def fake_handle(message):
        handled.append(message)

    adapter.handle_message = fake_handle
    await adapter._handle_sse_event({
        "type": "message",
        "data": {
            "msgId": "m-trigger-2",
            "threadId": "g-1",
            "senderId": "u-1",
            "content": "@Bot hello",
            "isGroup": True,
            "mentions": [{"uid": "self-1", "pos": 0, "len": 4}],
        },
    })

    assert len(handled) == 1
    assert handled[0].text == "hello"


@pytest.mark.asyncio
async def test_canonical_v2_mention_of_bot_triggers_handle_message():
    adapter = ZaloAdapter(config({"allowed_group_ids": ["g-1"]}), client=FakeClient())
    adapter._self_user_id = "self-1"
    handled = []

    async def fake_handle(message):
        handled.append(message)

    adapter.handle_message = fake_handle
    await adapter._handle_sse_event({
        "type": "message",
        "data": {
            "schemaVersion": 2,
            "messageId": "canonical-trigger-1",
            "threadId": "g-1",
            "senderId": "u-1",
            "text": "@Bot hello from canonical v2",
            "isGroup": True,
            "messageType": "text",
            "mentions": [{"uid": "self-1", "text": "@Bot", "pos": 0, "len": 4}],
            "atAll": False,
            "quote": None,
        },
    })

    assert len(handled) == 1
    assert handled[0].text == "hello from canonical v2"
    assert handled[0].message_id == "canonical-trigger-1"


@pytest.mark.asyncio
async def test_allowed_group_mention_of_other_user_does_not_trigger():
    adapter = ZaloAdapter(config({"allowed_group_ids": ["g-1"]}), client=FakeClient())
    adapter._self_user_id = "self-1"
    handled = []

    async def fake_handle(message):
        handled.append(message)

    adapter.handle_message = fake_handle
    await adapter._handle_sse_event({
        "type": "message",
        "data": {
            "msgId": "m-trigger-3",
            "threadId": "g-1",
            "senderId": "u-1",
            "content": "@Other hello",
            "isGroup": True,
            "mentions": [{"uid": "other-1", "pos": 0, "len": 6}],
        },
    })

    assert handled == []


@pytest.mark.asyncio
async def test_canonical_v2_mention_of_other_user_does_not_trigger():
    adapter = ZaloAdapter(config({"allowed_group_ids": ["g-1"]}), client=FakeClient())
    adapter._self_user_id = "self-1"
    handled = []

    async def fake_handle(message):
        handled.append(message)

    adapter.handle_message = fake_handle
    await adapter._handle_sse_event({
        "type": "message",
        "data": {
            "schemaVersion": 2,
            "messageId": "canonical-trigger-other",
            "threadId": "g-1",
            "senderId": "u-1",
            "text": "@Other hello",
            "isGroup": True,
            "messageType": "text",
            "mentions": [{"uid": "other-1", "text": "@Other", "pos": 0, "len": 6}],
            "atAll": False,
            "quote": None,
        },
    })

    assert handled == []


@pytest.mark.asyncio
async def test_canonical_v2_at_all_alone_does_not_count_as_bot_mention():
    adapter = ZaloAdapter(config({"allowed_group_ids": ["g-1"]}), client=FakeClient())
    adapter._self_user_id = "self-1"
    handled = []

    async def fake_handle(message):
        handled.append(message)

    adapter.handle_message = fake_handle
    await adapter._handle_sse_event({
        "type": "message",
        "data": {
            "schemaVersion": 2,
            "messageId": "canonical-at-all",
            "threadId": "g-1",
            "senderId": "u-1",
            "text": "@All hello",
            "isGroup": True,
            "messageType": "text",
            "mentions": [],
            "atAll": True,
            "quote": None,
        },
    })

    assert handled == []


@pytest.mark.asyncio
async def test_canonical_v2_quote_sender_id_self_triggers_handle_message():
    adapter = ZaloAdapter(config({"allowed_group_ids": ["g-1"]}), client=FakeClient())
    adapter._self_user_id = "self-1"
    handled = []

    async def fake_handle(message):
        handled.append(message)

    adapter.handle_message = fake_handle
    await adapter._handle_sse_event({
        "type": "message",
        "data": {
            "schemaVersion": 2,
            "messageId": "canonical-reply-1",
            "threadId": "g-1",
            "senderId": "u-1",
            "text": "reply to bot",
            "isGroup": True,
            "messageType": "text",
            "mentions": [],
            "atAll": False,
            "quote": {"messageId": "bot-message-1", "senderId": "self-1"},
        },
    })

    assert len(handled) == 1
    assert handled[0].text == "reply to bot"
    assert handled[0].reply_to_message_id == "bot-message-1"


@pytest.mark.asyncio
async def test_legacy_nested_mention_still_triggers_via_fallback():
    adapter = ZaloAdapter(config({"allowed_group_ids": ["g-1"]}), client=FakeClient())
    adapter._self_user_id = "self-1"
    handled = []

    async def fake_handle(message):
        handled.append(message)

    adapter.handle_message = fake_handle
    await adapter._handle_sse_event({
        "type": "message",
        "data": {
            "msgId": "legacy-nested-mention",
            "threadId": "g-1",
            "senderId": "u-1",
            "content": "@Bot legacy hello",
            "isGroup": True,
            "raw": {"mentionList": [{"uid": "self-1", "text": "@Bot", "pos": 0, "len": 4}]},
        },
    })

    assert len(handled) == 1
    assert handled[0].text == "legacy hello"


@pytest.mark.asyncio
async def test_allowed_group_prefix_triggers_and_strips_prefix():
    adapter = ZaloAdapter(config({"allowed_group_ids": ["g-1"], "prefixes": ["/ngan", "/hermes"]}), client=FakeClient())
    handled = []

    async def fake_handle(message):
        handled.append(message)

    adapter.handle_message = fake_handle
    await adapter._handle_sse_event({
        "type": "message",
        "data": {
            "msgId": "m-trigger-4",
            "threadId": "g-1",
            "senderId": "u-1",
            "content": "/ngan summarize this",
            "isGroup": True,
        },
    })

    assert len(handled) == 1
    assert handled[0].text == "summarize this"


@pytest.mark.asyncio
async def test_free_response_group_triggers_without_mention():
    adapter = ZaloAdapter(
        config({"allowed_group_ids": ["g-free"], "free_response_group_ids": ["g-free"]}),
        client=FakeClient(),
    )
    handled = []

    async def fake_handle(message):
        handled.append(message)

    adapter.handle_message = fake_handle
    await adapter._handle_sse_event({
        "type": "message",
        "data": {
            "msgId": "m-trigger-5",
            "threadId": "g-free",
            "senderId": "u-1",
            "content": "no explicit trigger",
            "isGroup": True,
        },
    })

    assert len(handled) == 1
    assert handled[0].text == "no explicit trigger"


@pytest.mark.asyncio
async def test_allowed_dm_user_calls_handle_message():
    adapter = ZaloAdapter(config({"allowed_user_ids": ["u-1"]}), client=FakeClient())
    handled = []

    async def fake_handle(message):
        handled.append(message)

    adapter.handle_message = fake_handle
    await adapter._handle_sse_event({
        "type": "message",
        "data": {"msgId": "dm-1", "threadId": "u-1", "senderId": "u-1", "content": "hello", "isGroup": False},
    })

    assert len(handled) == 1
    assert handled[0].text == "hello"


@pytest.mark.asyncio
async def test_unauthorized_dm_user_is_ignored():
    adapter = ZaloAdapter(config({"allowed_user_ids": ["u-1"]}), client=FakeClient())
    handled = []

    async def fake_handle(message):
        handled.append(message)

    adapter.handle_message = fake_handle
    await adapter._handle_sse_event({
        "type": "message",
        "data": {"msgId": "dm-2", "threadId": "u-2", "senderId": "u-2", "content": "hello", "isGroup": False},
    })

    assert handled == []


@pytest.mark.asyncio
async def test_self_loop_and_duplicate_events_are_dropped_before_handle_message():
    adapter = ZaloAdapter(config({"allowed_group_ids": ["g-1"], "require_mention": False}), client=FakeClient())
    adapter._self_user_id = "self-1"
    handled = []

    async def fake_handle(message):
        handled.append(message)

    adapter.handle_message = fake_handle
    await adapter._handle_sse_event({
        "type": "message",
        "data": {"msgId": "self-msg", "threadId": "g-1", "senderId": "self-1", "content": "echo", "isGroup": True},
    })
    event = {
        "type": "message",
        "data": {"msgId": "m-1", "threadId": "g-1", "senderId": "u-1", "content": "ping", "isGroup": True},
    }
    await adapter._handle_sse_event(event)
    await adapter._handle_sse_event(event)
    assert len(handled) == 1
    assert handled[0].text == "ping"
    assert handled[0].raw_message["threadId"] == "g-1"
    assert "data" not in handled[0].raw_message
    assert "senderName" not in handled[0].raw_message
    assert "mentions" not in handled[0].raw_message
    assert "atAll" not in handled[0].raw_message
    assert "timestamp" not in handled[0].raw_message


@pytest.mark.asyncio
async def test_send_requires_group_context_and_uses_contract_endpoint_payload():
    fake = FakeClient()
    adapter = ZaloAdapter(config({"allowed_group_ids": ["g-1"]}), client=fake)
    missing = await adapter.send("g-1", "hello")
    assert missing.success is False
    ok = await adapter.send("g-1", "hello", metadata={"isGroup": True})
    assert ok.success is True
    assert fake.sent == [{"threadId": "g-1", "message": "hello", "isGroup": True}]


@pytest.mark.asyncio
async def test_send_chunks_long_unicode_text_without_breaking_context():
    fake = FakeClient()
    adapter = ZaloAdapter(config({"allowed_group_ids": ["g-1"], "max_message_length": 5}), client=fake)
    ok = await adapter.send("g-1", "xin chào anh", metadata={"isGroup": True})
    assert ok.success is True
    assert [item["message"] for item in fake.sent] == ["xin c", "hào a", "nh"]


@pytest.mark.asyncio
async def test_hzca_client_parses_sse_fixture(monkeypatch):
    lines = Path("plugins/platforms/zalo/fixtures/sse-message.frame").read_bytes().splitlines(keepends=True)

    async def immediate_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    class FakeResponse:
        def __init__(self):
            self._lines = list(lines)
        def readline(self):
            return self._lines.pop(0) if self._lines else b""
        def close(self):
            pass

    monkeypatch.setattr("plugins.platforms.zalo.adapter.urlopen", lambda req, timeout: FakeResponse())
    monkeypatch.setattr("plugins.platforms.zalo.adapter.asyncio.to_thread", immediate_to_thread)
    client = HzcaClient(ZaloBackendConfig(base_url="http://127.0.0.1:1", request_timeout=0.1))
    events = []
    async for event in client.iter_events(asyncio.Event()):
        events.append(event)
    assert events[0]["type"] == "message"
    assert events[0]["data"]["msgId"] == "m-1"


@pytest.mark.asyncio
async def test_hzca_client_sse_uses_sse_timeout_not_rest_request_timeout(monkeypatch):
    lines = Path("plugins/platforms/zalo/fixtures/sse-message.frame").read_bytes().splitlines(keepends=True)
    seen_timeouts = []

    async def immediate_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    class FakeResponse:
        def __init__(self):
            self._lines = list(lines)

        def readline(self):
            return self._lines.pop(0) if self._lines else b""

        def close(self):
            pass

    def fake_urlopen(req, timeout):
        seen_timeouts.append(timeout)
        return FakeResponse()

    monkeypatch.setattr("plugins.platforms.zalo.adapter.urlopen", fake_urlopen)
    monkeypatch.setattr("plugins.platforms.zalo.adapter.asyncio.to_thread", immediate_to_thread)
    client = HzcaClient(ZaloBackendConfig(
        base_url="http://127.0.0.1:1",
        request_timeout=0.1,
        sse_timeout=3600.0,
    ))

    async for _event in client.iter_events(asyncio.Event()):
        break

    assert seen_timeouts == [3600.0]
