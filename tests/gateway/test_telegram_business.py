"""Telegram Business delegated-inbox routing regressions."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    MessageEvent,
    MessageType,
    SendResult,
    _thread_metadata_for_source,
)
from gateway.session import SessionSource, build_session_key
from plugins.platforms.telegram import adapter as telegram


def _adapter(business=None):
    async def _get_business_connection(connection_id):
        return SimpleNamespace(
            id=connection_id,
            user=SimpleNamespace(id="business-owner"),
            is_enabled=True,
        )

    business_config = dict(business or {})
    if business_config.get("enabled") and not (
        "allowed_owner_ids" in business_config
        or "allowed_connection_ids" in business_config
        or "allow_from" in business_config
    ):
        business_config["allowed_owner_ids"] = ["business-owner"]
    config = PlatformConfig(
        enabled=True,
        token="fake",
        extra={"business": business_config},
    )
    instance = object.__new__(telegram.TelegramAdapter)
    instance.config = config
    instance._config = config
    instance._platform = Platform.TELEGRAM
    instance.platform = Platform.TELEGRAM
    instance._bot = SimpleNamespace(
        id=999,
        username="hermes_bot",
        get_business_connection=_get_business_connection,
    )
    instance._reply_to_mode = "first"
    instance._rich_messages_enabled = False
    instance._send_path_degraded = False
    instance._disable_link_previews = False
    instance._telegram_typing_cooldown_until = {}
    instance._telegram_typing_cooldown_seconds = 30.0
    instance._notifications_mode = "all"
    return instance


def _message(
    *,
    text="Sigurd, hello",
    caption=None,
    user_id=123,
    chat_id=456,
    business_connection_id="bc-123",
    is_bot=False,
    sender_business_bot=None,
    reply_to_message=None,
):
    return SimpleNamespace(
        text=text,
        caption=caption,
        chat=SimpleNamespace(id=chat_id, type="private", title=None, full_name="Customer"),
        from_user=SimpleNamespace(id=user_id, full_name="Customer", is_bot=is_bot),
        message_thread_id=None,
        reply_to_message=reply_to_message,
        message_id=42,
        date=None,
        business_connection_id=business_connection_id,
        sender_business_bot=sender_business_bot,
        photo=None,
        video=None,
        audio=None,
        voice=None,
        document=None,
        sticker=None,
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            {},
            {
                "enabled": False,
                "allow_business_send_as_account": False,
                "allowed_chats": [],
                "allowed_owner_ids": [],
                "allowed_connection_ids": [],
                "trigger_words": [],
            },
        ),
        (
            {"enabled": "YES", "allowed_chats": 42, "trigger_words": "Sigurd"},
            {
                "enabled": True,
                "allow_business_send_as_account": False,
                "allowed_chats": ["42"],
                "allowed_owner_ids": [],
                "allowed_connection_ids": [],
                "trigger_words": ["Sigurd"],
            },
        ),
        (
            {"enabled": "off", "allowed_chats": ["", " 7 ", None], "trigger_words": [" Argus ", ""]},
            {
                "enabled": False,
                "allow_business_send_as_account": False,
                "allowed_chats": ["7"],
                "allowed_owner_ids": [],
                "allowed_connection_ids": [],
                "trigger_words": ["Argus"],
            },
        ),
    ],
)
def test_apply_yaml_config_normalizes_business(raw, expected):
    extras = telegram._apply_yaml_config({}, {"business": raw})
    assert extras["business"] == expected


def test_business_config_normalizes_explicit_authority():
    extras = telegram._apply_yaml_config(
        {},
        {
            "business": {
                "allowed_owner_ids": [" owner ", 7],
                "allowed_connection_ids": " connection ",
            }
        },
    )
    assert extras is not None
    assert extras["business"]["allowed_owner_ids"] == ["owner", "7"]
    assert extras["business"]["allowed_connection_ids"] == ["connection"]


def test_business_session_key_isolated_from_ordinary_telegram_dm():
    ordinary = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="456",
        user_id="123",
    )
    business = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="456",
        user_id="123",
        scope_id="telegram-business:bc-123",
    )
    assert build_session_key(ordinary) == "agent:main:telegram:dm:456"
    assert build_session_key(business) == (
        "agent:main:telegram:dm:telegram-business:bc-123:456"
    )
    assert build_session_key(ordinary) != build_session_key(business)


def test_business_connections_get_distinct_session_keys():
    first = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="456",
        user_id="123",
        scope_id="telegram-business:bc-one",
    )
    second = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="456",
        user_id="123",
        scope_id="telegram-business:bc-two",
    )
    assert build_session_key(first) != build_session_key(second)


def test_business_authority_proof_is_not_serialized():
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="456",
        user_id="customer",
        scope_id="telegram-business:bc-123",
        authorized_via_telegram_business=True,
        telegram_business_owner_id="business-owner",
    )
    serialized = source.to_dict()
    assert "authorized_via_telegram_business" not in serialized
    assert "telegram_business_owner_id" not in serialized
    restored = SessionSource.from_dict(serialized)
    assert restored.authorized_via_telegram_business is False
    assert restored.telegram_business_owner_id is None


def test_business_delivery_metadata_requires_trusted_matching_opt_in():
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="456",
        scope_id="telegram-business:bc-123",
    )
    assert _thread_metadata_for_source(source) is None
    assert _thread_metadata_for_source(
        source,
        event_metadata={"business_connection_id": "bc-123"},
    ) is None
    assert _thread_metadata_for_source(
        source,
        event_metadata={
            "allow_business_send_as_account": True,
            "business_connection_id": "different",
        },
    ) is None
    assert _thread_metadata_for_source(
        source,
        event_metadata={
            "allow_business_send_as_account": True,
            "business_connection_id": "bc-123",
        },
    ) == {
        "allow_business_send_as_account": True,
        "business_connection_id": "bc-123",
        "telegram_dm_topic_reply_fallback": True,
    }


def test_business_kwargs_fail_closed_without_exact_opt_in():
    assert telegram.TelegramAdapter._business_kwargs(None) == {}
    assert telegram.TelegramAdapter._business_kwargs(
        {"business_connection_id": "bc-123"}
    ) == {}
    assert telegram.TelegramAdapter._business_kwargs(
        {
            "allow_business_send_as_account": "true",
            "business_connection_id": "bc-123",
        }
    ) == {}
    assert telegram.TelegramAdapter._business_kwargs(
        {
            "allow_business_send_as_account": True,
            "business_connection_id": "bc-123",
        }
    ) == {"business_connection_id": "bc-123"}


@pytest.mark.asyncio
async def test_business_inbound_stamps_trusted_route_and_external_session_scope():
    accepted = []
    adapter = _adapter(
        {
            "enabled": True,
            "allow_business_send_as_account": True,
            "trigger_words": ["Sigurd"],
        }
    )
    adapter._build_message_event = lambda message, msg_type, update_id=None: MessageEvent(
        text=message.text or "",
        message_type=msg_type,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=str(message.chat.id),
            user_id=str(message.from_user.id),
            message_id=str(message.message_id),
        ),
        message_id=str(message.message_id),
        platform_update_id=update_id,
    )
    adapter._enqueue_text_event = accepted.append

    await adapter._handle_business_message(
        SimpleNamespace(update_id=7, business_message=_message()),
        SimpleNamespace(),
    )

    event = accepted[0]
    assert event.text == "hello"
    assert event.source.scope_id == "telegram-business:bc-123"
    assert event.source.telegram_business_owner_id == "business-owner"
    assert event.metadata == {
        "allow_business_send_as_account": True,
        "business_connection_id": "bc-123",
    }
    assert event.allow_gateway_control is False


@pytest.mark.asyncio
async def test_business_inbound_rejects_owner_and_unverifiable_connection():
    async def _owner(connection_id):
        return SimpleNamespace(
            id=connection_id,
            user=SimpleNamespace(id=123),
            is_enabled=True,
        )

    async def _unavailable(_connection_id):
        raise RuntimeError("lookup failed")

    for get_connection in (_owner, _unavailable):
        accepted = []
        adapter = _adapter(
            {"enabled": True, "trigger_words": ["Sigurd"]}
        )
        adapter._bot.get_business_connection = get_connection
        adapter._build_message_event = lambda message, msg_type, update_id=None: MessageEvent(
            text=message.text or "",
            message_type=msg_type,
            source=SessionSource(
                platform=Platform.TELEGRAM,
                chat_id=str(message.chat.id),
                user_id=str(message.from_user.id),
                message_id=str(message.message_id),
            ),
            message_id=str(message.message_id),
            platform_update_id=update_id,
        )
        adapter._enqueue_text_event = accepted.append

        await adapter._handle_business_message(
            SimpleNamespace(update_id=7, business_message=_message()),
            SimpleNamespace(),
        )

        assert accepted == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "authority",
    [
        {"allowed_owner_ids": [], "allowed_connection_ids": []},
        {"allowed_owner_ids": ["different-owner"]},
        {"allowed_connection_ids": ["different-connection"]},
    ],
)
async def test_business_inbound_requires_explicit_matching_authority(authority):
    accepted = []
    adapter = _adapter(
        {"enabled": True, "trigger_words": ["Sigurd"], **authority}
    )
    adapter._enqueue_text_event = lambda event: accepted.append(event)

    await adapter._handle_business_message(
        SimpleNamespace(update_id=7, business_message=_message()),
        SimpleNamespace(),
    )

    assert accepted == []


@pytest.mark.asyncio
async def test_business_update_is_ignored_by_ordinary_text_handler():
    message = _message()
    adapter = _adapter({"enabled": True, "trigger_words": ["Sigurd"]})
    accepted = []
    adapter._enqueue_text_event = accepted.append

    await adapter._handle_text_message(
        SimpleNamespace(
            update_id=7,
            business_message=message,
            effective_message=message,
            message=None,
        ),
        SimpleNamespace(),
    )

    assert accepted == []


@pytest.mark.asyncio
async def test_business_inbound_is_disabled_and_trigger_gated():
    accepted = []
    for config, message in [
        ({"enabled": False, "trigger_words": ["Sigurd"]}, _message()),
        ({"enabled": True, "trigger_words": ["Sigurd"]}, _message(text="ordinary")),
        (
            {"enabled": True, "allowed_chats": ["999"], "trigger_words": ["Sigurd"]},
            _message(),
        ),
        ({"enabled": True, "trigger_words": ["Sigurd"]}, _message(is_bot=True)),
        (
            {"enabled": True, "trigger_words": ["Sigurd"]},
            _message(sender_business_bot=SimpleNamespace(id=999)),
        ),
    ]:
        adapter = _adapter(config)
        adapter._enqueue_text_event = accepted.append
        await adapter._handle_business_message(
            SimpleNamespace(update_id=7, business_message=message),
            SimpleNamespace(),
        )
    assert accepted == []


@pytest.mark.asyncio
async def test_text_send_uses_business_account_only_with_explicit_opt_in():
    calls = []

    async def send_message(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(message_id=1)

    adapter = _adapter()
    adapter._bot = SimpleNamespace(id=999, username="hermes_bot", send_message=send_message)
    assert (await adapter.send(
        "456", "default bot", metadata={"business_connection_id": "bc-123", "notify": True}
    )).success
    assert (await adapter.send(
        "456",
        "delegated",
        metadata={
            "allow_business_send_as_account": True,
            "business_connection_id": "bc-123",
            "notify": True,
        },
    )).success

    assert "business_connection_id" not in calls[0]
    assert calls[1]["business_connection_id"] == "bc-123"


@pytest.mark.asyncio
async def test_business_identity_survives_stale_reply_fallback():
    from telegram.error import BadRequest

    calls = []

    async def send_message(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise BadRequest("Message to be replied not found")
        return SimpleNamespace(message_id=2)

    adapter = _adapter()
    adapter._bot = SimpleNamespace(
        id=999,
        username="hermes_bot",
        send_message=send_message,
    )
    result = await adapter.send(
        "456",
        "delegated",
        reply_to="42",
        metadata={
            "allow_business_send_as_account": True,
            "business_connection_id": "bc-123",
            "telegram_dm_topic_reply_fallback": True,
            "telegram_reply_to_message_id": "42",
            "notify": True,
        },
    )

    assert result.success
    assert calls[0]["business_connection_id"] == "bc-123"
    assert calls[0]["reply_to_message_id"] == 42
    assert calls[1]["business_connection_id"] == "bc-123"
    assert calls[1]["reply_to_message_id"] is None


@pytest.mark.asyncio
async def test_business_identity_survives_oversized_edit_reply_fallback():
    from telegram.error import BadRequest

    send_calls = []
    edit_calls = []

    async def edit_message_text(**kwargs):
        edit_calls.append(kwargs)
        return True

    async def send_message(**kwargs):
        send_calls.append(kwargs)
        if len(send_calls) == 1:
            raise BadRequest("Reply message not found")
        return SimpleNamespace(message_id=3)

    adapter = _adapter()
    adapter._bot = SimpleNamespace(
        id=999,
        username="hermes_bot",
        edit_message_text=edit_message_text,
        send_message=send_message,
    )
    result = await adapter._edit_overflow_split(
        "456",
        "1",
        "x" * (adapter.MAX_MESSAGE_LENGTH + 100),
        finalize=True,
        metadata={
            "allow_business_send_as_account": True,
            "business_connection_id": "bc-123",
            "telegram_dm_topic_reply_fallback": True,
        },
    )

    assert result.success
    assert edit_calls[0]["business_connection_id"] == "bc-123"
    assert send_calls[0]["business_connection_id"] == "bc-123"
    assert send_calls[1]["business_connection_id"] == "bc-123"


@pytest.mark.asyncio
async def test_business_rich_edit_keeps_identity_and_draft_streaming_is_disabled():
    rich_calls = []

    async def do_api_request(method, *, api_kwargs):
        rich_calls.append((method, api_kwargs))
        return True

    adapter = _adapter()
    adapter._bot = SimpleNamespace(
        id=999,
        username="hermes_bot",
        do_api_request=do_api_request,
        send_message_draft=lambda **_kwargs: None,
    )
    metadata = {
        "allow_business_send_as_account": True,
        "business_connection_id": "bc-123",
    }

    assert adapter.supports_draft_streaming("dm", metadata) is False
    result = await adapter._try_edit_rich("456", "42", "delegated", metadata)

    assert result.success
    assert rich_calls[0][0] == "editMessageText"
    assert rich_calls[0][1]["business_connection_id"] == "bc-123"


@pytest.mark.asyncio
async def test_business_typing_indicator_keeps_identity():
    calls = []

    async def send_chat_action(**kwargs):
        calls.append(kwargs)

    adapter = _adapter()
    adapter._bot = SimpleNamespace(
        id=999,
        username="hermes_bot",
        send_chat_action=send_chat_action,
    )
    await adapter.send_typing(
        "456",
        metadata={
            "allow_business_send_as_account": True,
            "business_connection_id": "bc-123",
        },
    )

    assert calls[0]["business_connection_id"] == "bc-123"


@pytest.mark.asyncio
async def test_business_send_as_account_false_is_preserved_from_config():
    accepted = []
    adapter = _adapter(
        {
            "enabled": True,
            "allow_business_send_as_account": False,
            "trigger_words": ["Sigurd"],
        }
    )
    adapter._build_message_event = lambda message, msg_type, update_id=None: MessageEvent(
        text=message.text or "",
        message_type=msg_type,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=str(message.chat.id),
            user_id=str(message.from_user.id),
            message_id=str(message.message_id),
        ),
        message_id=str(message.message_id),
        platform_update_id=update_id,
    )
    adapter._enqueue_text_event = accepted.append

    await adapter._handle_business_message(
        SimpleNamespace(update_id=8, business_message=_message()),
        SimpleNamespace(),
    )

    assert accepted[0].metadata["allow_business_send_as_account"] is False
    assert telegram.TelegramAdapter._business_kwargs(accepted[0].metadata) == {}


@pytest.mark.asyncio
async def test_native_media_send_honors_same_business_opt_in(tmp_path):
    calls = []

    async def send_photo(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(message_id=2)

    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"not-decoded-by-test")
    adapter = _adapter()
    adapter._bot = SimpleNamespace(id=999, username="hermes_bot", send_photo=send_photo)

    result = await adapter.send_image_file(
        "456",
        str(image_path),
        metadata={
            "allow_business_send_as_account": True,
            "business_connection_id": "bc-123",
        },
    )

    assert result.success
    assert calls[0]["business_connection_id"] == "bc-123"


def test_business_authorization_is_in_process_bound_and_fail_closed(monkeypatch):
    from gateway.run import GatewayRunner

    for key in (
        "TELEGRAM_ALLOWED_USERS",
        "TELEGRAM_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
    ):
        monkeypatch.delenv(key, raising=False)

    adapter = _adapter(
        business={
            "enabled": True,
            "allowed_chats": ["42"],
            "trigger_words": ["Sigurd"],
        }
    )
    runner = object.__new__(GatewayRunner)
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_kw: False)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.profile_adapters = {}

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="42",
        chat_type="dm",
        user_id="external-customer",
        scope_id="telegram-business:bc-123",
        authorized_via_telegram_business=True,
        telegram_business_owner_id="business-owner",
    )

    assert runner._is_user_authorized(source) is True
    assert runner._is_user_authorized(source, allow_adapter_delegation=False) is False

    monkeypatch.setenv("TELEGRAM_ALLOW_ALL_USERS", "true")
    source.telegram_business_owner_id = "different-owner"
    assert runner._is_user_authorized(source) is False
    source.telegram_business_owner_id = "business-owner"
    source.authorized_via_telegram_business = False
    assert runner._is_user_authorized(source) is False
    source.authorized_via_telegram_business = True
    adapter.config.extra["business"]["enabled"] = False
    assert runner._is_user_authorized(source) is False

    ordinary = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="42",
        chat_type="dm",
        user_id="ordinary-user",
    )
    assert runner._is_user_authorized(ordinary) is True

    source.scope_id = "telegram-business:bc-other"
    source.chat_id = "not-allowed"
    assert runner._is_user_authorized(source) is False


def test_gateway_runner_business_egress_metadata_is_event_bound():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="42",
        chat_type="dm",
        user_id="external-customer",
        scope_id="telegram-business:bc-123",
        thread_id="7",
        profile="coder",
    )
    trusted_event_metadata = {
        "allow_business_send_as_account": True,
        "business_connection_id": "bc-123",
    }

    metadata = runner._thread_metadata_for_source(
        source, "99", trusted_event_metadata
    )
    assert metadata["allow_business_send_as_account"] is True
    assert metadata["business_connection_id"] == "bc-123"
    assert metadata["telegram_reply_to_message_id"] == "99"
    assert metadata["hermes_profile"] == "coder"

    progress_metadata, progress_reply_to, status_metadata = (
        runner._run_agent_progress_threading(
            source, "99", False, trusted_event_metadata
        )
    )
    assert progress_reply_to is None
    assert progress_metadata is not None
    assert status_metadata is not None
    for routed_metadata in (progress_metadata, status_metadata):
        assert routed_metadata["allow_business_send_as_account"] is True
        assert routed_metadata["business_connection_id"] == "bc-123"

    threadless_source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="42",
        chat_type="dm",
        user_id="external-customer",
        scope_id="telegram-business:bc-123",
        profile="coder",
    )
    threadless_metadata = runner._thread_metadata_for_progress(
        threadless_source,
        None,
        "99",
        trusted_event_metadata,
    )
    assert threadless_metadata == {
        "allow_business_send_as_account": True,
        "business_connection_id": "bc-123",
        "hermes_profile": "coder",
    }

    assert "business_connection_id" not in runner._thread_metadata_for_source(
        source, "99"
    )
    assert "business_connection_id" not in runner._thread_metadata_for_source(
        source,
        "99",
        {
            "allow_business_send_as_account": True,
            "business_connection_id": "bc-other",
        },
    )


@pytest.mark.asyncio
async def test_business_drain_ack_preserves_event_bound_send_authority():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    adapter = SimpleNamespace(_send_with_retry=AsyncMock())
    runner._draining = True
    runner._is_user_authorized = lambda _source: True
    runner._effective_busy_input_mode = lambda _source: "interrupt"
    runner._adapter_for_source = lambda _source: adapter
    runner._reply_anchor_for_event = lambda _event: "42"
    runner._queue_during_drain_enabled = lambda _mode: False
    runner._status_action_gerund = lambda: "restarting"
    event = MessageEvent(
        text="Sigurd, follow up",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="42",
            chat_type="dm",
            user_id="external-customer",
            scope_id="telegram-business:bc-123",
        ),
        message_id="42",
        metadata={
            "allow_business_send_as_account": True,
            "business_connection_id": "bc-123",
        },
        allow_gateway_control=False,
    )

    assert await runner._handle_active_session_busy_message(event, "session") is True

    metadata = adapter._send_with_retry.await_args.kwargs["metadata"]
    assert metadata["allow_business_send_as_account"] is True
    assert metadata["business_connection_id"] == "bc-123"


@pytest.mark.asyncio
async def test_business_busy_ack_preserves_event_bound_send_authority():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    adapter = SimpleNamespace(_send_with_retry=AsyncMock())
    running_agent = MagicMock()
    running_agent.steer.return_value = True
    running_agent.get_activity_summary.return_value = {}
    turn = SimpleNamespace(agent=running_agent, busy_ack_ts=0, started_ts=0)
    state = SimpleNamespace(turn=turn)
    runner._draining = False
    runner._is_user_authorized = lambda _source: True
    runner._effective_busy_input_mode = lambda _source: "steer"
    runner._effective_busy_text_mode = lambda _source: "interrupt"
    runner._adapter_for_source = lambda _source: adapter
    runner._peek_session_state = lambda _session_key: state
    runner._session_state = lambda _session_key: state
    runner._prepare_busy_steer_text = AsyncMock(return_value="follow up")
    runner._reply_anchor_for_event = lambda _event: "42"
    event = MessageEvent(
        text="follow up",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="42",
            chat_type="dm",
            user_id="external-customer",
            scope_id="telegram-business:bc-123",
        ),
        message_id="42",
        metadata={
            "allow_business_send_as_account": True,
            "business_connection_id": "bc-123",
        },
        allow_gateway_control=False,
    )

    with patch("agent.onboarding.is_seen", return_value=True), patch(
        "gateway.run._load_gateway_config", return_value={}
    ), patch(
        "gateway.display_config.resolve_display_setting", return_value=True
    ):
        assert await runner._handle_active_session_busy_message(event, "session") is True

    metadata = adapter._send_with_retry.await_args.kwargs["metadata"]
    assert metadata["allow_business_send_as_account"] is True
    assert metadata["business_connection_id"] == "bc-123"


def test_business_authorization_proof_is_not_serialized():
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="42",
        chat_type="dm",
        user_id="external-customer",
        scope_id="telegram-business:bc-123",
        authorized_via_telegram_business=True,
    )

    payload = source.to_dict()

    assert "authorized_via_telegram_business" not in payload
    assert SessionSource.from_dict(payload).authorized_via_telegram_business is False


def test_restricted_business_event_cannot_dispatch_gateway_command():
    from gateway.run import GatewayRunner

    event = MessageEvent(
        text="/stop",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="42",
            chat_type="dm",
            user_id="external-customer",
        ),
        allow_gateway_control=False,
    )

    assert GatewayRunner._gateway_command_for_event(event) is None
    event.allow_gateway_control = True
    assert GatewayRunner._gateway_command_for_event(event) == "stop"


@pytest.mark.asyncio
async def test_business_stream_reconciliation_preserves_identity_and_failed_delivery():
    from gateway.run import GatewayRunner
    from gateway.stream_consumer import GatewayStreamConsumer

    metadata = {
        "allow_business_send_as_account": True,
        "business_connection_id": "bc-123",
    }
    adapter = SimpleNamespace(
        edit_message=AsyncMock(return_value=SendResult(success=False, error="rejected"))
    )
    consumer = GatewayStreamConsumer(
        adapter=adapter,
        chat_id="customer-chat",
        metadata=metadata,
    )
    consumer._message_id = "42"
    response = {}
    runner = object.__new__(GatewayRunner)

    await runner._run_agent_edit_streamed_message(
        consumer,
        SimpleNamespace(chat_id="customer-chat"),
        response,
        "final answer",
        _sk="session",
        ok=("edited",),
        fail_result=None,
        fail_exc="edit failed for %s: %s",
    )

    assert adapter.edit_message.await_args.kwargs["metadata"] == metadata
    assert "already_sent" not in response


@pytest.mark.asyncio
async def test_stream_reconciliation_keeps_legacy_adapter_edit_signature():
    from gateway.run import GatewayRunner
    from gateway.stream_consumer import GatewayStreamConsumer

    class LegacyAdapter:
        def __init__(self):
            self.calls = []

        async def edit_message(self, chat_id, message_id, content, finalize=False):
            self.calls.append((chat_id, message_id, content, finalize))
            return SendResult(success=True)

    adapter = LegacyAdapter()
    consumer = GatewayStreamConsumer(
        adapter=adapter,
        chat_id="legacy-chat",
        metadata={"thread_id": "topic"},
    )
    consumer._message_id = "42"
    response = {}
    runner = object.__new__(GatewayRunner)

    await runner._run_agent_edit_streamed_message(
        consumer,
        SimpleNamespace(chat_id="legacy-chat"),
        response,
        "final answer",
        _sk="session",
        ok=("edited",),
        fail_result=None,
        fail_exc="edit failed for %s: %s",
    )

    assert adapter.calls == [("legacy-chat", "42", "final answer", True)]
    assert response["already_sent"] is True
