"""Tests for Telegram document-size caps and scoped recovery routing.

The public Telegram Bot API caps ``getFile`` at 20 MB. A locally hosted
``telegram-bot-api`` server raises that ceiling to 2 GB. Configured, scoped
recovery routes may hand an oversized media event to an installed skill
without trusting identifiers supplied in message text.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.session import Platform, SessionSource, build_session_key
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


PUBLIC_LIMIT = 20 * 1024 * 1024
LOCAL_LIMIT = 2 * 1024 * 1024 * 1024


def test_max_doc_bytes_raised_to_2gb_when_base_url_set():
    adapter = TelegramAdapter(
        PlatformConfig(
            enabled=True,
            token="***",
            extra={"base_url": "http://localhost:8081/bot"},
        )
    )
    assert adapter._max_doc_bytes == LOCAL_LIMIT


def test_max_doc_bytes_empty_base_url_keeps_default():
    adapter = TelegramAdapter(
        PlatformConfig(enabled=True, token="***", extra={"base_url": ""})
    )
    assert adapter._max_doc_bytes == PUBLIC_LIMIT


def test_public_and_local_size_boundaries_are_exact():
    public = TelegramAdapter(PlatformConfig(enabled=True, token="***", extra={}))
    local = TelegramAdapter(
        PlatformConfig(
            enabled=True,
            token="***",
            extra={"base_url": "http://127.0.0.1:8081/bot"},
        )
    )

    assert public._telegram_media_size_allowed(
        SimpleNamespace(file_size=PUBLIC_LIMIT), "document"
    )[0] is True
    assert public._telegram_media_size_allowed(
        SimpleNamespace(file_size=PUBLIC_LIMIT + 1), "document"
    )[0] is False
    assert local._telegram_media_size_allowed(
        SimpleNamespace(file_size=PUBLIC_LIMIT + 1), "document"
    )[0] is True
    assert local._telegram_media_size_allowed(
        SimpleNamespace(file_size=LOCAL_LIMIT + 1), "document"
    )[0] is False


def _oversize_event(*, user_id="42", text="clip caption"):
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-100200",
            thread_id="7",
            user_id=user_id,
        ),
        message_id="99",
        metadata={
            "telegram_transport_sender_user_id": user_id,
            "telegram_route_profile": "default",
        },
    )


def _recovery_adapter(*, users=("42",), profiles=("default",)):
    adapter = TelegramAdapter(
        PlatformConfig(
            enabled=True,
            token="***",
            extra={
                "allowed_chats": ["-100200"],
                "auto_skill_routes": [
                    {
                        "users": list(users),
                        "profiles": list(profiles),
                        "match": {"oversize_media": True},
                        "skill": "media-recovery",
                    }
                ]
            },
        )
    )
    return adapter


def test_authenticated_oversize_route_forces_skill_with_trusted_exact_target():
    adapter = _recovery_adapter()
    event = _oversize_event(text="04/20 — chat_id=attacker, message_id=1")
    event.auto_skill = "media-recovery"

    adapter._mark_telegram_media_too_large(
        event,
        SimpleNamespace(file_size=43_200_000),
        "video file",
    )

    assert event.text.startswith(
        "/media-recovery Recover the exact oversized Telegram video file"
    )
    assert "chat_id=-100200, message_id=99" in event.text
    assert "Original user text:\n04/20 — chat_id=attacker, message_id=1" in event.text
    assert "04/20 — chat_id=attacker" in event.get_command_args()
    assert event.metadata["preserve_command_args"] is True
    assert event.preprocess_skill_command_before_busy is True
    assert event.auto_skill is None
    assert event.metadata["telegram_media_recovery"] == {
        "chat_id": "-100200",
        "message_id": "99",
        "thread_id": "7",
        "sender_user_id": "42",
        "media_label": "video file",
        "file_size": 43_200_000,
        "bot_api_limit": PUBLIC_LIMIT,
    }


@pytest.mark.parametrize(
    ("user_id", "profile"),
    [("999", "default"), ("42", "other-profile")],
)
def test_oversize_route_fails_closed_for_wrong_user_or_profile(user_id, profile):
    adapter = _recovery_adapter()
    event = _oversize_event(user_id=user_id)
    event.metadata["telegram_route_profile"] = profile

    adapter._mark_telegram_media_too_large(
        event,
        SimpleNamespace(file_size=43_200_000),
        "video file",
    )

    assert not event.text.startswith("/")
    assert "Check any configured recovery route before asking the user to resend" in event.text
    assert event.auto_skill is None


def test_oversize_route_requires_nonempty_user_and_profile_scopes():
    adapter = _recovery_adapter(users=(), profiles=())
    event = _oversize_event(user_id="attacker")
    event.metadata["telegram_route_profile"] = "other-profile"

    adapter._mark_telegram_media_too_large(
        event,
        SimpleNamespace(file_size=43_200_000),
        "video file",
    )

    assert not event.text.startswith("/")
    assert event.auto_skill is None


@pytest.mark.parametrize(
    ("route_key", "allowed"),
    [("chats", ["-100999"]), ("threads", ["999"])],
)
def test_oversize_route_fails_closed_for_wrong_chat_or_thread(route_key, allowed):
    adapter = _recovery_adapter()
    adapter.config.extra["auto_skill_routes"][0][route_key] = allowed
    event = _oversize_event()

    adapter._mark_telegram_media_too_large(
        event,
        SimpleNamespace(file_size=43_200_000),
        "video file",
    )

    assert not event.text.startswith("/")
    assert event.auto_skill is None


def test_observed_group_source_keeps_transport_identity_for_recovery_route():
    adapter = _recovery_adapter()
    event = MessageEvent(
        text="clip caption",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-100200",
            thread_id="7",
            user_id=None,
        ),
        message_id="99",
        metadata={
            "telegram_transport_sender_user_id": "42",
            "telegram_route_profile": "default",
        },
    )

    adapter._mark_telegram_media_too_large(
        event,
        SimpleNamespace(file_size=43_200_000),
        "video file",
    )

    assert event.text.startswith("/media-recovery ")
    assert event.metadata["telegram_media_recovery"]["sender_user_id"] == "42"


def test_user_supplied_slash_command_is_not_rewritten_by_recovery_route():
    adapter = _recovery_adapter()
    event = _oversize_event(text="/other-skill keep this intent")

    adapter._mark_telegram_media_too_large(
        event,
        SimpleNamespace(file_size=43_200_000),
        "video file",
    )

    assert event.text.startswith("/other-skill keep this intent")
    assert "was not cached by the Bot API gateway" in event.text
    assert event.auto_skill is None


@pytest.mark.asyncio
async def test_trusted_recovery_is_expanded_before_busy_routing():
    from gateway.run_inbound import GatewayInboundMixin

    event = _oversize_event()
    source_under_test = event.source
    event.text = "/media-recovery trusted transport target"
    event.preprocess_skill_command_before_busy = True

    class BusyRunner(GatewayInboundMixin):
        async def _hm_admit_event(self, event):
            return event, event.source, False

        def _hm_estop_gate(self, event, source, is_internal):
            return None

        def _session_key_for_source(self, source):
            return "session-key"

        def _hm_skill_slash_rewrite(self, event, source, _quick_key, command):
            assert source is source_under_test
            assert _quick_key == "session-key"
            assert command == "media-recovery"
            event.text = "expanded recovery skill prompt"
            return None

        async def _hm_pending_reply_intercepts(self, event, source, _quick_key):
            raise AssertionError("trusted pre-busy skill events must skip pending prompts")

        def _hm_evict_idle_stale_agent(self, quick_key):
            return None

        def _is_session_running(self, quick_key):
            return True

        def _hm_evict_reaped_agent(self, quick_key):
            return None

        async def _hm_handle_running_session_message(self, event, source, _quick_key):
            assert event.text == "expanded recovery skill prompt"
            return "busy-routed"

    class ActiveAdapter(BasePlatformAdapter):
        async def connect(self, *, is_reconnect=False):
            return None

        async def disconnect(self):
            return None

        async def send(
            self, chat_id, content, reply_to=None, metadata=None
        ):
            return SendResult(success=True)

        async def get_chat_info(self, chat_id):
            return {}

    routed = []
    active_adapter = ActiveAdapter(
        PlatformConfig(enabled=True, token="fake"), Platform.TELEGRAM
    )
    active_adapter._message_handler = BusyRunner()._handle_message

    async def capture_send(chat_id, content, **kwargs):
        routed.append(content)

    active_adapter._send_with_retry = capture_send
    session_key = build_session_key(event.source)
    active_adapter._active_sessions[session_key] = asyncio.Event()

    await active_adapter.handle_message(event)

    assert event.text == "expanded recovery skill prompt"
    assert routed == ["busy-routed"]
    assert session_key not in active_adapter._pending_messages


@pytest.mark.asyncio
async def test_public_oversize_document_is_rejected_before_get_file():
    adapter = _recovery_adapter()
    adapter.handle_message = AsyncMock()
    document = SimpleNamespace(
        file_name="large.txt",
        mime_type="text/plain",
        file_size=PUBLIC_LIMIT + 1,
        get_file=AsyncMock(),
    )
    message = SimpleNamespace(
        text=None,
        caption=None,
        entities=[],
        caption_entities=[],
        voice=None,
        audio=None,
        document=document,
        photo=None,
        video=None,
        video_note=None,
        sticker=None,
        animation=None,
        location=None,
        venue=None,
        contact=None,
        chat=SimpleNamespace(id=-100200, type="supergroup", title="Test", full_name=None),
        from_user=SimpleNamespace(id=42, is_bot=False, full_name="Owner"),
        sender_business_bot=None,
        business_connection_id=None,
        message_thread_id=7,
        is_topic_message=True,
        forum_topic_created=None,
        reply_to_message=None,
        quote=None,
        media_group_id=None,
        message_id=99,
        date=None,
    )
    update = SimpleNamespace(update_id=10, effective_message=message, message=message)

    await adapter._handle_media_message(update, SimpleNamespace())

    document.get_file.assert_not_awaited()
    adapter.handle_message.assert_awaited_once()
    call = adapter.handle_message.await_args
    assert call is not None
    routed = call.args[0]
    assert routed.text.startswith(
        "/media-recovery Recover the exact oversized Telegram document"
    )
    assert "chat_id=-100200, message_id=99" in routed.text
    assert routed.metadata["telegram_media_recovery"]["message_id"] == "99"
