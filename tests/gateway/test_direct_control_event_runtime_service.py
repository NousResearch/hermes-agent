from __future__ import annotations

from types import SimpleNamespace

from gateway.config import Platform
from gateway.platforms.base import MessageType
from gateway.session import SessionSource


def _make_source(
    *,
    platform: Platform = Platform.QQ_NAPCAT,
    chat_type: str = "dm",
    user_id: str = "179033731",
    chat_id: str = "179033731",
) -> SessionSource:
    return SessionSource(
        platform=platform,
        user_id=user_id,
        user_name="發發發",
        chat_id=chat_id,
        chat_type=chat_type,
    )


def _make_event(
    text: str,
    *,
    platform: Platform = Platform.QQ_NAPCAT,
    message_type: MessageType = MessageType.TEXT,
    media_urls=None,
    command: str = "",
):
    source = _make_source(platform=platform)
    event = SimpleNamespace(
        source=source,
        text=text,
        message_type=message_type,
        media_urls=media_urls or [],
    )
    event.get_command = lambda: command
    return event


def test_extract_platform_text_event_context_accepts_matching_text_turn():
    from gateway.direct_control_event_runtime_service import extract_platform_text_event_context

    event = _make_event("在吗")

    source, body = extract_platform_text_event_context(event, platform=Platform.QQ_NAPCAT)

    assert source == event.source
    assert body == "在吗"


def test_extract_platform_text_event_context_rejects_command_media_and_other_platforms():
    from gateway.direct_control_event_runtime_service import extract_platform_text_event_context

    assert extract_platform_text_event_context(
        _make_event("/status", command="/status"),
        platform=Platform.QQ_NAPCAT,
    ) == (None, "")
    assert extract_platform_text_event_context(
        _make_event("图来了", media_urls=["https://example.com/1.png"]),
        platform=Platform.QQ_NAPCAT,
    ) == (None, "")
    assert extract_platform_text_event_context(
        _make_event("在吗", platform=Platform.WEIXIN),
        platform=Platform.QQ_NAPCAT,
    ) == (None, "")


def test_build_admin_platform_text_context_includes_admin_flags():
    from gateway.direct_control_event_runtime_service import build_admin_platform_text_context

    event = _make_event("看看待处理的好友申请")

    context = build_admin_platform_text_context(
        event,
        platform=Platform.QQ_NAPCAT,
        configured_admin_user_ids_fn=lambda current_platform: ["179033731"],
        is_admin_user_fn=lambda source: True,
    )

    assert context["source"] == event.source
    assert context["body"] == "看看待处理的好友申请"
    assert context["admin_ids_configured"] is True
    assert context["is_admin_user"] is True


def test_build_admin_platform_text_context_handles_non_matching_turn():
    from gateway.direct_control_event_runtime_service import build_admin_platform_text_context

    event = _make_event("/status", command="/status")

    context = build_admin_platform_text_context(
        event,
        platform=Platform.QQ_NAPCAT,
        configured_admin_user_ids_fn=lambda current_platform: ["179033731"],
        is_admin_user_fn=lambda source: True,
    )

    assert context == {
        "source": None,
        "body": "",
        "admin_ids_configured": False,
        "is_admin_user": False,
    }
