"""See gateway/RUNTIME_SERVICES.md. Marked dead_runtime_service so suites can
optionally filter with ``-m "not dead_runtime_service"``; default still runs.
"""
from __future__ import annotations
import pytest



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


def test_normalize_direct_control_body_strips_leading_wrapper_phrases():
    from gateway.direct_control_event_runtime_service import normalize_direct_control_body

    assert normalize_direct_control_body("我让你停止QQ 群 192903718 的监听采集") == (
        "停止QQ 群 192903718 的监听采集"
    )
    assert normalize_direct_control_body("帮我把这个群切成只监听") == "这个群切成只监听"
    assert normalize_direct_control_body("麻烦你把往 QQ 群 192903718 发：你好") == (
        "往 QQ 群 192903718 发：你好"
    )


def test_normalize_direct_control_body_strips_stacked_wrapper_phrases():
    from gateway.direct_control_event_runtime_service import normalize_direct_control_body

    assert normalize_direct_control_body("我是说 帮我把这个群切成只监听") == "这个群切成只监听"


def test_shared_oral_intents_exports_direct_control_wrapper_patterns():
    from gateway.shared_oral_intents import DIRECT_CONTROL_WRAPPER_PATTERNS

    assert DIRECT_CONTROL_WRAPPER_PATTERNS
    assert all(hasattr(pattern, "sub") for pattern in DIRECT_CONTROL_WRAPPER_PATTERNS)


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


def test_build_admin_platform_text_context_uses_normalized_body():
    from gateway.direct_control_event_runtime_service import build_admin_platform_text_context

    event = _make_event("我让你停止QQ 群 192903718 的监听采集")

    context = build_admin_platform_text_context(
        event,
        platform=Platform.QQ_NAPCAT,
        configured_admin_user_ids_fn=lambda current_platform: ["179033731"],
        is_admin_user_fn=lambda source: True,
    )

    assert context["body"] == "停止QQ 群 192903718 的监听采集"


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
