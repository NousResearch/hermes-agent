"""See gateway/RUNTIME_SERVICES.md. Marked dead_runtime_service so suites can
optionally filter with ``-m "not dead_runtime_service"``; default still runs.
"""
from __future__ import annotations
import pytest



from unittest.mock import patch

from gateway.config import Platform
from gateway.session import SessionSource


def _make_source(
    *,
    platform: Platform = Platform.QQ_NAPCAT,
    chat_type: str = "group",
    user_id: str = "179033731",
    chat_id: str = "726109087",
) -> SessionSource:
    return SessionSource(
        platform=platform,
        user_id=user_id,
        user_name="發發發",
        chat_id=chat_id,
        chat_type=chat_type,
    )


def test_try_handle_admin_platform_group_runtime_status_returns_none_for_empty_source():
    from gateway.group_runtime_status_runtime_service import (
        try_handle_admin_platform_group_runtime_status,
    )

    result = try_handle_admin_platform_group_runtime_status(
        source=None,
        body="这个群现在谁在监听",
        conversation_history=None,
        admin_ids_configured=True,
        is_admin_user=True,
        looks_like_group_runtime_status_query=lambda body: True,
        target_extractor=lambda source, body: "group:726109087",
        history_target_extractor=lambda source, history: "",
        status_loader=lambda target: {"platform_label": "QQ 群"},
    )

    assert result is None


def test_try_handle_admin_platform_group_runtime_status_returns_none_when_no_target():
    from gateway.group_runtime_status_runtime_service import (
        try_handle_admin_platform_group_runtime_status,
    )

    source = _make_source()

    with patch(
        "gateway.group_runtime_status_runtime_service.match_group_runtime_status_request",
        return_value=None,
    ) as matcher:
        result = try_handle_admin_platform_group_runtime_status(
            source=source,
            body="这个群现在谁在监听",
            conversation_history=[],
            admin_ids_configured=True,
            is_admin_user=True,
            looks_like_group_runtime_status_query=lambda body: True,
            target_extractor=lambda source, body: "group:726109087",
            history_target_extractor=lambda source, history: "",
            status_loader=lambda target: {"platform_label": "QQ 群"},
        )

    assert result is None
    matcher.assert_called_once()


def test_try_handle_admin_platform_group_runtime_status_formats_loaded_status():
    from gateway.group_runtime_status_runtime_service import (
        try_handle_admin_platform_group_runtime_status,
    )

    source = _make_source()

    with patch(
        "gateway.group_runtime_status_runtime_service.match_group_runtime_status_request",
        return_value="group:726109087",
    ) as matcher:
        result = try_handle_admin_platform_group_runtime_status(
            source=source,
            body="这个群现在谁在监听，日报开了吗？",
            conversation_history=[{"role": "user", "content": "这个群"}],
            admin_ids_configured=True,
            is_admin_user=True,
            looks_like_group_runtime_status_query=lambda body: True,
            target_extractor=lambda source, body: "group:726109087",
            history_target_extractor=lambda source, history: "",
            status_loader=lambda target: {
                "platform_label": "QQ 群",
                "target_label": "726109087",
                "effective_mode": "collect_only",
                "can_reply_in_group": False,
                "archive_enabled": True,
                "daily_report_enabled": True,
                "daily_targets": ["qq_napcat:dm:179033731"],
                "manual_targets": ["qq_napcat:group:726109087"],
                "worker_names": ["钢镚"],
            },
        )

    assert "QQ 群 726109087 当前模式：collect_only。" in result
    assert "日报目标：qq_napcat:dm:179033731" in result
    assert "当前监听情报员：钢镚" in result
    matcher.assert_called_once_with(
        source=source,
        body="这个群现在谁在监听，日报开了吗？",
        conversation_history=[{"role": "user", "content": "这个群"}],
        admin_ids_configured=True,
        is_admin_user=True,
        looks_like_group_runtime_status_query=matcher.call_args.kwargs["looks_like_group_runtime_status_query"],
        target_extractor=matcher.call_args.kwargs["target_extractor"],
        history_target_extractor=matcher.call_args.kwargs["history_target_extractor"],
    )
