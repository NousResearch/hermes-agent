from __future__ import annotations

from unittest.mock import MagicMock, patch

from gateway.config import Platform
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


def test_match_admin_platform_send_request_returns_none_for_non_admin():
    from gateway.send_runtime_service import match_admin_platform_send_request

    source = _make_source()

    tool_args, error = match_admin_platform_send_request(
        source=source,
        body="往 QQ 群 192903718 发：绿帽哥！",
        conversation_history=None,
        admin_ids_configured=True,
        is_admin_user=False,
        inline_extractor=lambda body: ("group:192903718", "绿帽哥！"),
        history_target_extractor=lambda source, history: "",
        direct_target_extractor=lambda source, body: "group:192903718",
        query_prompt_formatter=lambda target: f"发到 {target}",
    )

    assert tool_args is None
    assert error is None


def test_match_admin_platform_send_request_passes_expected_context():
    from gateway.send_runtime_service import match_admin_platform_send_request

    source = _make_source()
    history = [{"role": "user", "content": "往 QQ 群 192903718 发：绿帽哥！"}]

    with patch(
        "gateway.send_runtime_service.match_send_request",
        return_value=({"target": "group:192903718", "message": "绿帽哥！"}, None),
    ) as matcher:
        tool_args, error = match_admin_platform_send_request(
            source=source,
            body="发这句",
            conversation_history=history,
            admin_ids_configured=True,
            is_admin_user=True,
            inline_extractor=lambda body: ("group:192903718", "绿帽哥！"),
            history_target_extractor=lambda source, history: "group:192903718",
            direct_target_extractor=lambda source, body: "group:192903718",
            query_prompt_formatter=lambda target: f"发到 {target}",
        )

    assert error is None
    assert tool_args == {"target": "group:192903718", "message": "绿帽哥！"}
    matcher.assert_called_once_with(
        source=source,
        body="发这句",
        conversation_history=history,
        inline_extractor=matcher.call_args.kwargs["inline_extractor"],
        history_target_extractor=matcher.call_args.kwargs["history_target_extractor"],
        direct_target_extractor=matcher.call_args.kwargs["direct_target_extractor"],
        looks_like_send_query=matcher.call_args.kwargs["looks_like_send_query"],
        looks_like_send_confirmation=matcher.call_args.kwargs["looks_like_send_confirmation"],
        extract_send_confirmation_message=matcher.call_args.kwargs["extract_send_confirmation_message"],
        query_prompt_formatter=matcher.call_args.kwargs["query_prompt_formatter"],
    )


def test_run_admin_send_shortcut_formats_success_reply():
    from gateway.send_runtime_service import run_admin_send_shortcut

    logger = MagicMock()

    result = run_admin_send_shortcut(
        tool_args={"target": "group:192903718", "message": "绿帽哥！"},
        shortcut_error=None,
        tool_runner=lambda tool_args: {"success": True},
        error_prefix="QQ 发消息执行失败",
        reply_formatter=lambda tool_args: "已发到 QQ 群 192903718：绿帽哥！",
        logger=logger,
    )

    assert result == "已发到 QQ 群 192903718：绿帽哥！"
    logger.warning.assert_not_called()


def test_run_admin_send_shortcut_surfaces_tool_exception():
    from gateway.send_runtime_service import run_admin_send_shortcut

    logger = MagicMock()

    result = run_admin_send_shortcut(
        tool_args={"target": "group:192903718", "message": "绿帽哥！"},
        shortcut_error=None,
        tool_runner=lambda tool_args: (_ for _ in ()).throw(RuntimeError("boom")),
        error_prefix="QQ 发消息执行失败",
        reply_formatter=lambda tool_args: "unused",
        logger=logger,
    )

    assert result == "QQ 发消息执行失败：boom"
    logger.warning.assert_called_once()
