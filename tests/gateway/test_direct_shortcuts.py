from types import SimpleNamespace
from unittest.mock import MagicMock

from gateway.direct_shortcuts import run_direct_shortcut_handlers


def test_run_direct_shortcut_handlers_preserves_canonical_order():
    event = object()
    runner = SimpleNamespace(
        _try_handle_background_job_status_shortcut=MagicMock(return_value=None),
        _try_handle_runtime_status_shortcut=MagicMock(return_value=None),
        _try_handle_admin_qq_send_shortcut=MagicMock(return_value=None),
        _try_handle_admin_weixin_send_shortcut=MagicMock(return_value=None),
        _try_handle_admin_qq_group_runtime_status=MagicMock(return_value=None),
        _try_handle_admin_qq_group_control=MagicMock(return_value="group-control"),
        _try_handle_admin_qq_group_moderation=MagicMock(return_value=None),
        _try_handle_admin_weixin_group_runtime_status=MagicMock(return_value=None),
        _try_handle_admin_weixin_group_control=MagicMock(return_value=None),
        _try_handle_admin_weixin_group_moderation=MagicMock(return_value=None),
        _try_handle_admin_qq_intel_control=MagicMock(return_value="intel-control"),
        _try_handle_admin_qq_social_control=MagicMock(return_value=None),
    )

    result = run_direct_shortcut_handlers(runner, event)

    assert result == "group-control"
    runner._try_handle_admin_qq_group_control.assert_called_once_with(event)
    runner._try_handle_admin_qq_intel_control.assert_not_called()


def test_run_direct_shortcut_handlers_passes_history_only_to_history_aware_handlers():
    event = object()
    history = [{"role": "user", "content": "往 QQ 群 123 发：你好"}]
    runner = SimpleNamespace(
        _try_handle_background_job_status_shortcut=MagicMock(return_value=None),
        _try_handle_runtime_status_shortcut=MagicMock(return_value=None),
        _try_handle_admin_qq_send_shortcut=MagicMock(return_value="sent"),
        _try_handle_admin_weixin_send_shortcut=MagicMock(return_value=None),
        _try_handle_admin_qq_group_runtime_status=MagicMock(return_value=None),
        _try_handle_admin_qq_group_control=MagicMock(return_value=None),
        _try_handle_admin_qq_group_moderation=MagicMock(return_value=None),
        _try_handle_admin_weixin_group_runtime_status=MagicMock(return_value=None),
        _try_handle_admin_weixin_group_control=MagicMock(return_value=None),
        _try_handle_admin_weixin_group_moderation=MagicMock(return_value=None),
        _try_handle_admin_qq_intel_control=MagicMock(return_value=None),
        _try_handle_admin_qq_social_control=MagicMock(return_value=None),
    )

    result = run_direct_shortcut_handlers(
        runner,
        event,
        conversation_history=history,
    )

    assert result == "sent"
    runner._try_handle_background_job_status_shortcut.assert_called_once_with(event)
    runner._try_handle_runtime_status_shortcut.assert_called_once_with(event)
    runner._try_handle_admin_qq_send_shortcut.assert_called_once_with(
        event,
        conversation_history=history,
    )


def test_run_direct_shortcut_handlers_falls_back_to_direct_control_router():
    event = object()
    router = SimpleNamespace(
        _try_handle_admin_qq_send_shortcut=MagicMock(return_value="router-sent"),
    )
    runner = SimpleNamespace(
        _try_handle_background_job_status_shortcut=MagicMock(return_value=None),
        _try_handle_runtime_status_shortcut=MagicMock(return_value=None),
        _get_direct_control_router=MagicMock(return_value=router),
    )

    result = run_direct_shortcut_handlers(
        runner,
        event,
        conversation_history=[{"role": "user", "content": "发消息"}],
    )

    assert result == "router-sent"
    runner._get_direct_control_router.assert_called_once_with()
    router._try_handle_admin_qq_send_shortcut.assert_called_once_with(
        event,
        conversation_history=[{"role": "user", "content": "发消息"}],
    )


def test_run_direct_shortcut_handlers_records_trace_for_handled_shortcut():
    event = SimpleNamespace(
        text="停止QQ 群 192903718 的监听采集",
        source=SimpleNamespace(platform=SimpleNamespace(value="qq_napcat"), chat_type="dm", chat_id="179033731"),
    )
    runner = SimpleNamespace(
        _recent_direct_shortcuts=[],
        _try_handle_background_job_status_shortcut=MagicMock(return_value=None),
        _try_handle_runtime_status_shortcut=MagicMock(return_value=None),
        _try_handle_admin_qq_send_shortcut=MagicMock(return_value=None),
        _try_handle_admin_weixin_send_shortcut=MagicMock(return_value=None),
        _try_handle_admin_qq_group_runtime_status=MagicMock(return_value=None),
        _try_handle_admin_qq_group_control=MagicMock(return_value="已切成监听"),
        _try_handle_admin_qq_group_moderation=MagicMock(return_value=None),
        _try_handle_admin_weixin_group_runtime_status=MagicMock(return_value=None),
        _try_handle_admin_weixin_group_control=MagicMock(return_value=None),
        _try_handle_admin_weixin_group_moderation=MagicMock(return_value=None),
        _try_handle_admin_qq_intel_control=MagicMock(return_value=None),
        _try_handle_admin_qq_social_control=MagicMock(return_value=None),
    )

    result = run_direct_shortcut_handlers(runner, event)

    assert result == "已切成监听"
    assert len(runner._recent_direct_shortcuts) == 1
    trace = runner._recent_direct_shortcuts[0]
    assert trace["matched_handler"] == "_try_handle_admin_qq_group_control"
    assert trace["attempt_count"] >= 6
    assert trace["text_preview"] == "停止QQ 群 192903718 的监听采集"
