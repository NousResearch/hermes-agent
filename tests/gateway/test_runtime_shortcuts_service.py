from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from gateway.config import Platform
from gateway.platforms.base import MessageType
from gateway.session import SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.QQ_NAPCAT,
        user_id="179033731",
        user_name="發發發",
        chat_id="179033731",
        chat_type="dm",
    )


def _make_event(text: str):
    source = _make_source()
    return SimpleNamespace(
        text=text,
        source=source,
        message_type=MessageType.TEXT,
        get_command=lambda: "",
    )


def test_format_background_job_short_status_includes_preview_error_and_pending_approval():
    from gateway.runtime_shortcuts_service import format_background_job_short_status

    runner = SimpleNamespace(
        _format_background_job_age=lambda job: "25s",
        _get_background_job_store=lambda: SimpleNamespace(
            count_pending_approval_requests=lambda session_key: 2
        ),
    )

    result = format_background_job_short_status(
        runner,
        {
            "task_id": "bg_123",
            "status": "running",
            "worker_name": "铁柱",
            "preview": "继续处理线上问题",
            "error": "ssh timeout",
            "session_key": "qq_napcat:dm:179033731",
        },
    )

    assert "后台任务 `bg_123` 当前进行中" in result
    assert "负责人：铁柱" in result
    assert "内容：继续处理线上问题" in result
    assert "错误：ssh timeout" in result
    assert "授权审批" in result


def test_try_handle_runtime_status_shortcut_prefers_foreground_then_background():
    from gateway.runtime_shortcuts_service import try_handle_runtime_status_shortcut

    source = _make_source()
    session_key = build_session_key(source)
    running_agent = MagicMock()
    running_agent.get_activity_summary.return_value = {
        "api_call_count": 4,
        "max_iterations": 60,
        "current_tool": "delegate_task",
        "last_activity_desc": "running: delegate_task",
    }
    runner = SimpleNamespace(
        _looks_like_runtime_status_query=lambda text: True,
        _session_key_for_source=lambda current_source: session_key,
        _running_agents={session_key: running_agent},
        _background_jobs_for_source=lambda current_source: [
            {"task_id": "bg_999", "status": "running", "updated_at": time.time()}
        ],
        _format_running_session_short_status=lambda key, agent: "当前前台这轮还在跑：delegate_task。",
        _format_background_job_short_status=lambda job: "后台任务状态",
    )

    result = try_handle_runtime_status_shortcut(runner, _make_event("你现在忙什么？"), pending_sentinel=object())

    assert result == "当前前台这轮还在跑：delegate_task。"


def test_try_handle_background_job_status_shortcut_returns_latest_job_status():
    from gateway.runtime_shortcuts_service import try_handle_background_job_status_shortcut

    source = _make_source()
    now = time.time()
    runner = SimpleNamespace(
        _looks_like_background_status_query=lambda text: True,
        _background_jobs_for_source=lambda current_source: [
            {"task_id": "bg_old", "status": "completed", "updated_at": now - 20},
            {"task_id": "bg_new", "status": "running", "updated_at": now},
        ],
        _format_background_job_short_status=lambda job: f"picked:{job['task_id']}",
    )

    result = try_handle_background_job_status_shortcut(runner, _make_event("前面那个任务还在做吗"))

    assert result == "picked:bg_new"
