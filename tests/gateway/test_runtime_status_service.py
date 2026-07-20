from __future__ import annotations

import time
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform
from gateway.background_jobs import BackgroundJobStore
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source(*, chat_type: str = "dm", chat_id: str = "179033731") -> SessionSource:
    return SessionSource(
        platform=Platform.QQ_NAPCAT,
        user_id="179033731",
        user_name="發發發",
        chat_id=chat_id,
        chat_type=chat_type,
    )


def test_build_runtime_status_summary_collects_live_runtime_snapshot(tmp_path):
    from gateway.runtime_status_service import build_runtime_status_summary

    source = _make_source(chat_type="group", chat_id="726109087")
    session_key = build_session_key(source)
    store = BackgroundJobStore(db_path=tmp_path / "background_jobs.db")
    store.create_job(
        task_id="bg_1",
        prompt="继续处理线上问题",
        source=source,
        session_key=session_key,
        worker_name="铁柱",
        job_kind="auto",
    )
    store.mark_job_running("bg_1")
    running_agent = MagicMock()
    running_agent.get_activity_summary.return_value = {
        "api_call_count": 3,
        "max_iterations": 60,
        "current_tool": "delegate_task",
        "last_activity_desc": "running: delegate_task",
    }
    pending_task = MagicMock()
    pending_task.done.return_value = False
    sentinel = object()

    runner = SimpleNamespace(
        _running_agents={session_key: running_agent},
        _running_agents_ts={session_key: 100.0},
        _auto_vision_tasks={"vision:1": pending_task},
        _auto_vision_cache={"cache:1": {"status": "success"}},
        _runtime_session_metadata=lambda key: {
            "platform": "qq_napcat",
            "chat_type": "group",
            "chat_id": "726109087",
        },
        _get_background_job_store=lambda: store,
        _ensure_background_job_state=lambda: None,
        _ensure_auto_vision_state=lambda: None,
        _prune_auto_vision_state=lambda: None,
        _auto_vision_cooldown_remaining=lambda: (9.0, "timeout"),
        _build_runtime_group_archive_summary=lambda: {
            "raw_message_count": 50,
            "raw_scope_count": 3,
            "platforms": {
                "qq_napcat": {"raw_message_count": 42, "raw_scope_count": 2},
                "weixin": {"raw_message_count": 8, "raw_scope_count": 1},
            },
        },
        _build_runtime_model_summary=lambda: {"configured_model": "gpt-5.4"},
        _build_runtime_approval_summary=lambda: {"pending_count": 1},
        _build_runtime_group_monitoring_summary=lambda: {
            "active_collect_only_groups": 2,
            "active_worker_count": 1,
            "platform_counts": {"qq_napcat": 1, "weixin": 1},
            "groups": [
                {
                    "platform": "qq_napcat",
                    "platform_label": "QQ 群",
                    "group_id": "726109087",
                    "group_name": "项目群",
                    "worker_names": ["钢镚"],
                    "daily_report_enabled": True,
                }
            ],
        },
        _build_runtime_direct_shortcut_summary=lambda: {
            "recent_count": 1,
            "recent": [
                {
                    "matched_handler": "_try_handle_admin_qq_group_control",
                    "text_preview": "停止QQ 群 192903718 的监听采集",
                }
            ],
        },
    )

    summary = build_runtime_status_summary(runner, now_ts=130.0, pending_sentinel=sentinel)

    assert summary["active_sessions_count"] == 1
    assert summary["active_sessions"][0]["current_tool"] == "delegate_task"
    assert summary["background_jobs"]["active_count"] == 1
    assert summary["background_jobs"]["active"][0]["preview"] == "继续处理线上问题"
    assert summary["auto_vision"]["state"] == "cooldown"
    assert summary["group_archive"]["raw_message_count"] == 50
    assert summary["model"]["configured_model"] == "gpt-5.4"
    assert summary["approvals"]["pending_count"] == 1
    assert summary["group_monitoring"]["platform_counts"] == {"qq_napcat": 1, "weixin": 1}
    assert summary["group_monitoring"]["active_collect_only_groups"] == 2
    assert summary["direct_shortcuts"]["recent_count"] == 1


@pytest.mark.asyncio
async def test_render_status_command_renders_status_lines_with_foreground_and_background():
    from gateway.runtime_status_service import render_status_command

    source = _make_source()
    session_entry = SessionEntry(
        session_key=build_session_key(source),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.QQ_NAPCAT,
        chat_type=source.chat_type,
        total_tokens=321,
    )
    running_agent = MagicMock()
    running_agent.get_activity_summary.return_value = {
        "api_call_count": 4,
        "max_iterations": 60,
        "current_tool": "delegate_task",
        "last_activity_desc": "running: delegate_task",
    }
    sentinel = object()

    runner = SimpleNamespace(
        adapters={Platform.QQ_NAPCAT: MagicMock()},
        session_store=SimpleNamespace(get_or_create_session=lambda current_source: session_entry),
        _session_db=SimpleNamespace(get_session_title=lambda session_id: "状态页会话"),
        _running_agents={session_entry.session_key: running_agent},
        _running_agents_ts={session_entry.session_key: time.time() - 12},
        _pending_approvals={},
        _auto_vision_tasks={},
        _background_jobs_for_source=lambda current_source: [
            {
                "task_id": "bg_123",
                "status": "running",
                "worker_name": "铁柱",
                "started_at": time.time() - 25,
                "created_at": time.time() - 30,
                "updated_at": time.time(),
            }
        ],
        _get_background_job_store=lambda: SimpleNamespace(count_pending_approval_requests=lambda session_key: 0),
        _ensure_auto_vision_state=lambda: None,
        _prune_auto_vision_state=lambda: None,
        _auto_vision_cooldown_remaining=lambda: (0.0, ""),
        _build_runtime_group_archive_summary=lambda: {
            "raw_message_count": 50,
            "raw_scope_count": 3,
            "platforms": {
                "qq_napcat": {"raw_message_count": 42, "raw_scope_count": 2},
                "weixin": {"raw_message_count": 8, "raw_scope_count": 1},
            },
        },
        _build_runtime_model_summary=lambda: {
            "configured_model": "gpt-5.4",
            "active_model": "gpt-5.4",
            "active_provider": "custom",
            "fallback_active": False,
            "fallback_pinned": False,
        },
        _build_runtime_group_monitoring_summary=lambda: {
            "active_collect_only_groups": 2,
            "platform_counts": {"qq_napcat": 1, "weixin": 1},
            "groups": [
                {
                    "platform": "qq_napcat",
                    "platform_label": "QQ 群",
                    "group_id": "726109087",
                    "group_name": "项目群",
                    "worker_names": ["钢镚"],
                    "daily_report_enabled": True,
                },
                {
                    "platform": "weixin",
                    "platform_label": "微信群",
                    "chat_id": "wx-group-1",
                    "group_name": "微信群项目组",
                    "worker_names": [],
                    "daily_report_enabled": True,
                },
            ],
        },
        _build_runtime_direct_shortcut_summary=lambda: {
            "recent_count": 1,
            "recent": [
                {
                    "matched_handler": "_try_handle_admin_qq_group_control",
                    "text_preview": "停止QQ 群 192903718 的监听采集",
                }
            ],
        },
        _format_background_job_age=lambda job: "25s",
    )

    event = SimpleNamespace(source=source)

    result = await render_status_command(runner, event, pending_sentinel=sentinel)

    assert "Hermes Gateway Status" in result
    assert "状态页会话" in result
    assert "Foreground" in result
    assert "delegate_task" in result
    assert "Model" in result
    assert "Group Archive" in result
    assert "Group Monitoring" in result
    assert "Direct Shortcuts" in result
    assert "Background Jobs" in result
    assert "`bg_123`" in result
    assert "微信群项目组" in result
