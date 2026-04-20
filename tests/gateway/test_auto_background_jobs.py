"""Tests for auto-detached gateway background jobs."""

import asyncio
import json
import time
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.background_delivery_service import recover_stale_background_jobs_once
from gateway.config import GatewayConfig, Platform, PlatformConfig, load_gateway_config
from gateway.background_jobs import BackgroundJobStore
from gateway.direct_shortcut_runtime_service import (
    get_direct_control_router,
    try_handle_direct_gateway_shortcuts,
)
from gateway.group_runtime_platform_specs import (
    GroupArchiveRuntimePlatformSpec,
    GroupMonitoringRuntimePlatformSpec,
)
from gateway.platforms.base import MessageEvent, MessageType
from gateway.runtime_status_service import build_runtime_status_summary
from gateway.runtime_shortcuts_service import try_handle_background_job_status_shortcut
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source(
    *,
    chat_type: str = "dm",
    user_id: str = "179033731",
    chat_id: str = "179033731",
) -> SessionSource:
    return SessionSource(
        platform=Platform.QQ_NAPCAT,
        user_id=user_id,
        user_name="發發發",
        chat_id=chat_id,
        chat_type=chat_type,
    )


def _make_event(
    text: str,
    *,
    chat_type: str = "dm",
    user_id: str = "179033731",
    chat_id: str = "179033731",
) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=_make_source(chat_type=chat_type, user_id=user_id, chat_id=chat_id),
        message_id="m1",
        message_type=MessageType.TEXT,
    )


def _make_session_entry() -> SessionEntry:
    source = _make_source()
    return SessionEntry(
        session_key=build_session_key(source),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.QQ_NAPCAT,
        chat_type="dm",
        total_tokens=42,
    )


def _make_runner(*, auto_background_work: bool = True, employee_routes=None):
    from gateway.run import GatewayRunner

    extra = {"auto_background_work": auto_background_work}
    if employee_routes is not None:
        extra["employee_routes"] = employee_routes

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.QQ_NAPCAT: PlatformConfig(
                enabled=True,
                token="***",
                extra=extra,
            )
        },
        auto_background_work=auto_background_work,
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.QQ_NAPCAT: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    session_entry = _make_session_entry()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = [
        {"role": "user", "content": "之前让你把线上部署问题查清楚。"},
        {"role": "assistant", "content": "收到，我继续排查。"},
    ]
    runner.session_store.has_any_sessions.return_value = True
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._background_tasks = set()
    runner._session_db = MagicMock()
    runner._session_db.get_session_title.return_value = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._effective_model = None
    runner._effective_provider = None
    runner._session_model_overrides = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = MagicMock()
    runner._auto_vision_cache = {}
    runner._auto_vision_tasks = {}
    runner._auto_vision_unhealthy_until = 0.0
    runner._auto_vision_unhealthy_reason = ""
    runner._update_prompt_pending = {}
    runner._failed_platforms = {}
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._get_unauthorized_dm_behavior = lambda _platform: "pair"
    runner._set_session_env = lambda _context: None
    runner._run_agent = AsyncMock(return_value={"final_response": "前台回复"})
    runner._load_reasoning_config = lambda: None
    runner.delivery_router = MagicMock()
    runner.pairing_store = MagicMock()
    runner._launch_background_worker = MagicMock(
        return_value={"launcher_type": "subprocess", "launcher_pid": 4321}
    )
    return runner


def _list_durable_jobs(runner):
    return runner._get_background_job_store().list_jobs()


def _latest_durable_job(runner):
    jobs = _list_durable_jobs(runner)
    assert jobs
    return jobs[-1]


def _install_background_store(runner, tmp_path):
    store = BackgroundJobStore(db_path=tmp_path / "background_jobs.db")
    runner._background_job_store = store
    return store


def _create_durable_background_job(
    runner,
    tmp_path,
    *,
    task_id: str,
    source: SessionSource | None = None,
    prompt: str,
    status: str = "running",
    worker_name: str = "",
    job_kind: str = "auto",
    session_key: str | None = None,
):
    source = source or _make_source()
    store = getattr(runner, "_background_job_store", None)
    if store is None:
        store = _install_background_store(runner, tmp_path)
    resolved_session_key = session_key or runner._session_key_for_source(source)
    store.create_job(
        task_id=task_id,
        prompt=prompt,
        source=source,
        session_key=resolved_session_key,
        job_kind=job_kind,
        worker_name=worker_name,
    )
    normalized_status = str(status or "queued").strip().lower()
    if normalized_status == "running":
        store.mark_job_running(task_id)
    elif normalized_status == "cancelling":
        store.mark_job_running(task_id)
        store.mark_job_cancelling(task_id)
    elif normalized_status == "completed":
        store.mark_job_running(task_id)
        store.mark_job_completed(task_id, raw_response="done")
    elif normalized_status == "failed":
        store.mark_job_running(task_id)
        store.mark_job_failed(task_id, error="failed")
    elif normalized_status == "cancelled":
        store.mark_job_cancelled(task_id, reason="cancelled")
    return store.get_job(task_id)


@pytest.mark.asyncio
async def test_handle_message_auto_backgrounds_long_work_request():
    runner = _make_runner(auto_background_work=True)
    event = _make_event("继续把上面的部署问题排查并修复，全部做完后给我汇报。")

    with patch(
        "gateway.run.asyncio.create_task",
        side_effect=lambda coro, *args, **kwargs: (coro.close(), MagicMock())[1],
    ):
        result = await runner._handle_message(event)

    assert "转后台" in result
    assert runner._run_agent.await_count == 0
    job = _latest_durable_job(runner)
    assert job["kind"] == "auto"
    assert job["status"] == "queued"


@pytest.mark.asyncio
async def test_handle_message_dispatches_design_polish_to_tiezhu_worker_when_heuristic_route_is_configured():
    runner = _make_runner(
        auto_background_work=True,
        employee_routes=[
            {
                "worker_name": "铁柱",
                "preloaded_skills": ["frontend-design-pro"],
                "match_modes": ["explicit", "heuristic"],
                "routing_hints": {
                    "action_terms": ["打磨", "优化"],
                    "subject_terms": ["页面", "样式", "排版"],
                    "pain_terms": ["粗糙"],
                },
            }
        ],
    )
    event = _make_event("把 fafafa-page 打磨一下，页面太粗糙了，顺手调调样式和排版。")

    with patch(
        "gateway.run.asyncio.create_task",
        side_effect=lambda coro, *args, **kwargs: (coro.close(), MagicMock())[1],
    ):
        result = await runner._handle_message(event)

    assert "铁柱" in result
    assert runner._run_agent.await_count == 0
    job = _latest_durable_job(runner)
    assert job["kind"] == "auto"
    assert job["worker_name"] == "铁柱"
    assert job["preloaded_skills"] == ["frontend-design-pro"]


@pytest.mark.asyncio
async def test_handle_message_uses_recent_context_for_short_design_follow_up_when_heuristic_route_is_configured():
    runner = _make_runner(
        auto_background_work=True,
        employee_routes=[
            {
                "worker_name": "铁柱",
                "preloaded_skills": ["frontend-design-pro"],
                "match_modes": ["explicit", "heuristic"],
                "routing_hints": {
                    "action_terms": ["打磨", "优化"],
                    "subject_terms": ["页面", "样式", "排版"],
                    "pain_terms": ["粗糙"],
                },
            }
        ],
    )
    runner.session_store.load_transcript.return_value = [
        {"role": "assistant", "content": "fafafa-page 主页已经能打开了，不过页面样式和排版还比较粗糙。"},
        {"role": "user", "content": "知道了。"},
    ]
    event = _make_event("打磨一下，好粗糙。")

    with patch(
        "gateway.run.asyncio.create_task",
        side_effect=lambda coro, *args, **kwargs: (coro.close(), MagicMock())[1],
    ):
        result = await runner._handle_message(event)

    assert "铁柱" in result
    job = _latest_durable_job(runner)
    assert job["worker_name"] == "铁柱"
    assert job["preloaded_skills"] == ["frontend-design-pro"]


@pytest.mark.asyncio
async def test_handle_message_employee_followup_does_not_route_to_intel_worker_control():
    runner = _make_runner(
        auto_background_work=True,
        employee_routes=[
            {
                "worker_name": "铁柱",
                "preloaded_skills": ["frontend-design-pro"],
                "match_modes": ["explicit"],
            }
        ],
    )
    runner.config.platforms[Platform.QQ_NAPCAT].extra["admin_users"] = ["179033731"]
    runner.session_store.load_transcript.return_value = [
        {"role": "assistant", "content": "公司主页还比较粗糙，交给铁柱继续打磨会更合适。"},
        {"role": "user", "content": "知道了。"},
    ]
    event = _make_event("让铁柱继续优化公司主页")

    with (
        patch(
            "gateway.run.asyncio.create_task",
            side_effect=lambda coro, *args, **kwargs: (coro.close(), MagicMock())[1],
        ),
        patch("tools.messaging_control_tool.messaging_control_tool") as control_mock,
    ):
        result = await runner._handle_message(event)

    control_mock.assert_not_called()
    assert "铁柱" in result
    assert runner._run_agent.await_count == 0
    job = _latest_durable_job(runner)
    assert job["worker_name"] == "铁柱"
    assert job["preloaded_skills"] == ["frontend-design-pro"]


@pytest.mark.asyncio
async def test_handle_message_without_configured_employee_route_keeps_worker_followup_in_foreground():
    runner = _make_runner(auto_background_work=True, employee_routes=[])
    runner.config.platforms[Platform.QQ_NAPCAT].extra["admin_users"] = ["179033731"]
    runner.session_store.load_transcript.return_value = [
        {"role": "assistant", "content": "公司主页还比较粗糙，交给铁柱继续打磨会更合适。"},
        {"role": "user", "content": "知道了。"},
    ]
    event = _make_event("让铁柱继续优化公司主页")

    with (
        patch(
            "gateway.run.asyncio.create_task",
            side_effect=lambda coro, *args, **kwargs: (coro.close(), MagicMock())[1],
        ),
        patch("tools.messaging_control_tool.messaging_control_tool") as control_mock,
    ):
        result = await runner._handle_message(event)

    control_mock.assert_not_called()
    runner._run_agent.assert_awaited_once()
    assert result == "前台回复"
    assert _list_durable_jobs(runner) == []


def test_direct_gateway_shortcuts_prioritize_group_control_over_intel_control():
    runner = _make_runner(auto_background_work=True)
    event = _make_event("停止QQ 群 192903718 的监听采集,允许开始聊天")

    runner._try_handle_background_job_status_shortcut = MagicMock(return_value=None)
    runner._try_handle_runtime_status_shortcut = MagicMock(return_value=None)
    runner._try_handle_admin_qq_send_shortcut = MagicMock(return_value=None)
    runner._try_handle_admin_qq_group_control = MagicMock(return_value="group-control")
    runner._try_handle_admin_qq_group_moderation = MagicMock(return_value=None)
    runner._try_handle_admin_weixin_group_runtime_status = MagicMock(return_value=None)
    runner._try_handle_admin_weixin_group_control = MagicMock(return_value=None)
    runner._try_handle_admin_weixin_group_moderation = MagicMock(return_value=None)
    runner._try_handle_admin_qq_intel_control = MagicMock(return_value="intel-control")
    runner._try_handle_admin_qq_social_control = MagicMock(return_value=None)
    runner._try_handle_admin_qq_group_runtime_status = MagicMock(return_value=None)

    result = try_handle_direct_gateway_shortcuts(runner, event)

    assert result == "group-control"
    runner._try_handle_admin_qq_group_control.assert_called_once_with(event)
    runner._try_handle_admin_qq_intel_control.assert_not_called()


def test_admin_qq_intel_shortcut_requires_explicit_worker_context():
    runner = _make_runner(auto_background_work=True)
    runner.config.platforms[Platform.QQ_NAPCAT].extra["admin_users"] = ["179033731"]
    event = _make_event("让铁柱继续优化公司主页")

    tool_args, shortcut_error = get_direct_control_router(
        runner
    )._match_admin_qq_intel_control_request(event)

    assert tool_args is None
    assert shortcut_error is None


@pytest.mark.asyncio
async def test_admin_dm_bare_intel_status_phrase_falls_back_to_agent():
    runner = _make_runner(auto_background_work=True)
    runner.config.platforms[Platform.QQ_NAPCAT].extra["admin_users"] = ["179033731"]
    event = _make_event("那个情报员还在吗")

    with patch("tools.messaging_control_tool.messaging_control_tool") as control_mock:
        result = await runner._handle_message(event)

    control_mock.assert_not_called()
    runner._run_agent.assert_awaited_once()
    assert result == "前台回复"


@pytest.mark.asyncio
async def test_admin_dm_bot_alias_intel_phrase_falls_back_to_agent():
    runner = _make_runner(auto_background_work=True)
    runner.config.platforms[Platform.QQ_NAPCAT].extra["admin_users"] = ["179033731"]
    event = _make_event("让马哥现在汇报")

    with (
        patch(
            "gateway.direct_control_router.load_known_qq_intel_worker_names",
            return_value={"马哥"},
        ),
        patch("tools.messaging_control_tool.messaging_control_tool") as control_mock,
    ):
        result = await runner._handle_message(event)

    control_mock.assert_not_called()
    runner._run_agent.assert_awaited_once()
    assert result == "前台回复"


@pytest.mark.asyncio
async def test_admin_dm_verbose_known_worker_report_request_falls_back_to_agent():
    runner = _make_runner(auto_background_work=True)
    runner.config.platforms[Platform.QQ_NAPCAT].extra["admin_users"] = ["179033731"]
    event = _make_event("让钢镚现在汇报一下这个页面为什么回退了")

    with (
        patch(
            "gateway.direct_control_router.load_known_qq_intel_worker_names",
            return_value={"钢镚"},
        ),
        patch("tools.messaging_control_tool.messaging_control_tool") as control_mock,
    ):
        result = await runner._handle_message(event)

    control_mock.assert_not_called()
    runner._run_agent.assert_awaited_once()
    assert result == "前台回复"


@pytest.mark.asyncio
async def test_known_worker_name_does_not_steal_explicit_employee_route_when_message_is_verbose_task():
    runner = _make_runner(
        auto_background_work=True,
        employee_routes=[
            {
                "worker_name": "铁柱",
                "preloaded_skills": ["frontend-design-pro"],
                "match_modes": ["explicit"],
            }
        ],
    )
    runner.config.platforms[Platform.QQ_NAPCAT].extra["admin_users"] = ["179033731"]
    runner.session_store.load_transcript.return_value = [
        {"role": "assistant", "content": "公司主页还比较粗糙，交给铁柱继续打磨会更合适。"},
        {"role": "user", "content": "知道了。"},
    ]
    event = _make_event("让铁柱继续优化公司主页，做完后向我汇报。")

    with (
        patch(
            "gateway.direct_control_router.load_known_qq_intel_worker_names",
            return_value={"铁柱"},
        ),
        patch(
            "gateway.run.asyncio.create_task",
            side_effect=lambda coro, *args, **kwargs: (coro.close(), MagicMock())[1],
        ),
        patch("tools.messaging_control_tool.messaging_control_tool") as control_mock,
    ):
        result = await runner._handle_message(event)

    control_mock.assert_not_called()
    assert "铁柱" in result
    assert runner._run_agent.await_count == 0
    job = _latest_durable_job(runner)
    assert job["worker_name"] == "铁柱"
    assert job["preloaded_skills"] == ["frontend-design-pro"]


@pytest.mark.asyncio
async def test_background_status_shortcut_does_not_steal_explicit_intel_status_query(tmp_path):
    runner = _make_runner(auto_background_work=True)
    runner.config.platforms[Platform.QQ_NAPCAT].extra["admin_users"] = ["179033731"]
    source = _make_source()
    _create_durable_background_job(
        runner,
        tmp_path,
        task_id="bg_intel_conflict",
        source=source,
        prompt="让铁柱继续优化公司主页",
        status="running",
        worker_name="铁柱",
    )
    event = _make_event("看看情报员钢镚现在什么状态。")

    with (
        patch(
            "gateway.direct_control_router.load_known_qq_intel_worker_names",
            return_value={"钢镚"},
        ),
        patch(
            "tools.messaging_control_tool.messaging_control_tool",
            return_value=json.dumps(
                {
                    "success": True,
                    "worker": {
                        "worker_name": "钢镚",
                        "status": "active_collecting",
                        "target_group_id": "726109087",
                    },
                },
                ensure_ascii=False,
            ),
        ) as control_mock,
    ):
        result = await runner._handle_message(event)

    runner._run_agent.assert_not_awaited()
    assert "钢镚" in result
    assert "726109087" in result
    assert "bg_intel_conflict" not in result
    assert control_mock.call_args.args[0] == {
        "platform": "qq_napcat",
        "action": "get_worker",
        "worker_name": "钢镚",
    }


@pytest.mark.asyncio
async def test_handle_message_does_not_route_new_server_task_to_tiezhu_from_stale_design_history():
    runner = _make_runner(
        auto_background_work=True,
        employee_routes=[
            {
                "worker_name": "铁柱",
                "preloaded_skills": ["frontend-design-pro"],
                "match_modes": ["explicit", "heuristic"],
                "routing_hints": {
                    "action_terms": ["打磨", "优化"],
                    "subject_terms": ["页面", "样式", "排版"],
                    "pain_terms": ["粗糙"],
                },
            }
        ],
    )
    runner.session_store.load_transcript.return_value = [
        {"role": "assistant", "content": "fafafa-page 页面样式还比较粗糙，后面可以继续打磨。"},
        {"role": "user", "content": "知道了。"},
    ]
    event = _make_event("上服务器看看日志，把这个问题查清楚。")

    with patch(
        "gateway.run.asyncio.create_task",
        side_effect=lambda coro, *args, **kwargs: (coro.close(), MagicMock())[1],
    ):
        result = await runner._handle_message(event)

    assert "转后台" in result
    assert "铁柱" not in result
    job = _latest_durable_job(runner)
    assert job["worker_name"] == ""
    assert job["preloaded_skills"] == []


@pytest.mark.asyncio
async def test_handle_message_default_employee_route_requires_explicit_worker_mention():
    runner = _make_runner(auto_background_work=True)
    event = _make_event("把 fafafa-page 打磨一下，页面太粗糙了，顺手调调样式和排版。")

    result = await runner._handle_message(event)

    runner._run_agent.assert_awaited_once()
    assert result == "前台回复"
    assert _list_durable_jobs(runner) == []


@pytest.mark.asyncio
async def test_handle_message_plain_worker_name_ping_does_not_dispatch_background_job():
    runner = _make_runner(auto_background_work=True)
    event = _make_event("铁柱在吗？")

    result = await runner._handle_message(event)

    runner._run_agent.assert_awaited_once()
    assert result == "前台回复"
    assert _list_durable_jobs(runner) == []


@pytest.mark.asyncio
async def test_handle_message_does_not_auto_background_bare_continue_without_work_context():
    runner = _make_runner(auto_background_work=True)
    runner.session_store.load_transcript.return_value = [
        {"role": "user", "content": "你们公司到底干啥？"},
        {"role": "assistant", "content": "我们可以一起想想定位和方向。"},
    ]
    event = _make_event("继续")

    result = await runner._handle_message(event)

    runner._run_agent.assert_awaited_once()
    assert result == "前台回复"
    assert _list_durable_jobs(runner) == []


@pytest.mark.asyncio
async def test_handle_message_keeps_short_casual_chat_in_foreground():
    runner = _make_runner(auto_background_work=True)
    event = _make_event("今天天气真好")

    result = await runner._handle_message(event)

    runner._run_agent.assert_awaited_once()
    assert not result or "后台" not in result
    assert _list_durable_jobs(runner) == []


@pytest.mark.asyncio
async def test_admin_dm_can_orally_switch_explicit_group_to_collect_only_monitoring():
    runner = _make_runner(auto_background_work=True)
    runner.config.platforms[Platform.QQ_NAPCAT].extra["admin_users"] = ["179033731"]
    event = _make_event("把 726109087 这个群切成只监听采集，不要走大模型，每天给我日报。")

    with patch(
        "tools.messaging_control_tool.messaging_control_tool",
        return_value=json.dumps(
            {
                "success": True,
                "policy": {
                    "group_id": "726109087",
                    "mode": "collect_only",
                    "archive_enabled": True,
                    "daily_report_enabled": True,
                    "daily_report_target": "qq_napcat:dm:179033731",
                },
            },
            ensure_ascii=False,
        ),
    ) as control_mock:
        result = await runner._handle_message(event)

    runner._run_agent.assert_not_awaited()
    assert "726109087" in result
    assert "监听" in result
    assert "日报" in result
    args = control_mock.call_args.args[0]
    assert args["platform"] == "qq_napcat"
    assert args["action"] == "enable_collect_only"
    assert args["target"] == "group:726109087"
    assert args["daily_report_enabled"] is True
    assert args["daily_report_target"] == "current_user_dm"
    assert args["manual_report_target"] == "current_user_dm"


@pytest.mark.asyncio
async def test_admin_group_can_orally_switch_current_group_to_collect_only_monitoring():
    runner = _make_runner(auto_background_work=True)
    runner.config.platforms[Platform.QQ_NAPCAT].extra["admin_users"] = ["179033731"]
    event = _make_event(
        "这个群只监听，不要走大模型。",
        chat_type="group",
        chat_id="726109087",
    )

    with patch(
        "tools.messaging_control_tool.messaging_control_tool",
        return_value=json.dumps(
            {
                "success": True,
                "policy": {
                    "group_id": "726109087",
                    "mode": "collect_only",
                    "archive_enabled": True,
                    "daily_report_enabled": False,
                },
            },
            ensure_ascii=False,
        ),
    ) as control_mock:
        result = await runner._handle_message(event)

    runner._run_agent.assert_not_awaited()
    assert "726109087" in result
    assert "监听" in result
    args = control_mock.call_args.args[0]
    assert args["platform"] == "qq_napcat"
    assert args["action"] == "enable_collect_only"
    assert args["target"] == "group:726109087"
    assert "daily_report_enabled" not in args


@pytest.mark.asyncio
async def test_admin_dm_can_orally_stop_explicit_group_monitoring_and_allow_chat():
    runner = _make_runner(auto_background_work=True)
    runner.config.platforms[Platform.QQ_NAPCAT].extra["admin_users"] = ["179033731"]
    event = _make_event("停止QQ 群 192903718 的监听采集,允许开始聊天")

    with patch(
        "tools.messaging_control_tool.messaging_control_tool",
        return_value=json.dumps(
            {
                "success": True,
                "policy": {
                    "group_id": "192903718",
                    "mode": "default",
                    "archive_enabled": False,
                    "daily_report_enabled": False,
                },
            },
            ensure_ascii=False,
        ),
    ) as control_mock:
        result = await runner._handle_message(event)

    runner._run_agent.assert_not_awaited()
    assert "已停止 QQ 群 192903718 的监听采集" in result
    assert "监听采集模式" not in result
    args = control_mock.call_args.args[0]
    assert args["platform"] == "qq_napcat"
    assert args["action"] == "resume_chat"
    assert args["target"] == "group:192903718"


@pytest.mark.asyncio
async def test_admin_group_can_orally_resume_current_group_chat_without_stop_listen_phrase():
    runner = _make_runner(auto_background_work=True)
    runner.config.platforms[Platform.QQ_NAPCAT].extra["admin_users"] = ["179033731"]
    event = _make_event(
        "这个群恢复聊天",
        chat_type="group",
        chat_id="192903718",
    )

    with patch(
        "tools.messaging_control_tool.messaging_control_tool",
        return_value=json.dumps(
            {
                "success": True,
                "policy": {
                    "group_id": "192903718",
                    "mode": "default",
                    "archive_enabled": False,
                    "daily_report_enabled": False,
                },
            },
            ensure_ascii=False,
        ),
    ) as control_mock:
        result = await runner._handle_message(event)

    runner._run_agent.assert_not_awaited()
    assert "已停止 QQ 群 192903718 的监听采集" in result
    args = control_mock.call_args.args[0]
    assert args == {
        "platform": "qq_napcat",
        "action": "resume_chat",
        "target": "group:192903718",
    }


@pytest.mark.asyncio
async def test_admin_dm_stop_monitoring_request_does_not_misparse_bot_pronoun_as_worker():
    runner = _make_runner(auto_background_work=True)
    runner.config.platforms[Platform.QQ_NAPCAT].extra["admin_users"] = ["179033731"]
    event = _make_event("我让你停止QQ 群 192903718 的监听采集,允许开始聊天")

    with patch(
        "tools.messaging_control_tool.messaging_control_tool",
        return_value=json.dumps(
            {
                "success": True,
                "policy": {
                    "group_id": "192903718",
                    "mode": "default",
                    "archive_enabled": False,
                    "daily_report_enabled": False,
                },
            },
            ensure_ascii=False,
        ),
    ) as control_mock:
        result = await runner._handle_message(event)

    runner._run_agent.assert_not_awaited()
    assert "已停止 QQ 群 192903718 的监听采集" in result
    args = control_mock.call_args.args[0]
    assert args["platform"] == "qq_napcat"
    assert args["action"] == "resume_chat"
    assert args["target"] == "group:192903718"
    assert "worker_name" not in args


@pytest.mark.asyncio
async def test_admin_dm_can_remove_group_from_monitoring_after_bot_was_kicked():
    runner = _make_runner(auto_background_work=True)
    runner.config.platforms[Platform.QQ_NAPCAT].extra["admin_users"] = ["179033731"]
    event = _make_event("726109087群你已经被踢出了 去掉")

    with patch(
        "tools.messaging_control_tool.messaging_control_tool",
        return_value=json.dumps(
            {
                "success": True,
                "policy": {
                    "group_id": "726109087",
                    "mode": "default",
                    "archive_enabled": False,
                    "daily_report_enabled": False,
                },
            },
            ensure_ascii=False,
        ),
    ) as control_mock:
        result = await runner._handle_message(event)

    runner._run_agent.assert_not_awaited()
    assert "726109087" in result
    assert "已停止" in result
    args = control_mock.call_args.args[0]
    assert args["platform"] == "qq_napcat"
    assert args["action"] == "disable_group"
    assert args["target"] == "group:726109087"


@pytest.mark.asyncio
async def test_admin_dm_can_orally_send_message_to_explicit_group_same_turn():
    runner = _make_runner(auto_background_work=True)
    runner.config.platforms[Platform.QQ_NAPCAT].extra["admin_users"] = ["179033731"]
    event = _make_event("往 QQ 群 192903718 发：绿帽哥！")

    with patch(
        "tools.send_message_tool.send_message_tool",
        return_value=json.dumps(
            {
                "success": True,
                "platform": "qq_napcat",
                "chat_id": "192903718",
            },
            ensure_ascii=False,
        ),
    ) as send_mock:
        result = await runner._handle_message(event)

    runner._run_agent.assert_not_awaited()
    send_mock.assert_called_once()
    args = send_mock.call_args.args[0]
    assert args["target"] == "qq_napcat:group:192903718"
    assert args["message"] == "绿帽哥！"
    assert "192903718" in result
    assert "绿帽哥！" in result


@pytest.mark.asyncio
async def test_admin_dm_can_orally_send_message_to_group_from_followup_confirmation():
    runner = _make_runner(auto_background_work=True)
    runner.config.platforms[Platform.QQ_NAPCAT].extra["admin_users"] = ["179033731"]
    runner.session_store.load_transcript.return_value = [
        {"role": "user", "content": "你现在能在 群 192903718 发送一句话吗"},
        {"role": "assistant", "content": "可以，把要发的内容直接发我。"},
    ]
    event = _make_event("绿帽哥!\n\n发这句")

    with patch(
        "tools.send_message_tool.send_message_tool",
        return_value=json.dumps(
            {
                "success": True,
                "platform": "qq_napcat",
                "chat_id": "192903718",
            },
            ensure_ascii=False,
        ),
    ) as send_mock:
        result = await runner._handle_message(event)

    runner._run_agent.assert_not_awaited()
    send_mock.assert_called_once()
    args = send_mock.call_args.args[0]
    assert args["target"] == "qq_napcat:group:192903718"
    assert args["message"] == "绿帽哥!"
    assert "192903718" in result
    assert "绿帽哥!" in result


@pytest.mark.asyncio
async def test_admin_weixin_can_orally_send_message_to_explicit_group_same_turn():
    runner = _make_runner(auto_background_work=True)
    runner.config.platforms[Platform.WEIXIN] = PlatformConfig(
        enabled=True,
        token="***",
        extra={"admin_users": ["179033731"]},
    )
    runner.adapters[Platform.WEIXIN] = MagicMock()
    event = MessageEvent(
        text="往 微信群 project@chatroom 发：开会了",
        source=SessionSource(
            platform=Platform.WEIXIN,
            user_id="179033731",
            user_name="發發發",
            chat_id="wxid_admin",
            chat_type="dm",
        ),
        message_id="m1",
        message_type=MessageType.TEXT,
    )

    with patch(
        "tools.send_message_tool.send_message_tool",
        return_value=json.dumps(
            {
                "success": True,
                "platform": "weixin",
                "chat_id": "project@chatroom",
            },
            ensure_ascii=False,
        ),
    ) as send_mock:
        result = await runner._handle_message(event)

    runner._run_agent.assert_not_awaited()
    send_mock.assert_called_once()
    args = send_mock.call_args.args[0]
    assert args["target"] == "weixin:project@chatroom"
    assert args["message"] == "开会了"
    assert "project@chatroom" in result
    assert "开会了" in result


@pytest.mark.asyncio
async def test_admin_weixin_can_orally_send_message_from_followup_confirmation():
    runner = _make_runner(auto_background_work=True)
    runner.config.platforms[Platform.WEIXIN] = PlatformConfig(
        enabled=True,
        token="***",
        extra={"admin_users": ["179033731"]},
    )
    runner.adapters[Platform.WEIXIN] = MagicMock()
    runner.session_store.load_transcript.return_value = [
        {"role": "user", "content": "你现在能在 微信群 project@chatroom 发送一句话吗"},
        {"role": "assistant", "content": "可以，把要发的内容直接发我。"},
    ]
    event = MessageEvent(
        text="收到\n发这句",
        source=SessionSource(
            platform=Platform.WEIXIN,
            user_id="179033731",
            user_name="發發發",
            chat_id="wxid_admin",
            chat_type="dm",
        ),
        message_id="m1",
        message_type=MessageType.TEXT,
    )

    with patch(
        "tools.send_message_tool.send_message_tool",
        return_value=json.dumps(
            {
                "success": True,
                "platform": "weixin",
                "chat_id": "project@chatroom",
            },
            ensure_ascii=False,
        ),
    ) as send_mock:
        result = await runner._handle_message(event)

    runner._run_agent.assert_not_awaited()
    send_mock.assert_called_once()
    args = send_mock.call_args.args[0]
    assert args["target"] == "weixin:project@chatroom"
    assert args["message"] == "收到"
    assert "project@chatroom" in result
    assert "收到" in result


@pytest.mark.asyncio
async def test_admin_dm_can_orally_hire_intel_worker():
    runner = _make_runner(auto_background_work=True)
    runner.config.platforms[Platform.QQ_NAPCAT].extra["admin_users"] = ["179033731"]
    event = _make_event("招一个情报员钢镚，去 726109087 这个群刺探情报，每天私聊向我汇报。")

    with patch(
        "tools.messaging_control_tool.messaging_control_tool",
        return_value=json.dumps(
            {
                "success": True,
                "worker": {
                    "worker_name": "钢镚",
                    "status": "awaiting_group_approval",
                    "target_group_ref": "group:726109087",
                    "daily_report_target": "qq_napcat:dm:179033731",
                    "manual_report_target": "qq_napcat:dm:179033731",
                },
            },
            ensure_ascii=False,
        ),
    ) as control_mock:
        result = await runner._handle_message(event)

    runner._run_agent.assert_not_awaited()
    assert "钢镚" in result
    assert "726109087" in result
    args = control_mock.call_args.args[0]
    assert args["platform"] == "qq_napcat"
    assert args["action"] == "hire_worker"
    assert args["worker_name"] == "钢镚"
    assert args["target_group"] == "group:726109087"
    assert args["daily_report_target"] == "current_user_dm"
    assert args["manual_report_target"] == "current_user_dm"
    assert "刺探情报" in args["objective"]


@pytest.mark.asyncio
async def test_admin_dm_can_orally_pause_intel_worker():
    runner = _make_runner(auto_background_work=True)
    runner.config.platforms[Platform.QQ_NAPCAT].extra["admin_users"] = ["179033731"]
    event = _make_event("让情报员钢镚暂停任务，先别监听了。")

    with patch(
        "tools.messaging_control_tool.messaging_control_tool",
        return_value=json.dumps(
            {
                "success": True,
                "worker": {
                    "worker_name": "钢镚",
                    "status": "paused",
                },
            },
            ensure_ascii=False,
        ),
    ) as control_mock:
        result = await runner._handle_message(event)

    runner._run_agent.assert_not_awaited()
    assert "钢镚" in result
    assert "暂停" in result
    args = control_mock.call_args.args[0]
    assert args == {
        "platform": "qq_napcat",
        "action": "pause_worker",
        "worker_name": "钢镚",
    }


@pytest.mark.asyncio
async def test_admin_dm_can_orally_resume_intel_worker():
    runner = _make_runner(auto_background_work=True)
    runner.config.platforms[Platform.QQ_NAPCAT].extra["admin_users"] = ["179033731"]
    event = _make_event("让情报员钢镚恢复任务，继续监听。")

    with patch(
        "tools.messaging_control_tool.messaging_control_tool",
        return_value=json.dumps(
            {
                "success": True,
                "worker": {
                    "worker_name": "钢镚",
                    "status": "active_collecting",
                },
            },
            ensure_ascii=False,
        ),
    ) as control_mock:
        result = await runner._handle_message(event)

    runner._run_agent.assert_not_awaited()
    assert "钢镚" in result
    assert "恢复" in result or "继续" in result
    args = control_mock.call_args.args[0]
    assert args == {
        "platform": "qq_napcat",
        "action": "resume_worker",
        "worker_name": "钢镚",
    }


@pytest.mark.asyncio
async def test_admin_dm_can_orally_request_intel_report_now():
    runner = _make_runner(auto_background_work=True)
    runner.config.platforms[Platform.QQ_NAPCAT].extra["admin_users"] = ["179033731"]
    event = _make_event("让情报员钢镚现在汇报，私聊发我。")

    with patch(
        "tools.messaging_control_tool.messaging_control_tool",
        return_value=json.dumps(
            {
                "success": True,
                "worker": {
                    "worker_name": "钢镚",
                    "status": "active_collecting",
                },
                "delivery": {"target": "qq_napcat:dm:179033731"},
            },
            ensure_ascii=False,
        ),
    ) as control_mock:
        result = await runner._handle_message(event)

    runner._run_agent.assert_not_awaited()
    assert "钢镚" in result
    assert "汇报" in result
    args = control_mock.call_args.args[0]
    assert args["platform"] == "qq_napcat"
    assert args["action"] == "run_report_now"
    assert args["worker_name"] == "钢镚"
    assert args["manual_report_target"] == "current_user_dm"


@pytest.mark.asyncio
async def test_admin_dm_can_orally_query_intel_worker_status():
    runner = _make_runner(auto_background_work=True)
    runner.config.platforms[Platform.QQ_NAPCAT].extra["admin_users"] = ["179033731"]
    event = _make_event("看看情报员钢镚现在什么状态。")

    with patch(
        "tools.messaging_control_tool.messaging_control_tool",
        return_value=json.dumps(
            {
                "success": True,
                "worker": {
                    "worker_name": "钢镚",
                    "status": "active_collecting",
                    "target_group_id": "726109087",
                    "target_group_name": "外部情报群",
                    "objective": "刺探情报",
                    "last_error": None,
                },
            },
            ensure_ascii=False,
        ),
    ) as control_mock:
        result = await runner._handle_message(event)

    runner._run_agent.assert_not_awaited()
    assert "钢镚" in result
    assert "正在潜伏采集" in result
    assert "726109087" in result
    args = control_mock.call_args.args[0]
    assert args == {
        "platform": "qq_napcat",
        "action": "get_worker",
        "worker_name": "钢镚",
    }


@pytest.mark.asyncio
async def test_admin_dm_worker_status_mentions_report_targets():
    runner = _make_runner(auto_background_work=True)
    runner.config.platforms[Platform.QQ_NAPCAT].extra["admin_users"] = ["179033731"]
    event = _make_event("看看情报员钢镚现在什么状态。")

    with patch(
        "tools.messaging_control_tool.messaging_control_tool",
        return_value=json.dumps(
            {
                "success": True,
                "worker": {
                    "worker_name": "钢镚",
                    "status": "active_collecting",
                    "target_group_id": "726109087",
                    "daily_report_enabled": True,
                    "daily_report_target": "qq_napcat:dm:179033731",
                    "manual_report_target": "qq_napcat:group:726109087",
                },
            },
            ensure_ascii=False,
        ),
    ):
        result = await runner._handle_message(event)

    runner._run_agent.assert_not_awaited()
    assert "日报目标：qq_napcat:dm:179033731" in result
    assert "立即汇报目标：qq_napcat:group:726109087" in result


@pytest.mark.asyncio
async def test_admin_dm_can_orally_list_joined_groups():
    runner = _make_runner(auto_background_work=True)
    runner.config.platforms[Platform.QQ_NAPCAT].extra["admin_users"] = ["179033731"]
    event = _make_event("把你现在加的群列一下。")

    with patch(
        "tools.messaging_control_tool.messaging_control_tool",
        return_value=json.dumps(
            {
                "success": True,
                "groups": [
                    {"group_id": "726109087", "group_name": "项目群"},
                    {"group_id": "888888", "group_name": "外部群"},
                ],
            },
            ensure_ascii=False,
        ),
    ) as control_mock:
        result = await runner._handle_message(event)

    runner._run_agent.assert_not_awaited()
    assert "726109087" in result
    assert "项目群" in result
    assert "888888" in result
    args = control_mock.call_args.args[0]
    assert args == {"platform": "qq_napcat", "action": "list_joined_groups"}


@pytest.mark.asyncio
async def test_admin_group_can_orally_query_current_group_runtime_status():
    runner = _make_runner(auto_background_work=True)
    runner.config.platforms[Platform.QQ_NAPCAT].extra["admin_users"] = ["179033731"]
    event = _make_event(
        "这个群现在谁在监听，日报开了吗？",
        chat_type="group",
        chat_id="726109087",
    )

    with patch(
        "gateway.direct_control_platform_specs.QQ_GROUP_RUNTIME_STATUS_PLATFORM_SPEC",
        new=SimpleNamespace(
            load_status_details=lambda target: {
                "platform_label": "QQ 群",
                "target_label": "726109087",
                "effective_mode": "collect_only",
                "can_reply_in_group": False,
                "archive_enabled": True,
                "daily_report_enabled": True,
                "daily_targets": ["qq_napcat:dm:179033731"],
                "manual_targets": [],
                "worker_names": ["钢镚", "二狗"],
            }
        ),
    ):
        result = await runner._handle_message(event)

    runner._run_agent.assert_not_awaited()
    assert "726109087" in result
    assert "collect_only" in result
    assert "钢镚" in result
    assert "二狗" in result
    assert "日报" in result


@pytest.mark.asyncio
async def test_background_status_shortcut_does_not_steal_explicit_group_runtime_status_query(tmp_path):
    runner = _make_runner(auto_background_work=True)
    runner.config.platforms[Platform.QQ_NAPCAT].extra["admin_users"] = ["179033731"]
    source = _make_source(chat_type="group", chat_id="726109087")
    _create_durable_background_job(
        runner,
        tmp_path,
        task_id="bg_group_conflict",
        source=source,
        prompt="继续处理线上问题",
        status="running",
        worker_name="铁柱",
    )
    event = _make_event("这个群现在什么状态", chat_type="group", chat_id="726109087")

    with patch(
        "gateway.direct_control_platform_specs.QQ_GROUP_RUNTIME_STATUS_PLATFORM_SPEC",
        new=SimpleNamespace(
            load_status_details=lambda target: {
                "platform_label": "QQ 群",
                "target_label": "726109087",
                "effective_mode": "collect_only",
                "can_reply_in_group": False,
                "archive_enabled": True,
                "daily_report_enabled": False,
                "daily_targets": [],
                "manual_targets": [],
                "worker_names": ["钢镚"],
            }
        ),
    ):
        result = await runner._handle_message(event)

    runner._run_agent.assert_not_awaited()
    assert "726109087" in result
    assert "collect_only" in result
    assert "bg_group_conflict" not in result


@pytest.mark.asyncio
async def test_admin_group_runtime_status_uses_effective_collect_only_overlay_and_report_targets():
    runner = _make_runner(auto_background_work=True)
    runner.config.platforms[Platform.QQ_NAPCAT].extra["admin_users"] = ["179033731"]
    event = _make_event("726109087 这个群现在谁在监听，日报发哪，立即汇报发哪？")

    with patch(
        "gateway.direct_control_platform_specs.QQ_GROUP_RUNTIME_STATUS_PLATFORM_SPEC",
        new=SimpleNamespace(
            load_status_details=lambda target: {
                "platform_label": "QQ 群",
                "target_label": "726109087",
                "effective_mode": "collect_only",
                "can_reply_in_group": False,
                "archive_enabled": True,
                "daily_report_enabled": True,
                "daily_targets": ["qq_napcat:dm:179033731"],
                "manual_targets": ["qq_napcat:group:726109087"],
                "worker_names": ["钢镚"],
            }
        ),
    ):
        result = await runner._handle_message(event)

    runner._run_agent.assert_not_awaited()
    assert "collect_only" in result
    assert "归档：开" in result
    assert "日报：开" in result
    assert "日报目标：qq_napcat:dm:179033731" in result
    assert "立即汇报目标：qq_napcat:group:726109087" in result
    assert "钢镚" in result


@pytest.mark.asyncio
async def test_admin_dm_group_runtime_status_query_can_infer_recent_group_target_from_history():
    runner = _make_runner(auto_background_work=True)
    runner.config.platforms[Platform.QQ_NAPCAT].extra["admin_users"] = ["179033731"]
    runner.session_store.load_transcript.return_value = [
        {"role": "user", "content": "往 QQ 群 192903718 发：绿帽哥！"},
        {"role": "assistant", "content": "已发到 QQ 群 192903718：绿帽哥！"},
    ]
    event = _make_event("你现在在群里能说话吗 不是监听模式了吗")

    with patch(
        "gateway.direct_control_platform_specs.QQ_GROUP_RUNTIME_STATUS_PLATFORM_SPEC",
        new=SimpleNamespace(
            load_status_details=lambda target: {
                "platform_label": "QQ 群",
                "target_label": "192903718",
                "effective_mode": "collect_only",
                "can_reply_in_group": False,
                "archive_enabled": True,
                "daily_report_enabled": False,
                "daily_targets": [],
                "manual_targets": [],
                "worker_names": [],
            }
        ),
    ), patch("tools.messaging_control_tool.messaging_control_tool") as control_mock:
        result = await runner._handle_message(event)

    runner._run_agent.assert_not_awaited()
    control_mock.assert_not_called()
    assert "192903718" in result
    assert "collect_only" in result
    assert "群里主动说话：不能" in result
    assert "要切群监听/日报" not in result


@pytest.mark.asyncio
async def test_admin_weixin_group_can_orally_query_current_group_runtime_status():
    runner = _make_runner(auto_background_work=True)
    runner.config.platforms[Platform.WEIXIN] = PlatformConfig(
        enabled=True,
        token="***",
        extra={"admin_users": ["179033731"]},
    )
    runner.adapters[Platform.WEIXIN] = MagicMock()
    event = MessageEvent(
        text="这个群现在谁在监听，日报开了吗？",
        source=SessionSource(
            platform=Platform.WEIXIN,
            user_id="179033731",
            user_name="發發發",
            chat_id="project@chatroom",
            chat_type="group",
        ),
        message_id="m1",
        message_type=MessageType.TEXT,
    )

    with patch(
        "gateway.direct_control_platform_specs.WEIXIN_GROUP_RUNTIME_STATUS_PLATFORM_SPEC",
        new=SimpleNamespace(
            load_status_details=lambda target: {
                "platform_label": "微信群",
                "target_label": "project@chatroom",
                "effective_mode": "collect_only",
                "can_reply_in_group": False,
                "archive_enabled": True,
                "daily_report_enabled": True,
                "daily_targets": ["weixin:wxid_admin"],
                "manual_targets": ["weixin:project@chatroom"],
                "worker_names": [],
            }
        ),
    ):
        result = await runner._handle_message(event)

    runner._run_agent.assert_not_awaited()
    assert "project@chatroom" in result
    assert "collect_only" in result
    assert "日报目标：weixin:wxid_admin" in result
    assert "立即汇报目标：weixin:project@chatroom" in result


@pytest.mark.asyncio
async def test_admin_weixin_group_can_orally_enable_collect_only():
    runner = _make_runner(auto_background_work=True)
    runner.config.platforms[Platform.WEIXIN] = PlatformConfig(
        enabled=True,
        token="***",
        extra={"admin_users": ["179033731"]},
    )
    runner.adapters[Platform.WEIXIN] = MagicMock()
    event = MessageEvent(
        text="这个群切到监听采集，日报发我私聊",
        source=SessionSource(
            platform=Platform.WEIXIN,
            user_id="179033731",
            user_name="發發發",
            chat_id="project@chatroom",
            chat_type="group",
        ),
        message_id="m1",
        message_type=MessageType.TEXT,
    )

    with patch(
        "tools.messaging_control_tool.messaging_control_tool",
        return_value=json.dumps(
            {
                "success": True,
                "policy": {
                    "chat_id": "project@chatroom",
                    "mode": "collect_only",
                    "daily_report_enabled": True,
                    "daily_report_target": "weixin:wxid_admin",
                    "manual_report_target": "weixin:wxid_admin",
                },
            },
            ensure_ascii=False,
        ),
    ) as control_mock:
        result = await runner._handle_message(event)

    runner._run_agent.assert_not_awaited()
    assert "project@chatroom" in result
    assert "监听采集模式" in result
    assert "日报已开启" in result
    args = control_mock.call_args.args[0]
    assert args["platform"] == "weixin"
    assert args["action"] == "collect_only"
    assert args["target"] == "project@chatroom"
    assert args["daily_report_target"] == "current_user_dm"


@pytest.mark.asyncio
async def test_admin_group_policy_reply_mentions_manual_report_target():
    runner = _make_runner(auto_background_work=True)
    runner.config.platforms[Platform.QQ_NAPCAT].extra["admin_users"] = ["179033731"]
    event = _make_event(
        "这个群只监听，不要走大模型，每天给我日报。",
        chat_type="group",
        chat_id="726109087",
    )

    with patch(
        "tools.messaging_control_tool.messaging_control_tool",
        return_value=json.dumps(
            {
                "success": True,
                "policy": {
                    "group_id": "726109087",
                    "mode": "collect_only",
                    "archive_enabled": True,
                    "daily_report_enabled": True,
                    "daily_report_target": "qq_napcat:dm:179033731",
                    "manual_report_target": "qq_napcat:group:726109087",
                },
            },
            ensure_ascii=False,
        ),
    ):
        result = await runner._handle_message(event)

    runner._run_agent.assert_not_awaited()
    assert "日报已开启" in result
    assert "立即汇报发到 qq_napcat:group:726109087" in result


@pytest.mark.asyncio
async def test_oral_background_status_query_returns_latest_job_state(tmp_path):
    runner = _make_runner(auto_background_work=True)
    source = _make_source()
    _create_durable_background_job(
        runner,
        tmp_path,
        task_id="bg_123",
        source=source,
        prompt="继续把线上部署问题排查并修复",
        status="running",
        worker_name="铁柱",
    )

    result = await runner._handle_message(_make_event("前面那个后台任务还在做吗？"))

    runner._run_agent.assert_not_awaited()
    assert "bg_123" in result
    assert "进行中" in result
    assert "铁柱" in result


@pytest.mark.asyncio
async def test_busy_session_still_answers_oral_background_status_query(tmp_path):
    runner = _make_runner(auto_background_work=True)
    source = _make_source()
    session_key = runner._session_key_for_source(source)
    runner._running_agents[session_key] = MagicMock()
    runner.adapters[Platform.QQ_NAPCAT]._busy_followup_ack.return_value = "busy ack"
    _create_durable_background_job(
        runner,
        tmp_path,
        task_id="bg_busy_1",
        source=source,
        prompt="继续处理线上问题",
        status="running",
        worker_name="铁柱",
    )

    result = await runner._handle_message(_make_event("前面那个后台任务还在做吗？"))

    runner._run_agent.assert_not_awaited()
    assert result is not None
    assert "bg_busy_1" in result
    assert "进行中" in result


@pytest.mark.asyncio
async def test_busy_session_plain_hai_zaima_prefers_background_status_shortcut(tmp_path):
    runner = _make_runner(auto_background_work=True)
    source = _make_source(chat_type="group", chat_id="726109087")
    session_key = runner._session_key_for_source(source)
    runner._running_agents[session_key] = MagicMock()
    _create_durable_background_job(
        runner,
        tmp_path,
        task_id="bg_busy_2",
        source=source,
        prompt="继续处理线上问题",
        status="running",
        worker_name="铁柱",
    )

    result = await runner._handle_message(
        _make_event("@马嘎 还在吗", chat_type="group", chat_id="726109087")
    )

    runner._run_agent.assert_not_awaited()
    assert result is not None
    assert "bg_busy_2" in result
    assert "进行中" in result


@pytest.mark.asyncio
async def test_busy_session_plain_background_task_ne_stays_on_shortcut_path(tmp_path):
    runner = _make_runner(auto_background_work=True)
    source = _make_source(chat_type="group", chat_id="726109087")
    session_key = runner._session_key_for_source(source)
    runner._running_agents[session_key] = MagicMock()
    _create_durable_background_job(
        runner,
        tmp_path,
        task_id="bg_busy_3",
        source=source,
        prompt="继续处理线上问题",
        status="running",
        worker_name="铁柱",
    )

    result = await runner._handle_message(
        _make_event("@马嘎 前面那个后台任务呢", chat_type="group", chat_id="726109087")
    )

    runner._run_agent.assert_not_awaited()
    assert result is not None
    assert "bg_busy_3" in result
    assert "进行中" in result
    assert "排队" not in result


@pytest.mark.asyncio
async def test_busy_session_still_answers_oral_foreground_runtime_status_query():
    runner = _make_runner(auto_background_work=True)
    source = _make_source()
    session_key = runner._session_key_for_source(source)
    running_agent = MagicMock()
    running_agent.get_activity_summary.return_value = {
        "api_call_count": 4,
        "max_iterations": 60,
        "current_tool": "delegate_task",
        "last_activity_desc": "running: delegate_task",
    }
    runner._running_agents[session_key] = running_agent

    result = await runner._handle_message(_make_event("你现在忙什么？"))

    runner._run_agent.assert_not_awaited()
    assert result is not None
    assert "前台" in result
    assert "delegate_task" in result


@pytest.mark.asyncio
async def test_busy_session_still_executes_oral_intel_control():
    runner = _make_runner(auto_background_work=True)
    runner.config.platforms[Platform.QQ_NAPCAT].extra["admin_users"] = ["179033731"]
    source = _make_source()
    session_key = runner._session_key_for_source(source)
    runner._running_agents[session_key] = MagicMock()
    runner.adapters[Platform.QQ_NAPCAT]._busy_followup_ack.return_value = "busy ack"
    event = _make_event("让情报员钢镚暂停任务，先别监听了。")

    with patch(
        "tools.messaging_control_tool.messaging_control_tool",
        return_value=json.dumps(
            {
                "success": True,
                "worker": {"worker_name": "钢镚", "status": "paused"},
            },
            ensure_ascii=False,
        ),
    ) as control_mock:
        result = await runner._handle_message(event)

    runner._run_agent.assert_not_awaited()
    assert "钢镚" in result
    assert "暂停" in result
    assert control_mock.call_args.args[0] == {
        "platform": "qq_napcat",
        "action": "pause_worker",
        "worker_name": "钢镚",
    }


@pytest.mark.asyncio
async def test_busy_session_still_executes_oral_group_policy_control():
    runner = _make_runner(auto_background_work=True)
    runner.config.platforms[Platform.QQ_NAPCAT].extra["admin_users"] = ["179033731"]
    source = _make_source(chat_type="group", chat_id="726109087")
    session_key = runner._session_key_for_source(source)
    runner._running_agents[session_key] = MagicMock()
    runner.adapters[Platform.QQ_NAPCAT]._busy_followup_ack.return_value = "busy ack"
    event = _make_event(
        "这个群只监听，不要走大模型。",
        chat_type="group",
        chat_id="726109087",
    )

    with patch(
        "tools.messaging_control_tool.messaging_control_tool",
        return_value=json.dumps(
            {
                "success": True,
                "policy": {
                    "group_id": "726109087",
                    "mode": "collect_only",
                    "archive_enabled": True,
                },
            },
            ensure_ascii=False,
        ),
    ) as control_mock:
        result = await runner._handle_message(event)

    runner._run_agent.assert_not_awaited()
    assert "726109087" in result
    assert "监听" in result
    assert control_mock.call_args.args[0]["platform"] == "qq_napcat"
    assert control_mock.call_args.args[0]["action"] == "enable_collect_only"


@pytest.mark.asyncio
async def test_busy_session_still_lists_pending_friend_requests_orally():
    runner = _make_runner(auto_background_work=True)
    runner.config.platforms[Platform.QQ_NAPCAT].extra["admin_users"] = ["179033731"]
    source = _make_source()
    session_key = runner._session_key_for_source(source)
    runner._running_agents[session_key] = MagicMock()
    event = _make_event("看看待处理的好友申请")

    with patch(
        "tools.messaging_control_tool.messaging_control_tool",
        return_value=json.dumps(
            {
                "success": True,
                "requests": [
                    {
                        "request_key": "friend:friend-flag-1",
                        "request_type": "friend",
                        "user_id": "456789",
                        "comment": "加一下",
                        "status": "pending",
                    }
                ],
            },
            ensure_ascii=False,
        ),
    ) as social_mock:
        result = await runner._handle_message(event)

    runner._run_agent.assert_not_awaited()
    assert "friend:friend-flag-1" in result
    assert "456789" in result
    assert social_mock.call_args.args[0] == {
        "platform": "qq_napcat",
        "action": "list_requests",
        "status": "pending",
        "request_type": "friend",
        "limit": 20,
    }


@pytest.mark.asyncio
async def test_busy_session_still_updates_social_policy_orally():
    runner = _make_runner(auto_background_work=True)
    runner.config.platforms[Platform.QQ_NAPCAT].extra["admin_users"] = ["179033731"]
    source = _make_source()
    session_key = runner._session_key_for_source(source)
    runner._running_agents[session_key] = MagicMock()
    event = _make_event("把自动通过好友申请打开，通知发我私聊。")

    with patch(
        "tools.messaging_control_tool.messaging_control_tool",
        return_value=json.dumps(
            {
                "success": True,
                "policy": {
                    "auto_approve_friend_requests": True,
                    "auto_approve_group_add_requests": False,
                    "auto_approve_group_invites": False,
                    "notify_target": "qq_napcat:dm:179033731",
                },
            },
            ensure_ascii=False,
        ),
    ) as social_mock:
        result = await runner._handle_message(event)

    runner._run_agent.assert_not_awaited()
    assert "好友申请" in result
    assert "开启" in result
    assert social_mock.call_args.args[0] == {
        "platform": "qq_napcat",
        "action": "set_social_policy",
        "auto_approve_friend_requests": True,
        "notify_target": "current_user_dm",
    }


@pytest.mark.asyncio
async def test_admin_group_can_orally_mute_member_via_control_plane():
    runner = _make_runner(auto_background_work=True)
    runner.config.platforms[Platform.QQ_NAPCAT].extra["admin_users"] = ["179033731"]
    event = _make_event(
        "把广告哥禁言10分钟，原因广告。",
        chat_type="group",
        chat_id="726109087",
    )

    with patch(
        "tools.messaging_control_tool.messaging_control_tool",
        return_value=json.dumps(
            {"success": True, "action": "mute_user", "target_user_id": "123456"},
            ensure_ascii=False,
        ),
    ) as control_mock:
        result = await runner._handle_message(event)

    runner._run_agent.assert_not_awaited()
    assert "禁言" in result
    args = control_mock.call_args.args[0]
    assert args["platform"] == "qq_napcat"
    assert args["action"] == "mute_user"
    assert args["target"] == "group:726109087"
    assert args["user_query"] == "广告哥"
    assert args["duration_seconds"] == 600
    assert args["reason"] == "广告"


@pytest.mark.asyncio
async def test_admin_group_can_orally_kick_member_via_control_plane():
    runner = _make_runner(auto_background_work=True)
    runner.config.platforms[Platform.QQ_NAPCAT].extra["admin_users"] = ["179033731"]
    event = _make_event(
        "把广告哥踢了，原因广告。",
        chat_type="group",
        chat_id="726109087",
    )

    with patch(
        "tools.messaging_control_tool.messaging_control_tool",
        return_value=json.dumps(
            {"success": True, "action": "kick_user", "target_user_id": "123456"},
            ensure_ascii=False,
        ),
    ) as control_mock:
        result = await runner._handle_message(event)

    runner._run_agent.assert_not_awaited()
    assert "踢" in result
    args = control_mock.call_args.args[0]
    assert args["platform"] == "qq_napcat"
    assert args["action"] == "kick_user"
    assert args["target"] == "group:726109087"
    assert args["user_query"] == "广告哥"
    assert args["reason"] == "广告"


@pytest.mark.asyncio
async def test_admin_weixin_group_moderation_returns_explicit_not_capable_reply():
    runner = _make_runner(auto_background_work=True)
    runner.config.platforms[Platform.WEIXIN] = PlatformConfig(
        enabled=True,
        token="***",
        extra={"admin_users": ["179033731"]},
    )
    runner.adapters[Platform.WEIXIN] = MagicMock()
    event = MessageEvent(
        text="把广告哥踢了，原因广告。",
        source=SessionSource(
            platform=Platform.WEIXIN,
            user_id="179033731",
            user_name="發發發",
            chat_id="project@chatroom",
            chat_type="group",
        ),
        message_id="wx-1",
        message_type=MessageType.TEXT,
    )

    with patch(
        "tools.messaging_control_tool.messaging_control_tool",
        return_value=json.dumps(
            {
                "success": False,
                "platform": "weixin",
                "action": "kick_user",
                "capability": "not_capable",
                "detail": "微信群暂不支持禁言/踢人。",
            },
            ensure_ascii=False,
        ),
    ) as control_mock:
        result = await runner._handle_message(event)

    runner._run_agent.assert_not_awaited()
    assert result == "微信群暂不支持禁言/踢人。"
    assert control_mock.call_args.args[0] == {
        "platform": "weixin",
        "action": "kick_user",
        "target": "project@chatroom",
        "user_query": "广告哥",
        "reason": "广告",
    }


@pytest.mark.asyncio
async def test_group_technical_discussion_with_implementation_word_stays_in_foreground():
    runner = _make_runner(auto_background_work=True)
    event = _make_event(
        "我的理解是解耦实现多线程",
        chat_type="group",
        chat_id="group:10001",
    )

    result = await runner._handle_message(event)

    runner._run_agent.assert_awaited_once()
    assert result == "前台回复"
    assert _list_durable_jobs(runner) == []


@pytest.mark.asyncio
async def test_group_explicit_maga_assignment_stays_foreground_without_configured_employee_heuristic():
    runner = _make_runner(auto_background_work=True)
    event = _make_event(
        "@马嘎 把 fafafa-page 打磨一下，页面太粗糙了，顺手调调样式和排版。",
        chat_type="group",
        chat_id="group:10001",
    )

    result = await runner._handle_message(event)

    runner._run_agent.assert_awaited_once()
    assert result == "前台回复"
    assert _list_durable_jobs(runner) == []


@pytest.mark.asyncio
async def test_status_command_includes_background_jobs(tmp_path):
    runner = _make_runner(auto_background_work=True)
    source = _make_source()
    _create_durable_background_job(
        runner,
        tmp_path,
        task_id="bg_123",
        source=source,
        prompt="继续处理线上问题",
        status="running",
        worker_name="铁柱",
    )

    result = await runner._handle_status_command(_make_event("/status"))

    assert "**Background Jobs:**" in result
    assert "`bg_123`" in result
    assert "running" in result
    assert "铁柱" in result


@pytest.mark.asyncio
async def test_status_command_includes_pending_approval_and_auto_vision_state(tmp_path):
    runner = _make_runner(auto_background_work=True)
    runner._background_job_store = BackgroundJobStore(db_path=tmp_path / "background_jobs.db")
    source = _make_source()
    session_key = runner._session_key_for_source(source)

    runner._background_job_store.create_job(
        task_id="bg_pending_approve",
        prompt="重启服务",
        source=source,
        session_key=session_key,
    )
    runner._background_job_store.create_approval_request(
        task_id="bg_pending_approve",
        session_key=session_key,
        source=source,
        approval_data={
            "command": "systemctl restart hermes-gateway.service",
            "description": "stop/disable system service",
            "prompt_title": "Dangerous command requires approval",
            "approver_name": "董事长",
            "allow_persistence": False,
            "pattern_key": "stop/disable system service",
            "pattern_keys": ["stop/disable system service"],
        },
    )
    runner._auto_vision_tasks["img:1"] = asyncio.get_running_loop().create_future()
    runner._auto_vision_unhealthy_until = time.time() + 30
    runner._auto_vision_unhealthy_reason = "provider_error"

    result = await runner._handle_status_command(_make_event("/status"))

    assert "Pending Approvals" in result
    assert "1" in result
    assert "Auto Vision" in result
    assert "warming" in result or "cooldown" in result


@pytest.mark.asyncio
async def test_status_command_includes_runtime_model_and_collect_only_monitoring_state(tmp_path):
    runner = _make_runner(auto_background_work=True)
    runner._background_job_store = BackgroundJobStore(db_path=tmp_path / "background_jobs.db")
    runner._effective_model = "gpt-5.4-mini"
    runner._effective_provider = "openrouter"
    runner._fallback_model = [{"provider": "openrouter", "model": "gpt-5.4-mini"}]
    source = _make_source(chat_type="group", chat_id="726109087")
    session_key = runner._session_key_for_source(source)
    _create_durable_background_job(
        runner,
        tmp_path,
        task_id="bg_model_1",
        source=source,
        prompt="继续处理线上问题",
        status="running",
        worker_name="铁柱",
        session_key=session_key,
    )
    runner._background_job_store.create_approval_request(
        task_id="bg_model_1",
        session_key=session_key,
        source=source,
        approval_data={
            "command": "systemctl restart hermes-gateway.service",
            "description": "stop/disable system service",
            "prompt_title": "Dangerous command requires approval",
            "approver_name": "董事长",
            "allow_persistence": False,
            "pattern_key": "stop/disable system service",
            "pattern_keys": ["stop/disable system service"],
        },
    )

    with patch(
        "gateway.run.build_group_monitoring_runtime_platform_specs",
        return_value=[
            GroupMonitoringRuntimePlatformSpec(
                platform="qq_napcat",
                load_summary=lambda: {
                    "active_worker_count": 1,
                    "groups": [
                        {
                            "platform": "qq_napcat",
                            "platform_label": "QQ 群",
                            "group_id": "726109087",
                            "chat_id": "726109087",
                            "group_name": "项目群",
                            "effective_mode": "collect_only",
                            "worker_names": ["钢镚"],
                        }
                    ],
                },
            )
        ],
    ), patch(
        "gateway.run._resolve_gateway_model",
        return_value="gpt-5.4",
    ), patch(
        "gateway.run._resolve_runtime_agent_kwargs",
        return_value={"provider": "custom"},
    ):
        result = await runner._handle_status_command(
            _make_event("/status", chat_type="group", chat_id="726109087")
        )

    assert "Model" in result
    assert "gpt-5.4-mini" in result
    assert "openrouter" in result
    assert "fallback" in result.lower()
    assert "Pending Approvals" in result
    assert "Group Monitoring" in result
    assert "QQ 群 · 项目群 (726109087)" in result
    assert "collect_only" in result
    assert "钢镚" in result


@pytest.mark.asyncio
async def test_status_command_includes_foreground_activity_detail():
    runner = _make_runner(auto_background_work=True)
    source = _make_source()
    session_key = runner._session_key_for_source(source)
    running_agent = MagicMock()
    running_agent.get_activity_summary.return_value = {
        "api_call_count": 4,
        "max_iterations": 60,
        "current_tool": "delegate_task",
        "last_activity_desc": "running: delegate_task",
    }
    runner._running_agents[session_key] = running_agent
    runner._running_agents_ts[session_key] = time.time() - 12

    result = await runner._handle_status_command(_make_event("/status"))

    assert "Foreground" in result
    assert "delegate_task" in result
    assert "4/60" in result


def test_runtime_status_summary_includes_foreground_background_vision_and_archive(tmp_path):
    runner = _make_runner(auto_background_work=True)
    runner._background_job_store = BackgroundJobStore(db_path=tmp_path / "background_jobs.db")
    source = _make_source(chat_type="group", chat_id="726109087")
    session_key = runner._session_key_for_source(source)
    runner._effective_model = "gpt-5.4-fallback"
    runner._effective_provider = "custom-fallback"
    runner._pending_approvals[session_key] = {"command": "systemctl restart hermes-gateway.service"}

    running_agent = MagicMock()
    running_agent.get_activity_summary.return_value = {
        "api_call_count": 3,
        "max_iterations": 60,
        "current_tool": "delegate_task",
        "last_activity_desc": "running: delegate_task",
    }
    runner._running_agents[session_key] = running_agent
    runner._running_agents_ts[session_key] = time.time() - 12
    _create_durable_background_job(
        runner,
        tmp_path,
        task_id="bg_runtime_1",
        source=source,
        prompt="继续处理线上问题",
        status="running",
        worker_name="铁柱",
        session_key=session_key,
    )
    _create_durable_background_job(
        runner,
        tmp_path,
        task_id="bg_runtime_2",
        source=source,
        prompt="整理群监听日报",
        status="queued",
        session_key=session_key,
    )
    pending_task = MagicMock()
    pending_task.done.return_value = False
    runner._auto_vision_tasks = {"vision:1": pending_task}
    runner._auto_vision_cache = {
        "cache:1": {
            "status": "success",
            "updated_at": time.time(),
            "expires_at": time.time() + 60,
        }
    }
    runner._auto_vision_unhealthy_until = time.time() + 15
    runner._auto_vision_unhealthy_reason = "timeout"

    with patch(
        "gateway.run.build_group_archive_runtime_platform_specs",
        return_value=[
            GroupArchiveRuntimePlatformSpec(
                platform="qq_napcat",
                load_runtime_stats=lambda: {
                    "raw_message_count": 42,
                    "raw_group_count": 2,
                    "due_rollup_count": 1,
                    "report_count": 5,
                },
            )
        ],
    ), patch(
        "gateway.run._resolve_gateway_model",
        return_value="gpt-5.4",
    ), patch(
        "gateway.run._resolve_runtime_agent_kwargs",
        return_value={"provider": "custom"},
    ), patch(
        "gateway.run.build_group_monitoring_runtime_platform_specs",
        return_value=[
            GroupMonitoringRuntimePlatformSpec(
                platform="qq_napcat",
                load_summary=lambda: {
                    "active_worker_count": 1,
                    "groups": [
                        {
                            "platform": "qq_napcat",
                            "platform_label": "QQ 群",
                            "group_id": "726109087",
                            "chat_id": "726109087",
                            "group_name": "726109087",
                            "worker_names": ["钢镚"],
                        }
                    ],
                },
            )
        ],
    ):
        summary = build_runtime_status_summary(runner)

    assert summary["active_sessions_count"] == 1
    assert summary["active_sessions"][0]["current_tool"] == "delegate_task"
    assert summary["background_jobs"]["active_count"] == 2
    assert summary["background_jobs"]["counts"]["running"] == 1
    assert summary["approvals"]["pending_count"] == 1
    assert summary["auto_vision"]["state"] == "cooldown"
    assert summary["auto_vision"]["inflight_count"] == 1
    assert summary["group_archive"]["platforms"]["qq_napcat"]["raw_message_count"] == 42
    assert summary["model"]["configured_model"] == "gpt-5.4"
    assert summary["model"]["active_model"] == "gpt-5.4-fallback"
    assert summary["model"]["fallback_active"] is True
    assert summary["group_monitoring"]["active_collect_only_groups"] == 1
    assert summary["group_monitoring"]["active_worker_count"] == 1


def test_runtime_status_summary_includes_model_fallback_approvals_and_collect_only_monitoring(tmp_path):
    runner = _make_runner(auto_background_work=True)
    runner._background_job_store = BackgroundJobStore(db_path=tmp_path / "background_jobs.db")
    source = _make_source(chat_type="group", chat_id="726109087")
    session_key = runner._session_key_for_source(source)
    runner._effective_model = "gpt-5.4-mini"
    runner._effective_provider = "openrouter"
    runner._fallback_model = [{"provider": "openrouter", "model": "gpt-5.4-mini"}]
    _create_durable_background_job(
        runner,
        tmp_path,
        task_id="bg_runtime_extra",
        source=source,
        prompt="继续处理线上问题",
        status="running",
        worker_name="铁柱",
        session_key=session_key,
    )
    runner._background_job_store.update_job_launcher(
        "bg_runtime_extra",
        {"launcher_type": "subprocess", "launcher_pid": 4321},
    )
    runner._background_job_store.create_approval_request(
        task_id="bg_runtime_extra",
        session_key=session_key,
        source=source,
        approval_data={
            "command": "systemctl restart hermes-gateway.service",
            "description": "stop/disable system service",
            "prompt_title": "Dangerous command requires approval",
            "approver_name": "董事长",
            "allow_persistence": False,
            "pattern_key": "stop/disable system service",
            "pattern_keys": ["stop/disable system service"],
        },
    )

    with patch(
        "gateway.run.build_group_monitoring_runtime_platform_specs",
        return_value=[
            GroupMonitoringRuntimePlatformSpec(
                platform="qq_napcat",
                load_summary=lambda: {
                    "active_worker_count": 1,
                    "groups": [
                        {
                            "platform": "qq_napcat",
                            "platform_label": "QQ 群",
                            "group_id": "726109087",
                            "chat_id": "726109087",
                            "group_name": "项目群",
                            "worker_names": ["钢镚"],
                        }
                    ],
                },
            )
        ],
    ), patch(
        "gateway.run._resolve_gateway_model",
        return_value="gpt-5.4",
    ), patch(
        "gateway.run._resolve_runtime_agent_kwargs",
        return_value={"provider": "custom"},
    ):
        summary = build_runtime_status_summary(runner)

    assert summary["model"]["configured_model"] == "gpt-5.4"
    assert summary["model"]["active_model"] == "gpt-5.4-mini"
    assert summary["model"]["active_provider"] == "openrouter"
    assert summary["model"]["fallback_active"] is True
    assert summary["approvals"]["pending_count"] == 1
    assert summary["group_monitoring"]["active_collect_only_groups"] == 1
    assert summary["group_monitoring"]["groups"][0]["group_id"] == "726109087"
    assert summary["group_monitoring"]["groups"][0]["worker_names"] == ["钢镚"]


@pytest.mark.asyncio
async def test_oral_background_status_query_mentions_pending_approval(tmp_path):
    runner = _make_runner(auto_background_work=True)
    runner._background_job_store = BackgroundJobStore(db_path=tmp_path / "background_jobs.db")
    source = _make_source()
    session_key = runner._session_key_for_source(source)
    runner._background_job_store.create_job(
        task_id="bg_waiting_approve",
        prompt="危险操作",
        source=source,
        session_key=session_key,
    )
    runner._background_job_store.create_approval_request(
        task_id="bg_waiting_approve",
        session_key=session_key,
        source=source,
        approval_data={
            "command": "rm -rf /tmp/demo",
            "description": "recursive delete",
            "prompt_title": "Dangerous command requires approval",
            "approver_name": "董事长",
            "allow_persistence": False,
            "pattern_key": "recursive delete",
            "pattern_keys": ["recursive delete"],
        },
    )

    result = try_handle_background_job_status_shortcut(runner, _make_event("前面那个任务还在做吗"))

    assert result is not None
    assert "授权" in result or "审批" in result


@pytest.mark.asyncio
async def test_watchdog_recovers_stale_subprocess_job(tmp_path, monkeypatch):
    runner = _make_runner(auto_background_work=True)
    runner._background_job_store = BackgroundJobStore(db_path=tmp_path / "background_jobs.db")
    source = _make_source()
    session_key = runner._session_key_for_source(source)
    runner._background_job_store.create_job(
        task_id="bg_stale_job",
        prompt="长任务",
        source=source,
        session_key=session_key,
    )
    runner._background_job_store.mark_job_running("bg_stale_job")
    runner._background_job_store.update_job_launcher(
        "bg_stale_job",
        {"launcher_type": "subprocess", "launcher_pid": 999999},
    )
    monkeypatch.setattr("gateway.background_jobs.os.kill", lambda pid, sig: (_ for _ in ()).throw(ProcessLookupError()))

    recovered = await recover_stale_background_jobs_once(
        runner,
        queued_grace_seconds=30,
        heartbeat_stale_seconds=30,
        now_ts=time.time() + 600,
    )

    assert [item["task_id"] for item in recovered] == ["bg_stale_job"]
    job = runner._background_job_store.get_job("bg_stale_job")
    assert job is not None
    assert job["status"] == "failed"


@pytest.mark.asyncio
async def test_stop_command_requests_external_background_job_stop(tmp_path):
    runner = _make_runner(auto_background_work=True)
    source = _make_source()
    session_key = runner._session_key_for_source(source)
    _create_durable_background_job(
        runner,
        tmp_path,
        task_id="bg_123",
        source=source,
        prompt="继续排查线上问题",
        status="running",
        session_key=session_key,
    )
    runner._stop_background_worker = MagicMock(return_value=True)

    result = await runner._handle_stop_command(_make_event("/stop bg_123"))

    runner._stop_background_worker.assert_called_once()
    assert "bg_123" in result
    job = runner._background_job_store.get_job("bg_123")
    assert job is not None
    assert job["status"] == "cancelled"


def test_background_jobs_are_scoped_to_session_not_entire_group_chat(tmp_path):
    runner = _make_runner(auto_background_work=True)
    source_a = _make_source(chat_type="group", user_id="179033731", chat_id="999")
    source_b = _make_source(chat_type="group", user_id="888888", chat_id="999")

    session_key_a = runner._session_key_for_source(source_a)
    _create_durable_background_job(
        runner,
        tmp_path,
        task_id="bg_a",
        source=source_a,
        prompt="继续处理线上问题",
        status="running",
        session_key=session_key_a,
    )

    jobs_a = runner._background_jobs_for_source(source_a)
    jobs_b = runner._background_jobs_for_source(source_b)

    assert [job["task_id"] for job in jobs_a] == ["bg_a"]
    assert jobs_b == []


def test_platform_override_wins_for_auto_background_work():
    config = GatewayConfig.from_dict(
        {
            "auto_background_work": True,
            "platforms": {
                "qq_napcat": {
                    "enabled": True,
                    "extra": {"auto_background_work": False},
                }
            },
        }
    )

    assert config.get_auto_background_work() is True
    assert config.get_auto_background_work(Platform.QQ_NAPCAT) is False


def test_load_gateway_config_reads_auto_background_work(monkeypatch, tmp_path):
    monkeypatch.setattr("gateway.config.get_hermes_home", lambda: tmp_path)
    (tmp_path / "config.yaml").write_text(
        "gateway:\n  auto_background_work: true\n",
        encoding="utf-8",
    )

    config = load_gateway_config()

    assert config.auto_background_work is True
