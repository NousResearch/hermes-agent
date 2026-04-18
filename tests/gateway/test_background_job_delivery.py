"""Tests for gateway durable background job delivery and approval polling."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.background_delivery_service import deliver_background_job_updates_once
from gateway.background_worker import run_background_job
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source(
    *,
    chat_type: str = "dm",
    chat_id: str = "179033731",
    user_id: str = "179033731",
) -> SessionSource:
    return SessionSource(
        platform=Platform.QQ_NAPCAT,
        user_id=user_id,
        user_name="發發發",
        chat_id=chat_id,
        chat_type=chat_type,
    )


def _make_runner(tmp_path: Path):
    from gateway.background_jobs import BackgroundJobStore
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.QQ_NAPCAT: PlatformConfig(
                enabled=True,
                token="***",
                extra={"admin_users": ["179033731"]},
            )
        }
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    adapter.extract_media = MagicMock(side_effect=lambda text: ([], text))
    adapter.extract_images = MagicMock(side_effect=lambda text: ([], text))
    runner.adapters = {Platform.QQ_NAPCAT: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    source = _make_source()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key=build_session_key(source),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.QQ_NAPCAT,
        chat_type=source.chat_type,
        total_tokens=0,
    )
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._background_tasks = set()
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._effective_model = None
    runner._effective_provider = None
    runner._session_model_overrides = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = MagicMock()
    runner._update_prompt_pending = {}
    runner._failed_platforms = {}
    runner._show_reasoning = False
    runner._background_job_store = BackgroundJobStore(db_path=tmp_path / "background_jobs.db")
    runner._launch_background_worker = MagicMock(
        return_value={"launcher_type": "subprocess", "launcher_pid": 4321}
    )
    return runner


def _install_background_worker_runtime(monkeypatch, store, source):
    monkeypatch.setattr(
        "gateway.background_worker.BackgroundJobStore",
        lambda: store,
    )
    heartbeat_stop = MagicMock()
    heartbeat_thread = MagicMock()
    monkeypatch.setattr(
        "gateway.background_worker._start_job_heartbeat",
        lambda _store, _task_id: (heartbeat_stop, heartbeat_thread),
    )
    monkeypatch.setattr(
        "gateway.background_worker._build_agent_runtime",
        lambda job: {
            "source": source,
            "runtime_spec": SimpleNamespace(
                loaded_skills=[],
                missing_skills=[],
            ),
            "loaded_skills": [],
            "missing_skills": [],
        },
    )
    return heartbeat_stop, heartbeat_thread


@pytest.mark.asyncio
async def test_delivery_tick_sends_completed_job_once(tmp_path):
    runner = _make_runner(tmp_path)
    store = runner._background_job_store
    source = _make_source()

    store.create_job(
        task_id="bg_200001_abcd12",
        prompt="继续做",
        source=source,
        session_key="qq_napcat:dm:179033731",
        job_kind="auto",
        worker_name="马嘎",
    )
    store.mark_job_completed("bg_200001_abcd12", raw_response="结果已经出来了")

    await deliver_background_job_updates_once(runner)
    await deliver_background_job_updates_once(runner)

    assert runner.adapters[Platform.QQ_NAPCAT].send.await_count == 1
    content = runner.adapters[Platform.QQ_NAPCAT].send.await_args.kwargs["content"]
    assert "后台任务完成" in content
    assert "`bg_200001_abcd12`" in content
    assert "任务：" in content
    assert "结果已经出来了" in content


@pytest.mark.asyncio
async def test_delivery_tick_sends_manual_completion_with_concise_template(tmp_path):
    runner = _make_runner(tmp_path)
    store = runner._background_job_store
    source = _make_source()

    store.create_job(
        task_id="bg_200010_abcd12",
        prompt="手工补一版对外说明",
        source=source,
        session_key="qq_napcat:dm:179033731",
        job_kind="manual",
    )
    store.mark_job_completed("bg_200010_abcd12", raw_response="说明已经整理好了")

    await deliver_background_job_updates_once(runner)

    content = runner.adapters[Platform.QQ_NAPCAT].send.await_args.kwargs["content"]
    assert "后台任务完成" in content
    assert "`bg_200010_abcd12`" in content
    assert "任务：手工补一版对外说明" in content
    assert "说明已经整理好了" in content
    assert "Background task complete" not in content


@pytest.mark.asyncio
async def test_btw_job_runs_through_worker_and_delivery(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path)
    store = runner._background_job_store
    source = _make_source()
    task_id = "bg_200020_abcd12"

    store.create_job(
        task_id=task_id,
        prompt="what changed in this chat?",
        source=source,
        session_key="qq_napcat:dm:179033731",
        job_kind="btw",
        conversation_history=[{"role": "user", "content": "hello"}],
    )

    worker_calls = {}

    heartbeat_stop, heartbeat_thread = _install_background_worker_runtime(
        monkeypatch,
        store,
        source,
    )
    monkeypatch.setattr(
        "gateway.background_worker.run_gateway_btw_conversation",
        lambda **kwargs: worker_calls.update(kwargs) or {"final_response": "side answer"},
    )

    exit_code = run_background_job(task_id)

    assert exit_code == 0
    assert worker_calls["session_id"] == task_id
    assert worker_calls["source"] == source
    assert worker_calls["question"] == "what changed in this chat?"
    assert worker_calls["conversation_history"] == [{"role": "user", "content": "hello"}]
    job = store.get_job(task_id)
    assert job["status"] == "completed"
    assert job["raw_response"] == "side answer"

    await deliver_background_job_updates_once(runner)

    content = runner.adapters[Platform.QQ_NAPCAT].send.await_args.kwargs["content"]
    assert '💬 /btw: "what changed in this chat?"' in content
    assert "side answer" in content
    heartbeat_stop.set.assert_called_once()
    heartbeat_thread.join.assert_called_once_with(timeout=1.0)


def test_background_worker_marks_job_failed_for_explicit_failure_without_response(tmp_path, monkeypatch):
    from gateway.background_jobs import BackgroundJobStore

    store = BackgroundJobStore(db_path=tmp_path / "background_jobs.db")
    source = _make_source()
    task_id = "bg_200021_abcd12"

    store.create_job(
        task_id=task_id,
        prompt="继续查问题",
        source=source,
        session_key="qq_napcat:dm:179033731",
        job_kind="auto",
    )

    heartbeat_stop, heartbeat_thread = _install_background_worker_runtime(
        monkeypatch,
        store,
        source,
    )
    monkeypatch.setattr(
        "gateway.background_worker.run_gateway_background_conversation",
        lambda **kwargs: {"success": False, "detail": "worker returned malformed payload"},
    )

    exit_code = run_background_job(task_id)

    assert exit_code == 1
    job = store.get_job(task_id)
    assert job["status"] == "failed"
    assert job["error"] == "worker returned malformed payload"
    heartbeat_stop.set.assert_called_once()
    heartbeat_thread.join.assert_called_once_with(timeout=1.0)


@pytest.mark.asyncio
async def test_delivery_tick_sends_failed_job_with_concise_template(tmp_path):
    runner = _make_runner(tmp_path)
    store = runner._background_job_store
    source = _make_source()

    store.create_job(
        task_id="bg_200011_abcd12",
        prompt="检查上线失败原因",
        source=source,
        session_key="qq_napcat:dm:179033731",
        job_kind="manual",
    )
    store.mark_job_failed("bg_200011_abcd12", error="ssh timeout")

    await deliver_background_job_updates_once(runner)

    content = runner.adapters[Platform.QQ_NAPCAT].send.await_args.kwargs["content"]
    assert "后台任务失败" in content
    assert "`bg_200011_abcd12`" in content
    assert "任务：检查上线失败原因" in content
    assert "错误：ssh timeout" in content
    assert "Background task" not in content


@pytest.mark.asyncio
async def test_delivery_tick_sends_external_approval_prompt(tmp_path):
    runner = _make_runner(tmp_path)
    store = runner._background_job_store
    source = _make_source()

    store.create_job(
        task_id="bg_200002_abcd12",
        prompt="重启服务",
        source=source,
        session_key="qq_napcat:dm:179033731",
    )
    store.create_approval_request(
        task_id="bg_200002_abcd12",
        session_key="qq_napcat:dm:179033731",
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

    await deliver_background_job_updates_once(runner)

    assert runner.adapters[Platform.QQ_NAPCAT].send.await_count == 1
    content = runner.adapters[Platform.QQ_NAPCAT].send.await_args.kwargs["content"]
    assert "后台任务待授权" in content
    assert "`bg_200002_abcd12`" in content
    assert "任务：重启服务" in content
    assert "危险命令需要授权" in content
    assert "董事长" in content
    assert "systemctl restart hermes-gateway.service" in content


def test_background_job_store_exposes_queryable_approval_summary(tmp_path):
    from gateway.background_jobs import BackgroundJobStore

    store = BackgroundJobStore(db_path=tmp_path / "background_jobs.db")
    source = _make_source()
    store.create_job(
        task_id="bg_queryable_1",
        prompt="重启服务",
        source=source,
        session_key="qq_napcat:dm:179033731",
    )
    store.create_approval_request(
        task_id="bg_queryable_1",
        session_key="qq_napcat:dm:179033731",
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

    job = store.get_job("bg_queryable_1")

    assert job is not None
    assert job["pending_approval_count"] == 1
    assert job["query_status"] == "approval_pending"
    assert job["query_status_text"] == "待授权"
    assert job["is_queryable_active"] is True
