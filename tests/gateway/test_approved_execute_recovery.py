import json
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.active_task import ActiveTaskStore
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    SessionSource,
    build_session_key,
)
from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL


class _Adapter(BasePlatformAdapter):
    def __init__(self, *, send_result: SendResult | None = None):
        super().__init__(PlatformConfig(), Platform.DISCORD)
        self._send_result = send_result or SendResult(success=True, message_id="ok")
        self.sent: list[str] = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None, **kwargs) -> SendResult:
        self.sent.append(content)
        return self._send_result

    async def send_typing(self, chat_id, metadata=None) -> None:
        pass

    async def get_chat_info(self, chat_id):
        return {"name": "test", "type": "direct", "chat_id": chat_id}


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="thread-parent",
        chat_type="thread",
        user_id="user-1",
        thread_id="thread-1",
    )


def _event(text: str = "status?") -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=_source(),
        message_id="msg-1",
    )


def _init_repo(path: Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=path, check=True)
    (path / "README.md").write_text("# test\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit_file(repo: Path, name: str, content: str) -> str:
    (repo / name).write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", name], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", f"add {name}"], cwd=repo, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.mark.asyncio
async def test_execute_completion_report_recovered_when_delivery_lost(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.active_task.get_hermes_home", lambda: tmp_path)
    store = ActiveTaskStore()

    event = _event("run the task")
    session_key = build_session_key(event.source)
    store.upsert(
        session_key=session_key,
        mode="background_process",
        status="active",
        command="python task.py",
        process_session_id="proc_done",
        final_report_status="pending",
    )
    adapter = _Adapter(send_result=SendResult(success=False, error="network down", retryable=False))
    adapter.set_message_handler(AsyncMock(return_value="final report body"))

    await adapter._process_message_background(event, session_key)

    record = store.get(session_key)
    assert record is not None
    assert record.final_report_status == "failed"
    assert record.final_report_path
    assert Path(record.final_report_path).read_text(encoding="utf-8") == "final report body"
    assert "network down" in (record.final_report_error or "")


@pytest.mark.asyncio
async def test_active_execute_without_process_runs_final_state_recovery(tmp_path):
    repo = tmp_path / "repo"
    head = _init_repo(repo)
    expected = repo / "expected.txt"
    expected.write_text("done\n", encoding="utf-8")

    store = ActiveTaskStore(tmp_path / "active_tasks.json")
    session_key = build_session_key(_source())
    store.upsert(
        session_key=session_key,
        repo_path=str(repo),
        branch="master",
        head=head,
        mode="background_process",
        command="python task.py",
        task_summary="approved execute recovery task",
        status="active",
        process_session_id="proc_missing",
        pid=999999,
        final_report_status="pending",
        expected_commit=head,
        expected_files=json.dumps([str(expected)]),
    )

    adapter = _Adapter()
    runner = object.__new__(GatewayRunner)
    runner.active_task_store = store
    runner.adapters = {Platform.DISCORD: adapter}
    runner._running_agents = {session_key: _AGENT_PENDING_SENTINEL}
    runner._running_agents_ts = {session_key: 1.0}
    runner._busy_ack_ts = {}
    runner._busy_input_mode = "queue"
    runner._busy_text_mode = "interrupt"
    runner._draining = False
    runner._is_user_authorized = lambda _source: True
    runner._release_running_agent_state = MagicMock(return_value=True)
    runner._reply_anchor_for_event = lambda event: event.message_id
    runner._thread_metadata_for_source = lambda source, reply_anchor=None: None

    handled = await runner._handle_active_session_busy_message(_event("still working?"), session_key)

    assert handled is True
    assert adapter.sent
    report = adapter.sent[-1]
    assert "Approved execute recovery" in report
    assert str(repo) in report
    assert head in report
    assert "expected.txt: present" in report
    reloaded = store.get(session_key)
    assert reloaded is not None
    assert reloaded.final_report_status == "recovered"
    assert reloaded.last_observed_process_state == "not_found"
    runner._release_running_agent_state.assert_called_once_with(session_key)


@pytest.mark.asyncio
async def test_cached_turn_recovers_completed_test_result_when_process_gone(tmp_path):
    repo = tmp_path / "repo"
    head = _init_repo(repo)

    store = ActiveTaskStore(tmp_path / "active_tasks.json")
    session_key = build_session_key(_source())
    store.upsert(
        session_key=session_key,
        repo_path=str(repo),
        branch="master",
        head=head,
        mode="background_process",
        command="python -m pytest tests/unit",
        task_summary="verification run",
        status="succeeded",
        process_session_id="proc_done",
        pid=999999,
        exit_code=0,
        completed_at="2026-06-02T00:00:00+00:00",
        output_tail="2 passed in 0.10s",
        last_observed_process_state="exited",
        final_report_status="pending",
    )

    adapter = _Adapter()
    runner = object.__new__(GatewayRunner)
    runner.active_task_store = store
    runner.adapters = {Platform.DISCORD: adapter}
    runner._running_agents = {session_key: _AGENT_PENDING_SENTINEL}
    runner._running_agents_ts = {session_key: 1.0}
    runner._busy_ack_ts = {}
    runner._busy_input_mode = "queue"
    runner._busy_text_mode = "interrupt"
    runner._draining = False
    runner._is_user_authorized = lambda _source: True
    runner._release_running_agent_state = MagicMock(return_value=True)
    runner._reply_anchor_for_event = lambda event: event.message_id
    runner._thread_metadata_for_source = lambda source, reply_anchor=None: None

    handled = await runner._handle_active_session_busy_message(_event("still working?"), session_key)

    assert handled is True
    report = adapter.sent[-1]
    assert "Approved execute recovery" in report
    assert "Process state: exited" in report
    assert "Exit code: 0" in report
    assert "2 passed in 0.10s" in report
    reloaded = store.get(session_key)
    assert reloaded is not None
    assert reloaded.final_report_status == "recovered"
    runner._release_running_agent_state.assert_called_once_with(session_key)


@pytest.mark.asyncio
async def test_recovery_uses_intended_repo_not_gateway_cwd(tmp_path, monkeypatch):
    gateway_repo = tmp_path / "gateway"
    task_repo = tmp_path / "task"
    _init_repo(gateway_repo)
    _init_repo(task_repo)
    gateway_head = _commit_file(gateway_repo, "gateway.txt", "gateway\n")
    task_head = _commit_file(task_repo, "task.txt", "task\n")
    expected = task_repo / "review-packages" / "signal-room-format-proof-ai-electricity-new-oil"
    expected.mkdir(parents=True)
    monkeypatch.chdir(gateway_repo)

    store = ActiveTaskStore(tmp_path / "active_tasks.json")
    session_key = build_session_key(_source())
    store.upsert(
        session_key=session_key,
        repo_path=str(task_repo),
        branch="longform-video-production-pipeline",
        head=task_head,
        mode="approved_execute",
        command="python pipeline.py",
        task_summary="Signal Room video pipeline",
        status="active",
        process_session_id="proc_missing",
        final_report_status="pending",
        expected_files=json.dumps(
            ["review-packages/signal-room-format-proof-ai-electricity-new-oil/"]
        ),
    )

    adapter = _Adapter()
    runner = object.__new__(GatewayRunner)
    runner.active_task_store = store
    runner.adapters = {Platform.DISCORD: adapter}
    runner._running_agents = {session_key: _AGENT_PENDING_SENTINEL}
    runner._running_agents_ts = {session_key: 1.0}
    runner._busy_ack_ts = {}
    runner._busy_input_mode = "queue"
    runner._busy_text_mode = "interrupt"
    runner._draining = False
    runner._is_user_authorized = lambda _source: True
    runner._release_running_agent_state = MagicMock(return_value=True)
    runner._reply_anchor_for_event = lambda event: event.message_id
    runner._thread_metadata_for_source = lambda source, reply_anchor=None: None

    handled = await runner._handle_active_session_busy_message(_event("recover"), session_key)

    assert handled is True
    report = adapter.sent[-1]
    assert f"Gateway cwd before re-anchor: {gateway_repo}" in report
    assert f"Intended task repo: {task_repo}" in report
    assert f"Actual repo used for inspection: {task_repo}" in report
    assert "Repo mismatch corrected: yes" in report
    assert task_head in report
    assert gateway_head not in report
    assert "signal-room-format-proof-ai-electricity-new-oil: present" in report


@pytest.mark.asyncio
async def test_recovery_asks_when_intended_repo_missing(tmp_path, monkeypatch):
    gateway_repo = tmp_path / "gateway"
    gateway_head = _init_repo(gateway_repo)
    monkeypatch.chdir(gateway_repo)

    store = ActiveTaskStore(tmp_path / "active_tasks.json")
    session_key = build_session_key(_source())
    store.upsert(
        session_key=session_key,
        mode="approved_execute",
        command="python pipeline.py",
        task_summary="Signal Room video pipeline",
        status="active",
        process_session_id="proc_missing",
        final_report_status="pending",
    )

    adapter = _Adapter()
    runner = object.__new__(GatewayRunner)
    runner.active_task_store = store
    runner.adapters = {Platform.DISCORD: adapter}
    runner._running_agents = {session_key: _AGENT_PENDING_SENTINEL}
    runner._running_agents_ts = {session_key: 1.0}
    runner._busy_ack_ts = {}
    runner._busy_input_mode = "queue"
    runner._busy_text_mode = "interrupt"
    runner._draining = False
    runner._is_user_authorized = lambda _source: True
    runner._release_running_agent_state = MagicMock(return_value=True)
    runner._reply_anchor_for_event = lambda event: event.message_id
    runner._thread_metadata_for_source = lambda source, reply_anchor=None: None

    handled = await runner._handle_active_session_busy_message(_event("recover"), session_key)

    assert handled is True
    report = adapter.sent[-1]
    assert "Recovery paused: intended task repo is missing or ambiguous." in report
    assert f"Gateway cwd before re-anchor: {gateway_repo}" in report
    assert "Intended task repo: unknown" in report
    assert gateway_head not in report
    reloaded = store.get(session_key)
    assert reloaded is not None
    assert reloaded.status == "active"
    runner._release_running_agent_state.assert_not_called()


@pytest.mark.asyncio
async def test_active_execute_recovery_report_send_failure_stays_replayable(tmp_path):
    repo = tmp_path / "repo"
    head = _init_repo(repo)

    store = ActiveTaskStore(tmp_path / "active_tasks.json")
    session_key = build_session_key(_source())
    store.upsert(
        session_key=session_key,
        repo_path=str(repo),
        branch="master",
        head=head,
        mode="background_process",
        command="python task.py",
        status="active",
        process_session_id="proc_missing",
        final_report_status="pending",
    )

    adapter = _Adapter(send_result=SendResult(success=False, error="network down", retryable=False))
    runner = object.__new__(GatewayRunner)
    runner.active_task_store = store
    runner.adapters = {Platform.DISCORD: adapter}
    runner._running_agents = {session_key: _AGENT_PENDING_SENTINEL}
    runner._running_agents_ts = {session_key: 1.0}
    runner._busy_ack_ts = {}
    runner._busy_input_mode = "queue"
    runner._busy_text_mode = "interrupt"
    runner._draining = False
    runner._is_user_authorized = lambda _source: True
    runner._release_running_agent_state = MagicMock(return_value=True)
    runner._reply_anchor_for_event = lambda event: event.message_id
    runner._thread_metadata_for_source = lambda source, reply_anchor=None: None

    handled = await runner._handle_active_session_busy_message(_event("still working?"), session_key)

    assert handled is True
    reloaded = store.get(session_key)
    assert reloaded is not None
    assert reloaded.status == "detached"
    assert reloaded.final_report_status == "failed"
    assert "network down" in (reloaded.final_report_error or "")
    assert reloaded.final_report_path
    assert "Approved execute recovery" in Path(reloaded.final_report_path).read_text(encoding="utf-8")
    runner._release_running_agent_state.assert_called_once_with(session_key)


@pytest.mark.asyncio
async def test_next_turn_surfaces_failed_persisted_final_report(tmp_path):
    store = ActiveTaskStore(tmp_path / "active_tasks.json")
    session_key = build_session_key(_source())
    report_path = tmp_path / "lost_report.txt"
    report_path.write_text("lost final report", encoding="utf-8")
    store.upsert(
        session_key=session_key,
        mode="background_process",
        status="detached",
        command="python task.py",
        final_report_status="failed",
        final_report_path=str(report_path),
        final_report_error="previous send failed",
    )

    adapter = _Adapter()
    runner = object.__new__(GatewayRunner)
    runner.active_task_store = store
    runner.adapters = {Platform.DISCORD: adapter}
    runner._reply_anchor_for_event = lambda event: event.message_id
    runner._thread_metadata_for_source = lambda source, reply_anchor=None: None

    surfaced = await runner._surface_pending_final_report(_event("next message"), session_key)

    assert surfaced is True
    assert "Recovered final report" in adapter.sent[-1]
    assert "lost final report" in adapter.sent[-1]
    reloaded = store.get(session_key)
    assert reloaded is not None
    assert reloaded.final_report_status == "recovered"


@pytest.mark.asyncio
async def test_final_report_uses_task_repo_head_not_gateway_repo_head(tmp_path, monkeypatch):
    gateway_repo = tmp_path / "gateway"
    task_repo = tmp_path / "task"
    _init_repo(gateway_repo)
    _init_repo(task_repo)
    gateway_head = _commit_file(gateway_repo, "gateway.txt", "gateway\n")
    task_head = _commit_file(task_repo, "task.txt", "task\n")
    monkeypatch.chdir(gateway_repo)

    store = ActiveTaskStore(tmp_path / "active_tasks.json")
    session_key = build_session_key(_source())
    report_path = tmp_path / "lost_report.txt"
    report_path.write_text("lost final report", encoding="utf-8")
    store.upsert(
        session_key=session_key,
        repo_path=str(task_repo),
        branch="longform-video-production-pipeline",
        head=task_head,
        mode="approved_execute",
        status="detached",
        command="python pipeline.py",
        task_summary="Signal Room video pipeline",
        final_report_status="failed",
        final_report_path=str(report_path),
        final_report_error="previous send failed",
    )

    adapter = _Adapter()
    runner = object.__new__(GatewayRunner)
    runner.active_task_store = store
    runner.adapters = {Platform.DISCORD: adapter}
    runner._reply_anchor_for_event = lambda event: event.message_id
    runner._thread_metadata_for_source = lambda source, reply_anchor=None: None

    surfaced = await runner._surface_pending_final_report(_event("next message"), session_key)

    assert surfaced is True
    sent = adapter.sent[-1]
    assert f"Gateway cwd before re-anchor: {gateway_repo}" in sent
    assert f"Intended task repo: {task_repo}" in sent
    assert f"Actual repo used for inspection: {task_repo}" in sent
    assert "Repo mismatch corrected: yes" in sent
    assert task_head in sent
    assert gateway_head not in sent
    assert "lost final report" in sent
