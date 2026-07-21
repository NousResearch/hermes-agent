"""Slack continuity transaction across compression, restart, and final delivery."""

import asyncio
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

import gateway.delivery_ledger as delivery_ledger
import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.session import SessionSource


FINAL_ANSWER = "The continuity transaction completed."


class _RestartBoundarySlackAdapter(BasePlatformAdapter):
    """Fake Slack transport that stops at the pre-ACK restart boundary."""

    def __init__(self, runner, session_key):
        super().__init__(PlatformConfig(enabled=True), Platform.SLACK)
        self.gateway_runner = runner
        self.session_key = session_key
        self.resume_pending_during_send = []

    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        entry = self.gateway_runner.session_store._entries[self.session_key]
        self.resume_pending_during_send.append(entry.resume_pending)
        raise asyncio.CancelledError


class _RecoveredSlackAdapter(BasePlatformAdapter):
    def __init__(self, runner=None):
        super().__init__(PlatformConfig(enabled=True), Platform.SLACK)
        self.gateway_runner = runner
        self.sent = []

    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((chat_id, content, metadata))
        return SendResult(success=True, message_id="recovered-1")


def _source():
    return SessionSource(
        platform=Platform.SLACK,
        chat_id="C-CONTINUITY",
        chat_type="channel",
        thread_id="1710000000.000100",
        user_id="U-OWNER",
    )


def _event(source):
    return MessageEvent(
        text="finish the long-running task",
        message_type=MessageType.TEXT,
        source=source,
        message_id="1710000001.000200",
    )


def _runner(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(gateway_run, "_hermes_home", home)
    monkeypatch.setattr(delivery_ledger, "_db_path", lambda: home / "state.db")

    runner = gateway_run.GatewayRunner(
        GatewayConfig(
            platforms={Platform.SLACK: PlatformConfig(enabled=True)},
            thread_sessions_per_user=False,
        )
    )
    runner._is_user_authorized = lambda source: True

    async def _skip_platform_notice(source, content):
        return None

    runner._deliver_platform_notice = _skip_platform_notice
    return runner


@pytest.mark.asyncio
async def test_slack_compression_restart_final_delivery_is_one_transaction(
    tmp_path, monkeypatch
):
    """A restart cannot clear recovery ownership before Slack ACKs the final."""
    runner = _runner(tmp_path, monkeypatch)
    source = _source()
    event = _event(source)
    session_key = runner._session_key_for_source(source)
    entry = runner.session_store.get_or_create_session(source)
    parent_id = entry.session_id

    # A real persisted long thread and compression lineage, not an in-memory
    # transcript fixture. The deterministic agent result below rotates the
    # active gateway binding from this parent to the child.
    long_history = []
    for index in range(12):
        role = "user" if index % 2 == 0 else "assistant"
        message = {"role": role, "content": f"long thread turn {index}"}
        long_history.append(message)
        runner.session_store.append_to_transcript(
            parent_id,
            message,
        )
    session_db = runner.session_store._db
    assert session_db is not None

    # Drive the real compression rotation twice. The first child create loses
    # a parent race and must roll back to the live parent; the next turn retries
    # successfully, clears the persisted failure guard, and produces one child.
    from run_agent import AIAgent

    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        compression_agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            platform="slack",
            quiet_mode=True,
            session_db=session_db,
            session_id=parent_id,
            skip_context_files=True,
            skip_memory=True,
        )

    compressor = MagicMock()

    def _deterministic_compress(*_args, **_kwargs):
        session_db.clear_compression_failure_cooldown(compression_agent.session_id)
        return [
            {"role": "user", "content": "compressed conversation summary"},
            {"role": "user", "content": "long thread tail"},
        ]

    compressor.compress.side_effect = _deterministic_compress
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_summary_auth_failure = False
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    compression_agent.context_compressor = compressor
    compression_agent.compression_in_place = False

    with patch.object(
        session_db,
        "create_session",
        side_effect=RuntimeError("FOREIGN KEY constraint failed"),
    ):
        compression_agent._compress_context(
            long_history,
            "system prompt",
            approx_tokens=120_000,
        )
    assert compression_agent.session_id == parent_id
    assert session_db.get_session(parent_id) is not None

    session_db.record_compression_failure_cooldown(
        parent_id,
        datetime.now().timestamp() + 300,
        "FOREIGN KEY constraint failed",
    )
    compression_agent._compress_context(
        long_history,
        "system prompt",
        approx_tokens=120_000,
    )
    child_id = compression_agent.session_id
    assert child_id != parent_id
    assert session_db.get_compression_tip(parent_id) == child_id
    assert session_db.get_compression_failure_cooldown(parent_id) is None
    runner.session_store.mark_resume_pending(session_key, reason="restart_timeout")

    async def _compressed_agent_result(
        message,
        context_prompt,
        history,
        source,
        session_id,
        **_kwargs,
    ):
        return {
            "final_response": FINAL_ANSWER,
            "messages": [
                {"role": "user", "content": "compressed conversation summary"},
                {"role": "user", "content": event.text},
                {"role": "assistant", "content": FINAL_ANSWER},
            ],
            "session_id": child_id,
            "history_offset": 0,
            "last_prompt_tokens": 128,
            "api_calls": 1,
            "completed": True,
            "failed": False,
            "tools": [],
        }

    runner._run_agent = _compressed_agent_result
    first_adapter = _RestartBoundarySlackAdapter(runner, session_key)
    first_adapter.set_message_handler(runner._handle_message)
    runner.adapters = {Platform.SLACK: first_adapter}
    first_adapter._active_sessions[session_key] = asyncio.Event()

    with pytest.raises(asyncio.CancelledError):
        await first_adapter._process_message_background(event, session_key)

    # The compressed child is authoritative, and the turn lease unwound at the
    # controlled restart boundary. Recovery ownership must still be durable
    # because Slack never ACKed the final response.
    assert runner.session_store._entries[session_key].session_id == child_id
    assert first_adapter.resume_pending_during_send == [True]
    assert runner.session_store._entries[session_key].resume_pending is True
    assert all(
        lease.holder is None
        for lease in runner._turn_leases._leases.values()
    )

    with delivery_ledger._connect() as conn:
        row = conn.execute(
            "SELECT obligation_id, state, thread_id FROM delivery_obligations"
        ).fetchone()
        assert row is not None
        obligation_id, state, thread_id = row
        assert state == "attempting"
        assert thread_id == source.thread_id
        conn.execute(
            "UPDATE delivery_obligations SET owner_pid=999999999, "
            "owner_started_at=1 WHERE obligation_id=?",
            (obligation_id,),
        )

    recovered_adapter = _RecoveredSlackAdapter()
    replacement = _runner(tmp_path, monkeypatch)
    replacement.adapters = {Platform.SLACK: recovered_adapter}

    assert await replacement._redeliver_pending_obligations() == 1
    assert await replacement._redeliver_pending_obligations() == 0

    assert recovered_adapter.sent == [
        (
            source.chat_id,
            delivery_ledger.RECOVERED_MARKER + FINAL_ANSWER,
            {"thread_id": source.thread_id},
        )
    ]
    recovered_text = recovered_adapter.sent[0][1]
    forbidden = ("{\"", "tool_calls", "Working", "no response was generated")
    assert recovered_text.strip()
    assert all(marker not in recovered_text for marker in forbidden)

    reloaded = replacement.session_store.get_or_create_session(source)
    assert reloaded.session_key == session_key
    assert reloaded.session_id == child_id
    assert reloaded.resume_pending is False


@pytest.mark.asyncio
async def test_successful_slack_ack_commits_restart_recovery(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch)
    source = _source()
    event = _event(source)
    session_key = runner._session_key_for_source(source)
    entry = runner.session_store.get_or_create_session(source)
    runner.session_store.mark_resume_pending(session_key, reason="restart_timeout")

    async def _completed_agent_result(
        message,
        context_prompt,
        history,
        source,
        session_id,
        **_kwargs,
    ):
        return {
            "final_response": FINAL_ANSWER,
            "messages": [
                {"role": "user", "content": event.text},
                {"role": "assistant", "content": FINAL_ANSWER},
            ],
            "session_id": entry.session_id,
            "history_offset": 0,
            "last_prompt_tokens": 64,
            "api_calls": 1,
            "completed": True,
            "failed": False,
            "tools": [],
        }

    runner._run_agent = _completed_agent_result
    adapter = _RecoveredSlackAdapter(runner)
    adapter.set_message_handler(runner._handle_message)
    runner.adapters = {Platform.SLACK: adapter}
    adapter._active_sessions[session_key] = asyncio.Event()

    await adapter._process_message_background(event, session_key)

    assert adapter.sent == [
        (source.chat_id, FINAL_ANSWER, {"thread_id": source.thread_id, "notify": True})
    ]
    assert runner.session_store._entries[session_key].resume_pending is False
