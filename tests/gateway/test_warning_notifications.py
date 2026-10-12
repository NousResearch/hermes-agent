"""Warning delivery is independent of normal replies and operational state."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.credits_tracker import AgentNotice
from agent.status_output import StatusOutputMixin
from gateway.config import Platform
from gateway.run import GatewayRunner, _load_gateway_config
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext


from tests.gateway.test_session_hygiene import HygieneCaptureAdapter


class RecordingAdapter(HygieneCaptureAdapter):
    config = SimpleNamespace(extra={})

    def __init__(self):
        super().__init__()
        self.sent = []

    async def send(self, chat_id, content, **kwargs):
        self.sent.append((chat_id, content, kwargs))
        return SimpleNamespace(success=True, message_id=str(len(self.sent)))


class Emitter(StatusOutputMixin):
    suppress_status_output = True
    log_prefix = ""


@pytest.mark.parametrize("platform", [Platform.SLACK, Platform.TELEGRAM, Platform.LOCAL])
@pytest.mark.parametrize("configured,enabled", [
    ("", True), ("display: null", True),
    ("display: {suppress_warning_notifications: null}", True),
    ("display: {suppress_warning_notifications: true}", False),
    ('display: {suppress_warning_notifications: "true"}', False),
    ("display: {suppress_warning_notifications: typo}", True),
    ("display: {suppress_warning_notifications: []}", True),
    ("display: broken", True),
    ("display: [broken]", True),
    ("display: {platforms: broken}", True),
    ("display: {suppress_warning_notifications: true, platforms: {slack: {suppress_warning_notifications: false}}}", True),
])
def test_warning_opt_out_preserves_other_delivery(tmp_path, monkeypatch, platform, configured, enabled):
    from gateway import run

    thread_id = "1700.1"  # threaded vs. unthreaded delivery is covered in test_warning_notifications_transport

    (tmp_path / "config.yaml").write_text(configured)
    monkeypatch.setattr(run, "_hermes_home", tmp_path)
    config = _load_gateway_config()
    if "platforms: {slack" in configured and platform != Platform.SLACK:
        enabled = False

    adapter = RecordingAdapter()
    source = SessionSource(platform=platform, chat_id="chat", user_id="user", thread_id=thread_id)
    gateway = object.__new__(GatewayRunner)
    gateway.config = None
    gateway._delivery_adapter_for = lambda source: adapter
    gateway._thread_metadata_for_source = lambda source: {"thread_id": source.thread_id} if source.thread_id else {}
    ctx = TurnContext(source=source, user_config=config, _run_still_current=lambda: True,
                      _status_adapter=adapter, _status_chat_id=source.chat_id,
                      _status_thread_metadata=gateway._thread_metadata_for_source(source))
    turn = TurnRunner(gateway, ctx)
    monkeypatch.setattr(run, "safe_schedule_threadsafe", lambda coro, *a, **k: asyncio.run(coro))
    emitter = Emitter()
    emitter.status_callback = turn._status_callback_sync
    emitter.notice_callback = turn._notice_callback_sync

    # These are engine rails, not text filtering on the adapter's general send.
    diagnostics = [
        ("warn", "Context overflow blocked; use /compress"),
        ("lifecycle", "⚠ Compression aborted: summary failed; history preserved"),
        ("lifecycle", "↻ Switched to fallback: secondary (provider)"),
        ("lifecycle", "Content filter terminated stream; switching to fallback..."),
        ("lifecycle", "🔐 Authentication failed and could not be refreshed — switching to fallback provider..."),
        ("lifecycle", "ℹ️ Estimated cost of these empty attempts: ~$1.25"),
        ("lifecycle", "⏳ Your Nous account has hit its rate limit; it resets in 1m."),
        ("lifecycle", "📐 Compression could not reduce the request further — removed retained vision payloads and retrying..."),
    ]
    for kind, text in diagnostics:
        if kind == "warn":
            emitter._emit_warning(text)
        else:
            emitter._emit_diagnostic_status(text)
    emitter._pending_fallback_notice = "↻ Switched to fallback: secondary (provider)"
    emitter._emit_pending_fallback_notice()
    assert emitter._pending_fallback_notice is None
    emitter._emit_notice(AgentNotice("Credit access paused", level="error", key="credits.depleted"))
    turn.progress_callback("subagent.complete", status="failed", goal="research", summary="provider unavailable")
    warning_count = len(diagnostics) + 3
    assert len(adapter.sent) == (warning_count if enabled else 0)

    emitter._emit_status("still on it")
    asyncio.run(gateway._deliver_platform_notice(source, "No home channel is set"))
    for text in ("⚠ The requested analysis", "Approval required", "Compression aborted: manual command result", "The task failed"):
        asyncio.run(adapter.send(source.chat_id, text))
    assert len(adapter.sent) == (warning_count if enabled else 0) + 6
    assert all(row[0] == source.chat_id for row in adapter.sent)


@pytest.mark.parametrize("enabled", [True, False])
def test_direct_warning_delivery_keeps_failure_state(tmp_path, monkeypatch, enabled):
    from gateway import run

    (tmp_path / "config.yaml").write_text(f"display: {{suppress_warning_notifications: {str(not enabled).lower()}}}")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(run, "_hermes_home", tmp_path)
    adapter = RecordingAdapter()
    source = SessionSource(platform=Platform.SLACK, chat_id="chat", user_id="user")
    gateway = object.__new__(GatewayRunner)
    gateway._delivery_adapter_for = lambda source: adapter
    gateway._session_db_init_error = "database is locked"
    gateway._session_db_handle_cache = None
    gateway._home_channel_transports = lambda: [(Platform.SLACK, None, SimpleNamespace(chat_id="chat"), adapter)]
    gateway._send_home_channel_message = AsyncMock()
    asyncio.run(gateway._hmwa_hygiene_notify(source, {}, "Compression failed", "failure"))
    asyncio.run(gateway._run_agent_inactivity_warning(SimpleNamespace(agent_warning=60, agent_timeout=120), source, {}))
    asyncio.run(gateway._send_session_db_warning_notifications())
    assert len(adapter.sent) == (2 if enabled else 0)
    assert gateway._send_home_channel_message.await_count == int(enabled)
    assert gateway._session_db_init_error == "database is locked"
