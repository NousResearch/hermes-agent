"""Regression: a busy follow-up the runner re-queues must not hot-loop through the adapter drain.

Field failure (Discord, Windows): the Discord websocket fell behind and the adapter was replaced
by a reconnect while the old adapter's task was still running a long agent turn. The NEW adapter
has no ``_active_sessions`` guard for that session, but the runner still has the running agent.
A message / watch-notification wake arriving while context compression was in flight then:

  adapter.handle_message -> no guard -> _process_message_background -> runner._handle_message
  -> running agent + compression in flight -> "PRIORITY interrupt demoted to queue"
  -> event put back in adapter._pending_messages, handler returns None
  -> adapter in-band drain pops the SAME event and immediately spawns a new drain task -> ...

~240-300 iterations/s for the whole compression window (88k identical INFO lines flooded the 3x5 MB
gateway.log rotation in ~6 minutes), and every iteration started and cancelled a typing refresh:
one aborted HTTPS request to discord.com per iteration, each on a fresh TCP connection. The home
router's IPS classified the burst as a DDoS from this host and blocked discord.com for hours.
"""

import asyncio
import time
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


class _Adapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.TELEGRAM)
        self.sent = []
        self.typing_calls = 0

    @property
    def name(self):
        return "telegram"

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return SendResult(success=True)

    async def send_typing(self, chat_id, metadata=None):
        self.typing_calls += 1

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "private"}


def _source() -> SessionSource:
    return SessionSource(platform=Platform.TELEGRAM, user_id="u1", chat_id="c1",
                         user_name="tester", chat_type="dm")


def _runner_with_running_agent(adapter, *, compression_in_flight):
    """Runner whose agent for the session is still running (owned by a task the adapter no
    longer tracks) — the post-reconnect split-brain."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")})
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    sk = build_session_key(_source())
    entry = SessionEntry(session_key=sk, session_id="sess-1", created_at=datetime.now(),
                         updated_at=datetime.now(), platform=Platform.TELEGRAM, chat_type="dm")
    store = MagicMock()
    store.get_or_create_session.return_value = entry
    store.load_transcript.return_value = []
    store.has_any_sessions.return_value = True
    runner.session_store = store
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._service_tier = None
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_a, **_k: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *a, **k: None
    runner._emit_gateway_run_progress = AsyncMock()
    runner._draining = False
    runner._busy_input_mode = "interrupt"
    runner._agent_has_active_subagents = lambda _agent: False
    runner._session_has_compression_in_flight = AsyncMock(return_value=compression_in_flight)
    agent = MagicMock()
    agent.get_activity_summary.return_value = {
        "seconds_since_activity": 0.0, "last_activity_desc": "api_call",
        "api_call_count": 1, "max_iterations": 60}
    runner._running_agents[sk] = agent
    runner._running_agents_ts[sk] = time.time() - 120
    return runner, agent, sk


@pytest.mark.asyncio
async def test_requeued_busy_event_does_not_hot_loop():
    adapter = _Adapter()
    runner, agent, sk = _runner_with_running_agent(adapter, compression_in_flight=True)
    adapter.set_message_handler(runner._handle_message)

    # The new adapter holds no guard for the session (the reconnect replaced it mid-turn).
    assert sk not in adapter._active_sessions
    await adapter.handle_message(MessageEvent(text="still there?", source=_source(), message_id="m1"))
    await asyncio.sleep(1.0)

    dispatches = runner._session_has_compression_in_flight.await_count
    typing = adapter.typing_calls
    await adapter.cancel_background_tasks()

    agent.interrupt.assert_not_called()
    # Unpatched: hundreds of dispatches (and typing refreshes) per second. Patched: a handful.
    assert dispatches <= 8, f"hot loop: {dispatches} re-dispatches in 1s"
    assert typing <= 8, f"typing churn: {typing} typing requests in 1s"


@pytest.mark.asyncio
async def test_requeued_event_runs_once_the_agent_finishes():
    """The back-off must defer, not drop: once the running agent is gone the event is processed."""
    adapter = _Adapter()
    runner, agent, sk = _runner_with_running_agent(adapter, compression_in_flight=True)
    handled = []
    real_handle = runner._handle_message

    async def handler(event):
        if sk not in runner._running_agents:
            handled.append(event.text)
            return "done"
        return await real_handle(event)

    adapter.set_message_handler(handler)
    await adapter.handle_message(MessageEvent(text="queued msg", source=_source(), message_id="m2"))
    await asyncio.sleep(0.6)
    runner._running_agents.pop(sk)  # the long turn finishes
    for _ in range(100):
        if handled:
            break
        await asyncio.sleep(0.1)
    await adapter.cancel_background_tasks()
    assert handled == ["queued msg"]
