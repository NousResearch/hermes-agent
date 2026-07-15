import asyncio
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.run import (
    GatewayRunner,
    _coerce_provider_rate_limit_reset_at,
    _is_verified_provider_rate_limit_resume_event,
    _provider_rate_limit_reset_at,
)
from gateway.session import SessionSource


class _Adapter:
    def __init__(self):
        self.events = []

    async def handle_message(self, event: MessageEvent):
        self.events.append(event)


class _RelayAdapter(_Adapter):
    authorization_is_upstream = True

    def __init__(self):
        super().__init__()
        self.captured = []

    def _capture_scope(self, event):
        self.captured.append(event)


class _SessionStore:
    def __init__(self):
        self.marks = []
        self.mark_threads = []
        self.clears = []
        self.can_resume = True

    def mark_resume_pending(
        self,
        session_key,
        reason="restart_timeout",
        not_before=None,
        resume_transport=None,
    ):
        self.mark_threads.append(threading.get_ident())
        self.marks.append((session_key, reason, not_before, resume_transport))
        return True

    def is_resume_pending(self, session_key, reason=None):
        return self.can_resume

    def clear_resume_pending(self, session_key, reason=None):
        self.clears.append((session_key, reason))
        if not self.can_resume or reason not in {None, "provider_rate_limit"}:
            return False
        self.can_resume = False
        return True


class _ConcurrencyAdapter(_Adapter):
    def __init__(self):
        super().__init__()
        self.active = 0
        self.max_active = 0
        self._session_tasks = {}
        self._both_active = asyncio.Event()

    async def _run_background(self, event: MessageEvent):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.active >= 2:
            self._both_active.set()
        try:
            await asyncio.wait_for(self._both_active.wait(), timeout=1.0)
            self.events.append(event)
        finally:
            self.active -= 1

    async def handle_message(self, event: MessageEvent):
        session_key = {"C1": "s1", "C2": "s2"}[event.source.chat_id]
        self._session_tasks[session_key] = asyncio.create_task(
            self._run_background(event)
        )


def _runner(adapter: _Adapter) -> GatewayRunner:
    runner: Any = object.__new__(GatewayRunner)
    runner.adapters = {Platform.SLACK: adapter}
    runner._background_tasks = set()
    runner._provider_rate_limit_resume_tasks = {}
    runner._session_run_generation = {"s1": 7}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._persist_active_agents = lambda: None
    runner._is_user_authorized = lambda source: True
    runner.session_store = _SessionStore()
    return runner


def test_provider_limit_reset_time_requires_future_timestamp():
    assert _coerce_provider_rate_limit_reset_at(1200, now=1000) == 1200
    assert (
        _coerce_provider_rate_limit_reset_at(
            "1970-01-01T00:20:00+00:00", now=1000
        )
        == 1200
    )
    assert _coerce_provider_rate_limit_reset_at("1970-01-01T00:20:00", now=1000) is None
    assert _coerce_provider_rate_limit_reset_at(900, now=1000) is None
    assert _coerce_provider_rate_limit_reset_at("not-a-time", now=1000) is None
    assert _coerce_provider_rate_limit_reset_at(float("nan"), now=1000) is None
    assert _coerce_provider_rate_limit_reset_at(float("inf"), now=1000) is None
    assert _coerce_provider_rate_limit_reset_at(1e100, now=1000) is None


def test_provider_limit_reset_time_rejects_deadlines_beyond_session_retention():
    now = 1_000.0

    assert _coerce_provider_rate_limit_reset_at(now + (90 * 24 * 60 * 60), now=now)
    assert (
        _coerce_provider_rate_limit_reset_at(
            now + (90 * 24 * 60 * 60) + 1,
            now=now,
        )
        is None
    )


def test_provider_limit_reset_only_uses_rate_limit_results():
    future_reset = time.time() + 120
    assert (
        _provider_rate_limit_reset_at(
            {"failure_reason": "billing", "error_context": {"reset_at": future_reset}}
        )
        is None
    )
    assert (
        _provider_rate_limit_reset_at(
            {"failure_reason": "rate_limit", "error_context": {}}
        )
        is None
    )
    assert (
        _provider_rate_limit_reset_at(
            {
                "failure_reason": "rate_limit",
                "error_context": {"reset_at": future_reset},
            }
        )
        == future_reset
    )


@pytest.mark.asyncio
async def test_provider_limit_resume_dispatches_internal_continuation():
    adapter = _Adapter()
    runner = _runner(adapter)
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", user_id="U1")

    await runner._run_provider_rate_limit_resume_after_delay(
        session_key="s1",
        source=source,
        reset_at=0,
        run_generation=7,
    )

    assert len(adapter.events) == 1
    event = adapter.events[0]
    assert event.internal is True
    assert event.source is source
    assert event.text == ""
    assert event.metadata.get("_hermes_provider_rate_limit_resume") is not None

    entry = SimpleNamespace(
        resume_pending=True,
        resume_reason="provider_rate_limit",
        resume_not_before=0.0,
    )
    assert _is_verified_provider_rate_limit_resume_event(
        event,
        entry,
        run_generation=8,
        now=1.0,
    )
    assert not _is_verified_provider_rate_limit_resume_event(
        MessageEvent(
            text="unrelated",
            source=source,
            internal=True,
        ),
        entry,
        run_generation=8,
        now=1.0,
    )
    assert not _is_verified_provider_rate_limit_resume_event(
        event,
        entry,
        run_generation=8,
        now=-1.0,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source",
    [
        SessionSource(
            platform=Platform.DISCORD,
            chat_id="C1",
            user_id="U1",
            scope_id="G1",
        ),
        SessionSource(
            platform=Platform.RELAY,
            chat_id="C1",
            user_id="U1",
            scope_id="G1",
        ),
    ],
)
async def test_provider_limit_resume_routes_relay_origin_through_relay_adapter(source):
    relay = _RelayAdapter()
    runner = _runner(relay)
    runner.adapters = {Platform.RELAY: relay}

    await runner._run_provider_rate_limit_resume_after_delay(
        session_key="s1",
        source=source,
        reset_at=0,
        run_generation=7,
        resume_transport=Platform.RELAY.value,
    )

    assert len(relay.events) == 1
    assert relay.events[0].source.platform == source.platform
    assert relay.events[0].source.delivered_via_upstream_relay is True
    assert relay.captured == relay.events


@pytest.mark.asyncio
async def test_provider_limit_resume_skips_suspended_or_cleared_session():
    adapter = _Adapter()
    runner = _runner(adapter)
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", user_id="U1")
    runner.session_store.can_resume = False

    await runner._run_provider_rate_limit_resume_after_delay(
        session_key="s1",
        source=source,
        reset_at=0,
        run_generation=7,
    )

    assert adapter.events == []


@pytest.mark.asyncio
async def test_provider_limit_resume_waits_for_origin_run_to_release_slot():
    adapter = _Adapter()
    runner = _runner(adapter)
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", user_id="U1")
    runner._running_agents["s1"] = object()

    continuation = asyncio.create_task(
        runner._run_provider_rate_limit_resume_after_delay(
            session_key="s1",
            source=source,
            reset_at=0,
            run_generation=7,
        )
    )
    await asyncio.sleep(0.05)
    assert adapter.events == []
    assert not continuation.done()

    runner._running_agents.pop("s1")
    await asyncio.wait_for(continuation, timeout=1.0)

    assert len(adapter.events) == 1


@pytest.mark.asyncio
async def test_provider_limit_resume_skips_when_newer_turn_exists():
    adapter = _Adapter()
    runner = _runner(adapter)
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", user_id="U1")
    runner._session_run_generation["s1"] = 8

    await runner._run_provider_rate_limit_resume_after_delay(
        session_key="s1",
        source=source,
        reset_at=0,
        run_generation=7,
    )

    assert adapter.events == []


@pytest.mark.asyncio
async def test_provider_limit_resume_does_not_overwrite_newer_turn_after_state_read():
    adapter = _Adapter()
    runner = _runner(adapter)
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", user_id="U1")
    origin = object()
    newer = object()
    runner._running_agents["s1"] = origin

    class _RacingAsyncStore:
        calls = 0

        def __init__(self, store):
            self._store = store

        async def is_resume_pending(self, session_key, reason=None):
            self.calls += 1
            if self.calls == 2:
                runner._session_run_generation["s1"] = 8
                runner._running_agents["s1"] = newer
            return True

    runner._async_session_store = _RacingAsyncStore(runner.session_store)
    continuation = asyncio.create_task(
        runner._run_provider_rate_limit_resume_after_delay(
            session_key="s1",
            source=source,
            reset_at=0,
            run_generation=7,
        )
    )
    await asyncio.sleep(0.05)
    runner._running_agents.pop("s1")

    await asyncio.wait_for(continuation, timeout=1.0)

    assert adapter.events == []
    assert runner._running_agents["s1"] is newer


@pytest.mark.asyncio
async def test_real_user_turn_supersedes_durable_provider_continuation():
    adapter = _Adapter()
    runner = _runner(adapter)
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", user_id="U1")
    reset_at = time.time() + 10_000
    assert await runner._schedule_provider_rate_limit_resume(
        session_key="s1",
        source=source,
        reset_at=reset_at,
        run_generation=7,
    )
    stale_task = runner._provider_rate_limit_resume_tasks["s1"]

    superseded = await runner._supersede_provider_rate_limit_resume_for_user_turn(
        session_key="s1"
    )

    assert superseded is True
    assert runner.session_store.clears == [("s1", "provider_rate_limit")]
    assert runner.session_store.can_resume is False
    assert "s1" not in runner._provider_rate_limit_resume_tasks
    await asyncio.gather(stale_task, return_exceptions=True)
    assert stale_task.cancelled()


@pytest.mark.asyncio
async def test_provider_limit_resume_scheduler_persists_deadline():
    adapter = _Adapter()
    runner = _runner(adapter)
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", user_id="U1")
    reset_at = time.time() + 10_000
    loop_thread = threading.get_ident()

    scheduled = await runner._schedule_provider_rate_limit_resume(
        session_key="s1",
        source=source,
        reset_at=reset_at,
        run_generation=7,
    )

    assert scheduled is True
    assert runner.session_store.marks == [
        ("s1", "provider_rate_limit", reset_at, None)
    ]
    assert runner.session_store.mark_threads != [loop_thread]
    assert "s1" in runner._provider_rate_limit_resume_tasks
    task = runner._provider_rate_limit_resume_tasks["s1"]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source",
    [
        SessionSource(
            platform=Platform.DISCORD,
            chat_id="C1",
            user_id="U1",
            delivered_via_upstream_relay=True,
        ),
        SessionSource(
            platform=Platform.RELAY,
            chat_id="C1",
            user_id="U1",
        ),
    ],
)
async def test_provider_limit_scheduler_persists_relay_transport(source):
    relay = _RelayAdapter()
    runner = _runner(relay)
    runner.adapters = {Platform.RELAY: relay}
    reset_at = time.time() + 10_000

    assert await runner._schedule_provider_rate_limit_resume(
        session_key="s1",
        source=source,
        reset_at=reset_at,
        run_generation=7,
    )
    assert runner.session_store.marks == [
        ("s1", "provider_rate_limit", reset_at, Platform.RELAY.value)
    ]
    task = runner._provider_rate_limit_resume_tasks["s1"]
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_provider_limit_scheduler_replaces_stale_wait_for_newer_turn():
    adapter = _Adapter()
    runner = _runner(adapter)
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", user_id="U1")

    assert await runner._schedule_provider_rate_limit_resume(
        session_key="s1",
        source=source,
        reset_at=time.time() + 10_000,
        run_generation=7,
    )
    first = runner._provider_rate_limit_resume_tasks["s1"]
    runner._session_run_generation["s1"] = 8

    assert await runner._schedule_provider_rate_limit_resume(
        session_key="s1",
        source=source,
        reset_at=time.time() + 20_000,
        run_generation=8,
    )
    second = runner._provider_rate_limit_resume_tasks["s1"]

    assert second is not first
    assert first.cancelled() or first.cancelling()
    second.cancel()
    await asyncio.gather(first, second, return_exceptions=True)


@pytest.mark.asyncio
async def test_provider_limit_scheduler_does_not_cancel_dispatched_resume():
    adapter = _Adapter()
    runner = _runner(adapter)
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", user_id="U1")

    assert await runner._schedule_provider_rate_limit_resume(
        session_key="s1",
        source=source,
        reset_at=time.time() + 10_000,
        run_generation=7,
    )
    first = runner._provider_rate_limit_resume_tasks["s1"]
    setattr(first, "_hermes_provider_resume_dispatched", True)
    runner._session_run_generation["s1"] = 8

    assert await runner._schedule_provider_rate_limit_resume(
        session_key="s1",
        source=source,
        reset_at=time.time() + 20_000,
        run_generation=8,
    )
    second = runner._provider_rate_limit_resume_tasks["s1"]

    assert not first.cancelling()
    first.cancel()
    second.cancel()
    await asyncio.gather(first, second, return_exceptions=True)


@pytest.mark.asyncio
async def test_provider_limit_scheduler_deduplicates_same_dispatched_identity():
    adapter = _Adapter()
    runner = _runner(adapter)
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", user_id="U1")
    reset_at = time.time() + 10_000

    assert runner._install_provider_rate_limit_resume(
        session_key="s1",
        source=source,
        reset_at=reset_at,
        run_generation=7,
    )
    first = runner._provider_rate_limit_resume_tasks["s1"]
    setattr(first, "_hermes_provider_resume_dispatched", True)

    assert runner._install_provider_rate_limit_resume(
        session_key="s1",
        source=source,
        reset_at=reset_at,
        run_generation=7,
    )

    assert runner._provider_rate_limit_resume_tasks["s1"] is first
    first.cancel()
    await asyncio.gather(first, return_exceptions=True)


@pytest.mark.asyncio
async def test_provider_limit_resume_uses_source_profile_adapter():
    adapter = _Adapter()
    runner = _runner(adapter)
    runner.adapters = {}
    runner.__dict__["_profile_adapters"] = {"work": {Platform.SLACK: adapter}}
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C1",
        user_id="U1",
        profile="work",
    )

    await runner._run_provider_rate_limit_resume_after_delay(
        session_key="s1", source=source, reset_at=0, run_generation=7
    )

    assert len(adapter.events) == 1


@pytest.mark.asyncio
async def test_provider_limit_resumes_run_in_parallel_across_sessions():
    adapter = _ConcurrencyAdapter()
    runner = _runner(adapter)
    runner._session_run_generation["s2"] = 7
    source1 = SessionSource(platform=Platform.SLACK, chat_id="C1", user_id="U1")
    source2 = SessionSource(platform=Platform.SLACK, chat_id="C2", user_id="U2")

    await asyncio.gather(
        runner._run_provider_rate_limit_resume_after_delay(
            session_key="s1", source=source1, reset_at=0, run_generation=7
        ),
        runner._run_provider_rate_limit_resume_after_delay(
            session_key="s2", source=source2, reset_at=0, run_generation=7
        ),
    )

    assert len(adapter.events) == 2
    assert adapter.max_active == 2


@pytest.mark.asyncio
async def test_unrelated_internal_turn_does_not_steal_provider_resume_guidance(
    monkeypatch, tmp_path
):
    """Fresh transcript must not unlock provider guidance before the due event."""
    import sys
    import threading
    import types
    from datetime import datetime
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    import gateway.run as gateway_run
    from gateway.session import SessionEntry

    class _CapturingAgent:
        messages: list[str] = []

        def __init__(self, *args, **kwargs):
            self.tools = []
            self.tool_progress_callback = None
            self.tool_start_callback = None
            self.step_callback = None
            self.stream_delta_callback = None
            self.status_callback = None
            self.background_review_callback = None

        def run_conversation(self, user_message, **kwargs):
            text = user_message if isinstance(user_message, str) else str(user_message)
            type(self).messages.append(text)
            return {
                "final_response": "ok",
                "messages": [],
                "api_calls": 1,
                "completed": True,
                "failed": False,
                "history_offset": 0,
                "last_prompt_tokens": 0,
            }

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _CapturingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._service_tier = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._pending_model_notes = {}
    runner._session_db = None
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._session_model_overrides = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False, emit=AsyncMock())
    runner.config = SimpleNamespace(streaming=None, multiplex_profiles=False)
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    runner._enrich_message_with_vision = AsyncMock(side_effect=lambda *_a, **_k: _a[0] if _a else "")
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_a, **_kw: False
    runner._reply_anchor_for_event = lambda _event: None
    runner._get_guild_id = lambda _event: None
    runner._consume_pending_native_image_paths = lambda _key: []
    runner._refresh_fallback_model = lambda: None
    runner._enforce_agent_cache_cap = lambda: None

    session_key = "agent:main:slack:dm:C1"
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", user_id="U1")
    entry = SessionEntry(
        session_key=session_key,
        session_id="sess-provider",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.SLACK,
        chat_type="dm",
        origin=source,
        resume_pending=True,
        resume_reason="provider_rate_limit",
        last_resume_marked_at=datetime.now(),
        resume_not_before=time.time() + 3_600,
    )
    runner.session_store = MagicMock()
    runner.session_store._entries = {session_key: entry}
    runner.session_store.get_or_create_session.return_value = entry
    runner.session_store.load_transcript.return_value = [
        {
            "role": "user",
            "content": "recent turn",
            "timestamp": datetime.now().isoformat(),
        }
    ]
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner.session_store.is_resume_pending.return_value = True
    runner.session_store.clear_resume_pending = MagicMock(return_value=True)

    (tmp_path / "config.yaml").write_text("agent:\n  system_prompt: test\n", encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_env_path", tmp_path / ".env")
    monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "***",
        },
    )
    import hermes_cli.tools_config as tools_config

    monkeypatch.setattr(
        tools_config, "_get_platform_tools", lambda user_config, platform_key: {"core"}
    )

    history = [
        {
            "role": "user",
            "content": "recent turn",
            "timestamp": datetime.now().isoformat(),
        }
    ]

    # Unrelated turn with a fresh transcript must not get provider guidance.
    _CapturingAgent.messages = []
    await runner._run_agent(
        message="background process finished",
        context_prompt="",
        history=history,
        source=source,
        session_id=entry.session_id,
        session_key=session_key,
        provider_rate_limit_resume=False,
    )
    assert _CapturingAgent.messages
    assert "reset window has now arrived" not in _CapturingAgent.messages[0]
    assert "provider rate limit" not in _CapturingAgent.messages[0].lower()

    # Verified provider resume event may inject the guidance.
    _CapturingAgent.messages = []
    entry.resume_not_before = time.time() - 10
    await runner._run_agent(
        message="",
        context_prompt="",
        history=history,
        source=source,
        session_id=entry.session_id,
        session_key=session_key,
        provider_rate_limit_resume=True,
        suppress_current_user_message_persistence=True,
    )
    assert _CapturingAgent.messages
    assert "reset window has now arrived" in _CapturingAgent.messages[0]
