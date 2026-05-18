"""Phase 3 (T3.4): connect/disconnect lifecycle for the NATS gateway adapter.

Covers:

* Happy-path ``connect()`` — lock acquisition, ``natsagent.connect`` kwargs,
  :class:`natsagent.Agent` construction, prompt-handler registration,
  ``agent.start()`` call order, ``_mark_connected``.
* Lock conflict — second local profile on the same agent/owner/name fails
  fast with ``retryable=False`` so ``gateway/run.py`` doesn't schedule a
  30-s reconnect loop for something only a human can resolve.
* Exception propagation — errors from ``natsagent.connect`` /
  ``Agent(...)`` / ``agent.start()`` each yield a ``retryable=True`` fatal
  error, release the lock, and leave no dangling agent/nc handles.
* Fatal-after-init — a misconfigured adapter (no servers/context) stays
  fatal and never touches the SDK when ``connect()`` is called.
* Idempotent ``disconnect()`` — teardown order is agent.stop → nc.close →
  release lock, and repeat calls are no-ops.

The ``_ensure_natsagent_mock`` autouse in ``conftest.py`` installs a mock
``natsagent`` module; ``gateway.status.acquire_scoped_lock`` /
``release_scoped_lock`` are monkeypatched per-test so nothing touches the
real filesystem lock directory.
"""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.nats import NatsAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_extra(**overrides) -> dict:
    """Return a minimal-but-valid config.extra dict for a NATS adapter.

    Caller can ``_valid_extra(name="other")`` to tweak individual fields.
    """
    base = {
        "servers": ["nats://127.0.0.1:4222"],
        "owner": "rene",
        "name": "gateway",
    }
    base.update(overrides)
    return base


def _build_adapter(**extra_overrides) -> NatsAdapter:
    return NatsAdapter(PlatformConfig(enabled=True, extra=_valid_extra(**extra_overrides)))


@pytest.fixture
def mock_natsagent(monkeypatch):
    """Reset the natsagent mock to a clean state for each test.

    The conftest autouse plants a module-level mock that persists across
    tests; without a fresh reset ``call_args`` from one test bleeds into
    the next and assertions become order-dependent.
    """
    mod = sys.modules["natsagent"]

    # Fresh AsyncMock for connect() — return value's .close must stay
    # awaitable (see conftest rationale).
    mod.connect = AsyncMock()
    mod.connect.return_value.close = AsyncMock()

    # Fresh Agent factory. Each Agent(...) call returns the *same* mock
    # instance so tests can assert on start/stop/on_prompt calls without
    # re-reaching through return_value every time.
    agent_instance = MagicMock()
    agent_instance.start = AsyncMock()
    agent_instance.stop = AsyncMock()
    # on_prompt is synchronous in the real SDK; keep it as a plain
    # MagicMock so assert_called_once_with works without await semantics.
    agent_instance.on_prompt = MagicMock()
    mod.Agent = MagicMock(return_value=agent_instance)

    return mod


@pytest.fixture
def lock_granted(monkeypatch):
    """Install a lock stub that always grants the lock.

    Records calls so tests can verify scope + identity without hitting
    the real filesystem-backed lock directory.
    """
    calls: list[tuple] = []
    releases: list[tuple] = []

    def _acquire(scope, identity, metadata=None):
        calls.append((scope, identity, metadata or {}))
        return True, None

    def _release(scope, identity):
        releases.append((scope, identity))

    monkeypatch.setattr("gateway.status.acquire_scoped_lock", _acquire)
    monkeypatch.setattr("gateway.status.release_scoped_lock", _release)
    return {"acquires": calls, "releases": releases}


# ---------------------------------------------------------------------------
# Happy-path connect
# ---------------------------------------------------------------------------


class TestConnectHappyPath:
    @pytest.mark.asyncio
    async def test_connect_returns_true_and_marks_connected(
        self, mock_natsagent, lock_granted
    ):
        adapter = _build_adapter()
        assert await adapter.connect() is True

        assert adapter.is_connected is True
        assert adapter.has_fatal_error is False
        # Both SDK handles must be stored so disconnect() / send() can use them.
        assert adapter._nc is mock_natsagent.connect.return_value
        assert adapter._agent is mock_natsagent.Agent.return_value

    @pytest.mark.asyncio
    async def test_connect_passes_servers_to_natsagent_connect(
        self, mock_natsagent, lock_granted
    ):
        adapter = _build_adapter(servers=["nats://a:4222", "nats://b:4222"])
        await adapter.connect()

        mock_natsagent.connect.assert_awaited_once()
        kwargs = mock_natsagent.connect.await_args.kwargs
        assert kwargs == {"servers": ["nats://a:4222", "nats://b:4222"]}

    @pytest.mark.asyncio
    async def test_connect_passes_context_to_natsagent_connect(
        self, mock_natsagent, lock_granted, monkeypatch
    ):
        # context is xor-exclusive with servers, so start from a context-only
        # PlatformConfig rather than the _valid_extra() default.
        adapter = NatsAdapter(
            PlatformConfig(
                enabled=True,
                extra={"context": "prod-nats", "owner": "rene", "name": "gateway"},
            )
        )
        await adapter.connect()

        mock_natsagent.connect.assert_awaited_once()
        assert mock_natsagent.connect.await_args.kwargs == {"context": "prod-nats"}

    @pytest.mark.asyncio
    async def test_connect_constructs_agent_with_full_settings(
        self, mock_natsagent, lock_granted
    ):
        adapter = _build_adapter(
            agent="hermes",
            owner="acme",
            name="prod-1",
            heartbeat_interval_s=15,
            max_payload="2MB",
            attachments_ok=False,
        )
        await adapter.connect()

        mock_natsagent.Agent.assert_called_once()
        kwargs = mock_natsagent.Agent.call_args.kwargs
        assert kwargs["agent"] == "hermes"
        assert kwargs["owner"] == "acme"
        assert kwargs["name"] == "prod-1"
        assert kwargs["nc"] is mock_natsagent.connect.return_value
        assert kwargs["heartbeat_interval_s"] == 15
        assert kwargs["max_payload"] == "2MB"
        assert kwargs["attachments_ok"] is False

    @pytest.mark.asyncio
    async def test_connect_registers_prompt_handler_before_start(
        self, mock_natsagent, lock_granted
    ):
        adapter = _build_adapter()
        await adapter.connect()

        agent = mock_natsagent.Agent.return_value
        # ``on_prompt`` is mandatory before ``start()`` per natsagent SDK —
        # if we ever reordered these, start() would raise at runtime with
        # an unhelpful message.
        agent.on_prompt.assert_called_once()
        passed_handler = agent.on_prompt.call_args.args[0]
        assert passed_handler == adapter._on_prompt

        agent.start.assert_awaited_once()

        # Method-call order: on_prompt → start.
        all_calls = agent.mock_calls
        on_prompt_idx = next(
            i for i, c in enumerate(all_calls) if c == call.on_prompt(passed_handler)
        )
        start_idx = next(i for i, c in enumerate(all_calls) if c == call.start())
        assert on_prompt_idx < start_idx

    @pytest.mark.asyncio
    async def test_connect_acquires_scope_lock_with_identity(
        self, mock_natsagent, lock_granted
    ):
        adapter = _build_adapter(agent="hermes", owner="rene", name="gateway")
        await adapter.connect()

        assert lock_granted["acquires"] == [
            ("nats", "hermes:rene:gateway", {"platform": "nats"})
        ]


# ---------------------------------------------------------------------------
# Phase 3 prompt handler stub
# ---------------------------------------------------------------------------


class TestPromptHandlerStub:
    @pytest.mark.asyncio
    async def test_on_prompt_sends_placeholder_response(
        self, mock_natsagent, lock_granted
    ):
        adapter = _build_adapter()
        envelope = MagicMock()
        stream = MagicMock()
        stream.send = AsyncMock()

        await adapter._on_prompt(envelope, stream)

        stream.send.assert_awaited_once()
        payload = stream.send.await_args.args[0]
        # Phase 4 will replace this with the full MessageEvent pipeline.
        # For now we just assert a non-empty string is emitted so callers
        # can verify end-to-end wiring.
        assert isinstance(payload, str)
        assert payload.strip()

    @pytest.mark.asyncio
    async def test_on_prompt_registers_and_deregisters_current_task(
        self, mock_natsagent, lock_granted
    ):
        # The handler must register its own asyncio task in
        # ``_in_flight_handlers`` before any await so _teardown_handles
        # can cancel it mid-flight; the finally block must remove it on
        # normal completion so the set doesn't leak references.
        adapter = _build_adapter()
        observed_in_flight: list = []
        stream = MagicMock()

        async def _peek_registration(_payload):
            # During the send await, the current task should be tracked.
            observed_in_flight.append(set(adapter._in_flight_handlers))

        stream.send = AsyncMock(side_effect=_peek_registration)

        await adapter._on_prompt(MagicMock(), stream)

        # Observed mid-handler: exactly one task (this test's task).
        assert len(observed_in_flight) == 1
        assert len(observed_in_flight[0]) == 1
        # After the handler returns, the set is clean.
        assert adapter._in_flight_handlers == set()

    @pytest.mark.asyncio
    async def test_on_prompt_finally_survives_teardown_clear(
        self, mock_natsagent, lock_granted
    ):
        # Teardown calls ``_in_flight_handlers.clear()`` after gather(),
        # so the handler's finally block may find the set empty — using
        # ``discard`` (not ``remove``) is what keeps the finally block
        # from raising KeyError and masking the original CancelledError.
        adapter = _build_adapter()
        stream = MagicMock()

        async def _clear_mid_send(_payload):
            adapter._in_flight_handlers.clear()

        stream.send = AsyncMock(side_effect=_clear_mid_send)

        # Must not raise — the regression guard for ``remove`` vs.
        # ``discard``.
        await adapter._on_prompt(MagicMock(), stream)


# ---------------------------------------------------------------------------
# Fatal-after-init
# ---------------------------------------------------------------------------


class TestConnectWithFatalInit:
    @pytest.mark.asyncio
    async def test_connect_short_circuits_when_config_was_invalid(
        self, mock_natsagent, lock_granted
    ):
        adapter = NatsAdapter(PlatformConfig(enabled=True, extra={"owner": "rene"}))
        # _init_ already set a non-retryable fatal error.
        assert adapter.has_fatal_error is True
        assert adapter.fatal_error_retryable is False

        assert await adapter.connect() is False
        # Must not touch the SDK or the lock table.
        mock_natsagent.connect.assert_not_called()
        mock_natsagent.Agent.assert_not_called()
        assert lock_granted["acquires"] == []


# ---------------------------------------------------------------------------
# Lock conflict
# ---------------------------------------------------------------------------


class TestLockConflict:
    @pytest.mark.asyncio
    async def test_conflict_reports_fatal_nonretryable_and_skips_sdk(
        self, mock_natsagent, monkeypatch
    ):
        monkeypatch.setattr(
            "gateway.status.acquire_scoped_lock",
            lambda scope, identity, metadata=None: (False, {"pid": 9999}),
        )
        released: list[tuple] = []
        monkeypatch.setattr(
            "gateway.status.release_scoped_lock",
            lambda scope, identity: released.append((scope, identity)),
        )

        adapter = _build_adapter()
        ok = await adapter.connect()

        assert ok is False
        assert adapter.has_fatal_error is True
        assert adapter.fatal_error_code == "nats_lock"
        assert adapter.fatal_error_retryable is False
        assert "already in use" in adapter.fatal_error_message
        # When acquire fails, the SDK must not be touched at all.
        mock_natsagent.connect.assert_not_called()
        mock_natsagent.Agent.assert_not_called()
        # And release_scoped_lock must not have been called — we never
        # owned the lock in the first place, so releasing would nuke the
        # *other* process's record.
        assert released == []


# ---------------------------------------------------------------------------
# Exceptions during connect
# ---------------------------------------------------------------------------


class TestConnectFailurePaths:
    @pytest.mark.asyncio
    async def test_natsagent_connect_failure_marks_retryable_and_releases_lock(
        self, mock_natsagent, lock_granted
    ):
        mock_natsagent.connect.side_effect = RuntimeError("boom")

        adapter = _build_adapter()
        ok = await adapter.connect()

        assert ok is False
        assert adapter.has_fatal_error is True
        assert adapter.fatal_error_code == "nats_connect_error"
        assert adapter.fatal_error_retryable is True
        assert "boom" in adapter.fatal_error_message
        # Lock must be returned on failure — otherwise the next retry
        # attempt would self-conflict on the very same process.
        assert lock_granted["releases"] == [("nats", "hermes:rene:gateway")]
        # No dangling agent handle (we never even got to Agent()).
        assert adapter._agent is None
        assert adapter._nc is None

    @pytest.mark.asyncio
    async def test_agent_construction_failure_releases_and_closes_nc(
        self, mock_natsagent, lock_granted
    ):
        # nc connects fine, but Agent(...) raises — common case when the
        # SDK's AgentSubject.new() rejects a sanitized but still invalid
        # owner/name combo.
        mock_natsagent.Agent.side_effect = ValueError("bad subject")

        adapter = _build_adapter()
        ok = await adapter.connect()

        assert ok is False
        assert adapter.fatal_error_code == "nats_connect_error"
        assert adapter.fatal_error_retryable is True
        # Partial-init nc handle was closed during teardown.
        mock_natsagent.connect.return_value.close.assert_awaited_once()
        assert adapter._nc is None
        assert adapter._agent is None
        assert lock_granted["releases"] == [("nats", "hermes:rene:gateway")]

    @pytest.mark.asyncio
    async def test_agent_start_failure_stops_agent_and_closes_nc(
        self, mock_natsagent, lock_granted
    ):
        agent = mock_natsagent.Agent.return_value
        agent.start.side_effect = RuntimeError("start failed")

        adapter = _build_adapter()
        ok = await adapter.connect()

        assert ok is False
        assert adapter.fatal_error_code == "nats_connect_error"
        assert adapter.fatal_error_retryable is True
        # Teardown must run stop() before close() — heartbeat publisher
        # needs a live nc to finalize, and closing nc first would surface
        # noisy "connection closed" warnings from the heartbeat loop.
        agent.stop.assert_awaited_once()
        mock_natsagent.connect.return_value.close.assert_awaited_once()
        assert adapter._agent is None
        assert adapter._nc is None
        assert lock_granted["releases"] == [("nats", "hermes:rene:gateway")]


# ---------------------------------------------------------------------------
# Disconnect
# ---------------------------------------------------------------------------


class TestDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect_after_successful_connect_tears_down_in_order(
        self, mock_natsagent, lock_granted
    ):
        adapter = _build_adapter()
        await adapter.connect()

        agent = mock_natsagent.Agent.return_value
        nc = mock_natsagent.connect.return_value

        # Strict ordering: agent.stop() must run before nc.close() so
        # the heartbeat loop can exit on a live connection instead of
        # racing the socket close. Record the call order via side_effect
        # lambdas rather than inspecting mock_calls — the latter only
        # captures attribute access per-mock, so cross-mock ordering
        # needs a shared recorder.
        call_order: list[str] = []
        agent.stop.side_effect = lambda: call_order.append("stop")
        nc.close.side_effect = lambda: call_order.append("close")

        await adapter.disconnect()

        assert call_order == ["stop", "close"]
        agent.stop.assert_awaited_once()
        nc.close.assert_awaited_once()
        assert adapter._agent is None
        assert adapter._nc is None
        assert adapter.is_connected is False
        # Lock must be returned so the same profile can reconnect later.
        assert lock_granted["releases"] == [("nats", "hermes:rene:gateway")]

    @pytest.mark.asyncio
    async def test_disconnect_is_idempotent_after_connect(
        self, mock_natsagent, lock_granted
    ):
        adapter = _build_adapter()
        await adapter.connect()
        await adapter.disconnect()
        await adapter.disconnect()  # second call must not blow up

        # stop() / close() still called exactly once — the second
        # disconnect finds nothing to stop because the first already
        # dropped the handles.
        assert mock_natsagent.Agent.return_value.stop.await_count == 1
        assert mock_natsagent.connect.return_value.close.await_count == 1

    @pytest.mark.asyncio
    async def test_disconnect_without_connect_is_safe_noop(
        self, mock_natsagent, lock_granted
    ):
        adapter = _build_adapter()
        await adapter.disconnect()

        # Never called ``connect()``, so the SDK objects should never have
        # been built — and teardown should tolerate that gracefully.
        mock_natsagent.connect.assert_not_called()
        mock_natsagent.Agent.assert_not_called()
        assert adapter.is_connected is False
        # No lock was acquired, so nothing to release.
        assert lock_granted["releases"] == []

    @pytest.mark.asyncio
    async def test_disconnect_tolerates_agent_stop_errors(
        self, mock_natsagent, lock_granted
    ):
        adapter = _build_adapter()
        await adapter.connect()

        mock_natsagent.Agent.return_value.stop.side_effect = RuntimeError("late")
        # Must not raise — gateway shutdown runs this in a loop over all
        # adapters and one raising aborts the shutdown of every platform
        # after it.
        await adapter.disconnect()

        # nc still closed; adapter handles cleared; lock still released.
        mock_natsagent.connect.return_value.close.assert_awaited_once()
        assert adapter._agent is None
        assert adapter._nc is None
        assert lock_granted["releases"] == [("nats", "hermes:rene:gateway")]

    @pytest.mark.asyncio
    async def test_disconnect_cancels_in_flight_handlers(
        self, mock_natsagent, lock_granted
    ):
        # A long-running handler parked on ``asyncio.sleep`` simulates
        # Phase 4's streaming body awaiting the next model delta when
        # gateway shutdown fires. Without cancellation, ``disconnect()``
        # would block indefinitely.
        adapter = _build_adapter()
        await adapter.connect()

        hang_started = asyncio.Event()

        async def _hanging_handler():
            hang_started.set()
            try:
                await asyncio.sleep(60)  # would outlast the test
            except asyncio.CancelledError:
                # Phase 4 handlers will do real cleanup here (flush
                # partial response, emit error chunk). Phase 3's
                # placeholder has nothing to clean up — just re-raise
                # so the cancellation propagates into gather().
                raise

        task = asyncio.create_task(_hanging_handler())
        adapter._in_flight_handlers.add(task)
        await hang_started.wait()

        # Bound the await so a regression would fail the test instead of
        # hanging the whole suite.
        await asyncio.wait_for(adapter.disconnect(), timeout=2.0)

        assert task.cancelled()
        assert adapter._in_flight_handlers == set()
        # Teardown must still run the full sequence after cancellation —
        # stop, close, release lock.
        mock_natsagent.Agent.return_value.stop.assert_awaited_once()
        mock_natsagent.connect.return_value.close.assert_awaited_once()
        assert lock_granted["releases"] == [("nats", "hermes:rene:gateway")]

    @pytest.mark.asyncio
    async def test_disconnect_sets_shutdown_event_before_stop(
        self, mock_natsagent, lock_granted
    ):
        # Phase 4 handlers will gate their streaming loops on
        # ``self._shutdown_event`` — verify the event is set BEFORE the
        # agent is stopped, so a handler checking the event between
        # deltas sees the shutdown signal before the SDK deregisters
        # the endpoint underneath it.
        adapter = _build_adapter()
        await adapter.connect()

        observed: dict[str, bool] = {}

        def _record_state():
            observed["shutdown_event_set_at_stop"] = adapter._shutdown_event.is_set()

        mock_natsagent.Agent.return_value.stop.side_effect = _record_state

        await adapter.disconnect()

        assert observed["shutdown_event_set_at_stop"] is True

    @pytest.mark.asyncio
    async def test_connect_clears_shutdown_event_on_retry(
        self, mock_natsagent, lock_granted
    ):
        # After a prior teardown (connect failure or disconnect), the
        # shutdown event is set. A retry must clear it so Phase 4's
        # long-running handlers don't see the stale signal and bail out
        # on their first await.
        adapter = _build_adapter()
        adapter._shutdown_event.set()

        assert await adapter.connect() is True

        assert adapter._shutdown_event.is_set() is False


# ---------------------------------------------------------------------------
# Platform identity — sanity checks that platform enum wiring is correct.
# ---------------------------------------------------------------------------


class TestPlatformIdentity:
    def test_adapter_reports_nats_platform(self, mock_natsagent, lock_granted):
        adapter = _build_adapter()
        assert adapter.platform is Platform.NATS
