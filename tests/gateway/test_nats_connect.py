"""Phase 3 (T3.4): connect/disconnect lifecycle for the NATS gateway adapter.

Covers:

* Happy-path ``connect()`` — lock acquisition, ``nats.connect`` kwargs,
  :class:`AgentService` construction, prompt-handler registration,
  ``service.start()`` call order, ``_mark_connected``.
* Lock conflict — second local profile on the same agent/owner/session_name
  fails fast with ``retryable=False`` so ``gateway/run.py`` doesn't schedule
  a 30-s reconnect loop for something only a human can resolve.
* Exception propagation — errors from ``nats.connect`` /
  ``AgentService(...)`` / ``service.start()`` each yield a
  ``retryable=True`` fatal error, release the lock, and leave no
  dangling service/nc handles.
* Fatal-after-init — a misconfigured adapter (no servers/context) stays
  fatal and never touches the SDK when ``connect()`` is called.
* Idempotent ``disconnect()`` — teardown order is service.stop → nc.close →
  release lock, and repeat calls are no-ops.

The ``_ensure_synadia_agents_mock`` autouse in ``conftest.py`` installs a mock
``synadia_ai.agents`` module; ``gateway.status.acquire_scoped_lock`` /
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

    Caller can ``_valid_extra(session_name="other")`` to tweak individual
    fields.
    """
    base = {
        "servers": ["nats://127.0.0.1:4222"],
        "owner": "rene",
        "session_name": "default",
    }
    base.update(overrides)
    return base


def _build_adapter(**extra_overrides) -> NatsAdapter:
    return NatsAdapter(PlatformConfig(enabled=True, extra=_valid_extra(**extra_overrides)))


@pytest.fixture
def mock_synadia_agents(monkeypatch):
    """Reset the synadia_ai.{agents,agent_service} mocks for each test.

    The conftest autouse plants module-level mocks that persist across
    tests; without a fresh reset ``call_args`` from one test bleeds into
    the next and assertions become order-dependent.

    After the v0.5 client / v0.1 agent SDK split, ``AgentService`` lives
    in ``synadia_ai.agent_service`` and the wire types
    (``load_context_options`` etc.) live in ``synadia_ai.agents``. The
    returned proxy exposes ``AgentService`` from the agent_service module
    so existing test assertions (``mock_synadia_agents.AgentService...``)
    keep working transparently.
    """
    client_mod = sys.modules["synadia_ai.agents"]
    svc_mod = sys.modules["synadia_ai.agent_service"]

    # Fresh AgentService factory. Each AgentService(...) call returns the
    # *same* mock instance so tests can assert on start/stop/on_prompt
    # calls without re-reaching through return_value every time.
    service_instance = MagicMock()
    service_instance.start = AsyncMock()
    service_instance.stop = AsyncMock()
    # on_prompt is synchronous in the real SDK; keep it as a plain
    # MagicMock so assert_called_once_with works without await semantics.
    service_instance.on_prompt = MagicMock()
    svc_mod.AgentService = MagicMock(return_value=service_instance)

    # ``load_context_options`` translates a `nats` context name → kwargs
    # for nats.connect. The adapter splats the result.
    client_mod.load_context_options = MagicMock(return_value={"servers": ["nats://stub:4222"]})

    # Proxy that surfaces AgentService from agent_service while keeping
    # the wire-type attributes accessible for existing tests.
    class _SdkProxy:
        AgentService = svc_mod.AgentService
        load_context_options = client_mod.load_context_options
        agents = client_mod
        agent_service = svc_mod

    return _SdkProxy()


@pytest.fixture
def mock_nats(monkeypatch):
    """Reset the nats-py mock to a clean state for each test.

    The adapter calls ``nats.connect(...)`` directly — the SDK does NOT
    own NATS connections. Tests assert against ``mock_nats.connect`` for
    URL/context resolution and against the returned client mock for
    ``.close()`` lifecycle.

    Sets ``return_value.max_payload`` to 1 MiB so the new
    broker-derivation path in ``_on_connect`` (PR #41 alignment) has an
    integer to format. Tests that exercise larger brokers can override.
    """
    mod = sys.modules["nats"]
    mod.connect = AsyncMock()
    mod.connect.return_value.close = AsyncMock()
    mod.connect.return_value.max_payload = 1024 * 1024  # 1 MiB
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
        self, mock_synadia_agents, mock_nats, lock_granted
    ):
        adapter = _build_adapter()
        assert await adapter.connect() is True

        assert adapter.is_connected is True
        assert adapter.has_fatal_error is False
        # Both SDK handles must be stored so disconnect() / send() can use them.
        assert adapter._nc is mock_nats.connect.return_value
        assert adapter._service is mock_synadia_agents.AgentService.return_value

    @pytest.mark.asyncio
    async def test_connect_passes_servers_to_nats_connect(
        self, mock_synadia_agents, mock_nats, lock_granted
    ):
        adapter = _build_adapter(servers=["nats://a:4222", "nats://b:4222"])
        await adapter.connect()

        mock_nats.connect.assert_awaited_once()
        kwargs = mock_nats.connect.await_args.kwargs
        assert kwargs == {"servers": ["nats://a:4222", "nats://b:4222"]}

    @pytest.mark.asyncio
    async def test_connect_routes_context_through_load_context_options(
        self, mock_synadia_agents, mock_nats, lock_granted, monkeypatch
    ):
        # The SDK does NOT own NATS connections — the adapter calls
        # ``nats.connect(**sdk.load_context_options(name))`` directly.
        # Verify the context name is forwarded to the SDK helper and the
        # resulting kwargs are splatted into nats.connect.
        mock_synadia_agents.load_context_options.return_value = {
            "servers": ["nats://prod:4222"],
            "user_credentials": "/secret/creds",
        }
        adapter = NatsAdapter(
            PlatformConfig(
                enabled=True,
                extra={"context": "prod-nats", "owner": "rene", "session_name": "default"},
            )
        )
        await adapter.connect()

        mock_synadia_agents.load_context_options.assert_called_once_with("prod-nats")
        mock_nats.connect.assert_awaited_once()
        assert mock_nats.connect.await_args.kwargs == {
            "servers": ["nats://prod:4222"],
            "user_credentials": "/secret/creds",
        }

    @pytest.mark.asyncio
    async def test_connect_constructs_service_with_full_settings(
        self, mock_synadia_agents, mock_nats, lock_granted
    ):
        adapter = _build_adapter(
            agent="hermes",
            owner="acme",
            session_name="prod-1",
            heartbeat_interval_s=15,
            max_payload="2MB",
            attachments_ok=False,
        )
        await adapter.connect()

        mock_synadia_agents.AgentService.assert_called_once()
        kwargs = mock_synadia_agents.AgentService.call_args.kwargs
        assert kwargs["agent"] == "hermes"
        assert kwargs["owner"] == "acme"
        assert kwargs["session_name"] == "prod-1"
        assert kwargs["nc"] is mock_nats.connect.return_value
        assert kwargs["heartbeat_interval_s"] == 15
        assert kwargs["max_payload"] == "2MB"
        assert kwargs["attachments_ok"] is False
        # v0.3: AgentService no longer accepts a separate ``session`` kwarg.
        assert "session" not in kwargs

    @pytest.mark.asyncio
    async def test_connect_derives_max_payload_from_broker_when_unset(
        self, mock_synadia_agents, mock_nats, lock_granted
    ):
        # PR #41 alignment: when config.extra.max_payload is omitted,
        # the adapter must read nc.max_payload (the broker's negotiated
        # INFO value) and pass that into AgentService, so a 64MB broker
        # isn't capped at 1MB by hermes itself.
        mock_nats.connect.return_value.max_payload = 8 * 1024 * 1024

        adapter = _build_adapter()  # no max_payload override
        await adapter.connect()

        kwargs = mock_synadia_agents.AgentService.call_args.kwargs
        assert kwargs["max_payload"] == "8MB"

    @pytest.mark.asyncio
    async def test_connect_passes_user_max_payload_through_unchanged(
        self, mock_synadia_agents, mock_nats, lock_granted
    ):
        # When the user explicitly sets max_payload, hermes forwards it
        # untouched. The SDK clamps down at start() if the value is
        # larger than the broker — that's the SDK's job, unit-tested
        # upstream, not re-proven here.
        mock_nats.connect.return_value.max_payload = 64 * 1024 * 1024
        adapter = _build_adapter(max_payload="512KB")
        await adapter.connect()

        kwargs = mock_synadia_agents.AgentService.call_args.kwargs
        assert kwargs["max_payload"] == "512KB"

    @pytest.mark.asyncio
    async def test_connect_falls_back_to_1mb_when_broker_reports_zero(
        self, mock_synadia_agents, mock_nats, lock_granted
    ):
        # Old nats-py builds didn't surface max_payload from the INFO
        # frame; the field defaults to 0. Match the SDK's own fallback
        # path so we don't try to format "0B".
        mock_nats.connect.return_value.max_payload = 0
        adapter = _build_adapter()
        await adapter.connect()

        kwargs = mock_synadia_agents.AgentService.call_args.kwargs
        assert kwargs["max_payload"] == "1MB"

    @pytest.mark.asyncio
    async def test_connect_registers_prompt_handler_before_start(
        self, mock_synadia_agents, mock_nats, lock_granted
    ):
        adapter = _build_adapter()
        await adapter.connect()

        service = mock_synadia_agents.AgentService.return_value
        # ``on_prompt`` is mandatory before ``start()`` per the SDK —
        # if we ever reordered these, start() would raise at runtime with
        # an unhelpful message.
        service.on_prompt.assert_called_once()
        passed_handler = service.on_prompt.call_args.args[0]
        assert passed_handler == adapter._on_prompt

        service.start.assert_awaited_once()

        # Method-call order: on_prompt → start.
        all_calls = service.mock_calls
        on_prompt_idx = next(
            i for i, c in enumerate(all_calls) if c == call.on_prompt(passed_handler)
        )
        start_idx = next(i for i, c in enumerate(all_calls) if c == call.start())
        assert on_prompt_idx < start_idx

    @pytest.mark.asyncio
    async def test_connect_acquires_scope_lock_with_identity(
        self, mock_synadia_agents, mock_nats, lock_granted
    ):
        adapter = _build_adapter(agent="hermes", owner="rene", session_name="default")
        await adapter.connect()

        assert lock_granted["acquires"] == [
            ("nats", "hermes:rene:default", {"platform": "nats"})
        ]


# ---------------------------------------------------------------------------
# Fatal-after-init
# ---------------------------------------------------------------------------


class TestConnectWithFatalInit:
    @pytest.mark.asyncio
    async def test_connect_short_circuits_when_config_was_invalid(
        self, mock_synadia_agents, mock_nats, lock_granted
    ):
        adapter = NatsAdapter(PlatformConfig(enabled=True, extra={"owner": "rene"}))
        # _init_ already set a non-retryable fatal error.
        assert adapter.has_fatal_error is True
        assert adapter.fatal_error_retryable is False

        assert await adapter.connect() is False
        # Must not touch the SDK or the lock table.
        mock_nats.connect.assert_not_called()
        mock_synadia_agents.AgentService.assert_not_called()
        assert lock_granted["acquires"] == []


# ---------------------------------------------------------------------------
# Lock conflict
# ---------------------------------------------------------------------------


class TestLockConflict:
    @pytest.mark.asyncio
    async def test_conflict_reports_fatal_nonretryable_and_skips_sdk(
        self, mock_synadia_agents, mock_nats, monkeypatch
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
        mock_nats.connect.assert_not_called()
        mock_synadia_agents.AgentService.assert_not_called()
        # And release_scoped_lock must not have been called — we never
        # owned the lock in the first place, so releasing would nuke the
        # *other* process's record.
        assert released == []


# ---------------------------------------------------------------------------
# Exceptions during connect
# ---------------------------------------------------------------------------


class TestConnectFailurePaths:
    @pytest.mark.asyncio
    async def test_nats_connect_failure_marks_retryable_and_releases_lock(
        self, mock_synadia_agents, mock_nats, lock_granted
    ):
        mock_nats.connect.side_effect = RuntimeError("boom")

        adapter = _build_adapter()
        ok = await adapter.connect()

        assert ok is False
        assert adapter.has_fatal_error is True
        assert adapter.fatal_error_code == "nats_connect_error"
        assert adapter.fatal_error_retryable is True
        assert "boom" in adapter.fatal_error_message
        # Lock must be returned on failure — otherwise the next retry
        # attempt would self-conflict on the very same process.
        assert lock_granted["releases"] == [("nats", "hermes:rene:default")]
        # No dangling service handle (we never even got to AgentService()).
        assert adapter._service is None
        assert adapter._nc is None

    @pytest.mark.asyncio
    async def test_service_construction_failure_releases_and_closes_nc(
        self, mock_synadia_agents, mock_nats, lock_granted
    ):
        # nc connects fine, but AgentService(...) raises — common case
        # when the SDK's AgentSubject.new() rejects a sanitized but still
        # invalid owner/session_name combo.
        mock_synadia_agents.AgentService.side_effect = ValueError("bad subject")

        adapter = _build_adapter()
        ok = await adapter.connect()

        assert ok is False
        assert adapter.fatal_error_code == "nats_connect_error"
        assert adapter.fatal_error_retryable is True
        # Partial-init nc handle was closed during teardown.
        mock_nats.connect.return_value.close.assert_awaited_once()
        assert adapter._nc is None
        assert adapter._service is None
        assert lock_granted["releases"] == [("nats", "hermes:rene:default")]

    @pytest.mark.asyncio
    async def test_service_start_failure_stops_service_and_closes_nc(
        self, mock_synadia_agents, mock_nats, lock_granted
    ):
        service = mock_synadia_agents.AgentService.return_value
        service.start.side_effect = RuntimeError("start failed")

        adapter = _build_adapter()
        ok = await adapter.connect()

        assert ok is False
        assert adapter.fatal_error_code == "nats_connect_error"
        assert adapter.fatal_error_retryable is True
        # Teardown must run stop() before close() — heartbeat publisher
        # needs a live nc to finalize, and closing nc first would surface
        # noisy "connection closed" warnings from the heartbeat loop.
        service.stop.assert_awaited_once()
        mock_nats.connect.return_value.close.assert_awaited_once()
        assert adapter._service is None
        assert adapter._nc is None
        assert lock_granted["releases"] == [("nats", "hermes:rene:default")]


# ---------------------------------------------------------------------------
# Disconnect
# ---------------------------------------------------------------------------


class TestDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect_after_successful_connect_tears_down_in_order(
        self, mock_synadia_agents, mock_nats, lock_granted
    ):
        adapter = _build_adapter()
        await adapter.connect()

        service = mock_synadia_agents.AgentService.return_value
        nc = mock_nats.connect.return_value

        # Strict ordering: service.stop() must run before nc.close() so
        # the heartbeat loop can exit on a live connection instead of
        # racing the socket close. Record the call order via side_effect
        # lambdas rather than inspecting mock_calls — the latter only
        # captures attribute access per-mock, so cross-mock ordering
        # needs a shared recorder.
        call_order: list[str] = []
        service.stop.side_effect = lambda: call_order.append("stop")
        nc.close.side_effect = lambda: call_order.append("close")

        await adapter.disconnect()

        assert call_order == ["stop", "close"]
        service.stop.assert_awaited_once()
        nc.close.assert_awaited_once()
        assert adapter._service is None
        assert adapter._nc is None
        assert adapter.is_connected is False
        # Lock must be returned so the same profile can reconnect later.
        assert lock_granted["releases"] == [("nats", "hermes:rene:default")]

    @pytest.mark.asyncio
    async def test_disconnect_is_idempotent_after_connect(
        self, mock_synadia_agents, mock_nats, lock_granted
    ):
        adapter = _build_adapter()
        await adapter.connect()
        await adapter.disconnect()
        await adapter.disconnect()  # second call must not blow up

        # stop() / close() still called exactly once — the second
        # disconnect finds nothing to stop because the first already
        # dropped the handles.
        assert mock_synadia_agents.AgentService.return_value.stop.await_count == 1
        assert mock_nats.connect.return_value.close.await_count == 1

    @pytest.mark.asyncio
    async def test_disconnect_without_connect_is_safe_noop(
        self, mock_synadia_agents, mock_nats, lock_granted
    ):
        adapter = _build_adapter()
        await adapter.disconnect()

        # Never called ``connect()``, so the SDK objects should never have
        # been built — and teardown should tolerate that gracefully.
        mock_nats.connect.assert_not_called()
        mock_synadia_agents.AgentService.assert_not_called()
        assert adapter.is_connected is False
        # No lock was acquired, so nothing to release.
        assert lock_granted["releases"] == []

    @pytest.mark.asyncio
    async def test_disconnect_tolerates_service_stop_errors(
        self, mock_synadia_agents, mock_nats, lock_granted
    ):
        adapter = _build_adapter()
        await adapter.connect()

        mock_synadia_agents.AgentService.return_value.stop.side_effect = RuntimeError("late")
        # Must not raise — gateway shutdown runs this in a loop over all
        # adapters and one raising aborts the shutdown of every platform
        # after it.
        await adapter.disconnect()

        # nc still closed; adapter handles cleared; lock still released.
        mock_nats.connect.return_value.close.assert_awaited_once()
        assert adapter._service is None
        assert adapter._nc is None
        assert lock_granted["releases"] == [("nats", "hermes:rene:default")]

    @pytest.mark.asyncio
    async def test_disconnect_cancels_in_flight_handlers(
        self, mock_synadia_agents, mock_nats, lock_granted
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
        mock_synadia_agents.AgentService.return_value.stop.assert_awaited_once()
        mock_nats.connect.return_value.close.assert_awaited_once()
        assert lock_granted["releases"] == [("nats", "hermes:rene:default")]

    @pytest.mark.asyncio
    async def test_disconnect_sets_shutdown_event_before_stop(
        self, mock_synadia_agents, mock_nats, lock_granted
    ):
        # Phase 4 handlers will gate their streaming loops on
        # ``self._shutdown_event`` — verify the event is set BEFORE the
        # service is stopped, so a handler checking the event between
        # deltas sees the shutdown signal before the SDK deregisters
        # the endpoint underneath it.
        adapter = _build_adapter()
        await adapter.connect()

        observed: dict[str, bool] = {}

        def _record_state():
            observed["shutdown_event_set_at_stop"] = adapter._shutdown_event.is_set()

        mock_synadia_agents.AgentService.return_value.stop.side_effect = _record_state

        await adapter.disconnect()

        assert observed["shutdown_event_set_at_stop"] is True

    @pytest.mark.asyncio
    async def test_connect_rebuilds_session_lock_after_teardown(
        self, mock_synadia_agents, mock_nats, lock_granted
    ):
        # The single-session lock collapses the v0.2 per-chat_id Lock
        # pool: a Lock held by a cancelled task wouldn't release cleanly,
        # so ``connect()`` must rebuild from scratch on each attempt.
        adapter = _build_adapter()
        await adapter.connect()
        first_lock = adapter._session_lock

        await adapter.disconnect()
        await adapter.connect()

        assert adapter._session_lock is not first_lock

    @pytest.mark.asyncio
    async def test_connect_clears_shutdown_event_on_retry(
        self, mock_synadia_agents, mock_nats, lock_granted
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
    def test_adapter_reports_nats_platform(self, mock_synadia_agents, lock_granted):
        adapter = _build_adapter()
        assert adapter.platform is Platform.NATS
