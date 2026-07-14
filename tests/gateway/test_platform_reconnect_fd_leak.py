"""Regression tests for the gateway platform fd-leak fix (#37011).

Without an explicit ``disconnect()`` on adapters that fail to connect in
the reconnect watcher, every retry leaks the resources the adapter
opened in ``__init__`` — for ``APIServerAdapter`` that means 2 file
descriptors per attempt (the SQLite ``response_store.db`` and its WAL
sidecar). At the 300s backoff cap that's ~12 fds/hour; the default
2560-fd ulimit is exhausted in ~12h of continuous failure, after which
the gateway raises ``OSError: [Errno 24] Too many open files`` on
every ``open()`` and becomes a zombie.

These tests pin all three failure paths in
``_platform_reconnect_watcher`` (non-retryable error, retryable error,
exception during connect) to call ``adapter.disconnect()`` on the
unowned adapter, plus the path-level ``APIServerAdapter.disconnect()``
behavior of also closing the ``ResponseStore``. The pre-fix
implementation did not call ``disconnect()`` on any of these paths;
this file would have caught the regression and now pins the fix.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, ResponseStore
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.run import GatewayRunner, _dispose_unused_adapter


def _make_runner() -> GatewayRunner:
    """Create a minimal GatewayRunner via object.__new__ to skip __init__.

    Mirrors the helper in test_platform_reconnect.py so this file
    is drop-in compatible with the existing reconnect test suite.
    """
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="test")}
    )
    runner._running = True
    runner._shutdown_event = asyncio.Event()
    runner._exit_reason = None
    runner._exit_with_failure = False
    runner._exit_cleanly = False
    runner._failed_platforms = {}
    runner.adapters = {}
    return runner


async def _run_watcher_one_iteration(runner: GatewayRunner) -> None:
    """Drive ``_platform_reconnect_watcher`` for exactly one retry pass.

    Patches ``asyncio.sleep`` to advance the watcher's internal
    ``await asyncio.sleep(10)`` initial delay and the 1-second inner
    sleeps without actually waiting. Mirrors the pattern used in
    ``test_platform_reconnect.py::TestPlatformReconnectWatcher``.
    """
    real_sleep = asyncio.sleep
    call_count = 0

    async def fake_sleep(_n: float) -> None:
        nonlocal call_count
        call_count += 1
        if call_count > 2:
            # Two sleeps is enough to get past the initial 10s wait
            # and the first inner-tick check. After that, stop the
            # watcher so the test returns.
            runner._running = False
        await real_sleep(0)

    with patch("asyncio.sleep", side_effect=fake_sleep):
        await runner._platform_reconnect_watcher()


class _CountingAdapter(BasePlatformAdapter):
    """Adapter that records every disconnect() call for fd-leak assertions.

    The base ``BasePlatformAdapter.disconnect()`` is a no-op by default
    for stub adapters, which is why the pre-fix reconnect watcher
    silently leaked: the would-be dispose calls were happening on
    objects that did nothing on disconnect. This stub mimics the real
    ``APIServerAdapter`` shape — every constructor call opens 2 fds
    (the SQLite db + WAL), and every disconnect() must close them.
    """

    def __init__(self, *, succeed: bool = False, fatal_error: str | None = None,
                 fatal_retryable: bool = True, raise_during_connect: bool = False):
        super().__init__(PlatformConfig(enabled=True, token="t"), Platform.TELEGRAM)
        # 2 fds to track: the canonical "ResponseStore" pair. The
        # reconnect watcher should call disconnect() once per
        # construction; otherwise these stay open and contribute to
        # the gateway-wide fd count.
        self._open_fds = 2
        self._disconnect_calls = 0
        self._succeed = succeed
        self._fatal_error = fatal_error
        self._fatal_retryable = fatal_retryable
        self._raise_during_connect = raise_during_connect

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if self._raise_during_connect:
            raise RuntimeError("simulated connect exception")
        if self._fatal_error:
            self._set_fatal_error(
                "test_code", self._fatal_error, retryable=self._fatal_retryable,
            )
            return False
        return self._succeed

    async def disconnect(self) -> None:
        self._disconnect_calls += 1
        self._open_fds = 0  # fd release on dispose

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="1")

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _seed_runner_with_one_failure(runner: GatewayRunner) -> None:
    """Queue a single platform for the reconnect watcher to pick up."""
    runner._failed_platforms[Platform.TELEGRAM] = {
        "config": PlatformConfig(enabled=True, token="t"),
        "attempts": 0,
        "next_retry": time.monotonic() - 1,  # eligible immediately
    }


class TestReconnectFDLeakRegression:
    """All three reconnect failure paths must dispose the unowned adapter.

    The pre-fix implementation constructed a fresh adapter on every
    retry and dropped it on the floor when connect() failed. That leaks
    2 fds per retry (for ``APIServerAdapter``) at the 300s backoff cap,
    exhausting the 2560-fd ulimit in ~12h of continuous failure (#37011).
    """

    @pytest.mark.asyncio
    async def test_nonretryable_failure_disposes_unowned_adapter(self):
        """A fatal error (bad auth, etc.) must call disconnect() exactly once.

        The adapter failed to connect and is being removed from the
        retry queue. Nothing else owns it, so the watcher is the only
        code with a chance to call disconnect() — and disconnect() is
        the only place the SQLite fds get closed. One retry, one
        dispose, no leak.
        """
        runner = _make_runner()
        _seed_runner_with_one_failure(runner)
        adapter = _CountingAdapter(
            succeed=False, fatal_error="bad token", fatal_retryable=False,
        )
        with patch.object(runner, "_create_adapter", return_value=adapter), \
             patch.object(runner, "_connect_adapter_with_timeout",
                          new=AsyncMock(return_value=False)):
            await _run_watcher_one_iteration(runner)

        # The intent of this test is "the watcher calls disconnect()
        # exactly once on the unowned adapter" — not "at least once".
        # An accidental double-dispose would be a new bug to catch
        # (e.g. the watcher's two failure paths both calling dispose
        # for the same adapter instance). Tighten to == 1.
        assert adapter._disconnect_calls == 1, (
            f"non-retryable reconnect failure must call adapter.disconnect() "
            f"exactly once; got {adapter._disconnect_calls} calls. "
            "Without it, 2 fds leak per retry at the 300s backoff cap "
            "(#37011). More than one call would also be a bug — the "
            "adapter has already been disposed once, a second call is "
            "wasted work and may itself raise."
        )
        assert adapter._open_fds == 0, (
            f"adapter fds not released after disconnect(); "
            f"{adapter._open_fds} still open. This is the fd leak #37011."
        )

    @pytest.mark.asyncio
    async def test_retryable_failure_disposes_unowned_adapter(self):
        """A retryable failure (network blip) must also call disconnect().

        This is the path that fires most often in production: a
        transient DNS resolution failure or upstream outage, which
        back-offs to 300s and retries indefinitely. The watcher
        tracks ``info["attempts"]`` and reschedules, but the failed
        adapter is still dropped on the floor without dispose.
        """
        runner = _make_runner()
        _seed_runner_with_one_failure(runner)
        adapter = _CountingAdapter(
            succeed=False, fatal_error="dns timeout", fatal_retryable=True,
        )
        with patch.object(runner, "_create_adapter", return_value=adapter), \
             patch.object(runner, "_connect_adapter_with_timeout",
                          new=AsyncMock(return_value=False)):
            await _run_watcher_one_iteration(runner)

        assert adapter._disconnect_calls >= 1, (
            f"retryable reconnect failure must call adapter.disconnect(); "
            f"got {adapter._disconnect_calls} calls. This is the hot path "
            "for the fd leak in #37011."
        )

    @pytest.mark.asyncio
    async def test_exception_during_connect_disposes_unowned_adapter(self):
        """An exception escaping connect() (aiohttp start crash, etc.) disposes.

        The ``except Exception`` arm in the watcher used to skip the
        dispose call entirely. Pre-fix, this leaked the same 2 fds
        per retry as the other two branches.
        """
        runner = _make_runner()
        _seed_runner_with_one_failure(runner)
        adapter = _CountingAdapter(raise_during_connect=True)
        with patch.object(runner, "_create_adapter", return_value=adapter), \
             patch.object(runner, "_connect_adapter_with_timeout",
                          new=AsyncMock(side_effect=RuntimeError("boom"))):
            await _run_watcher_one_iteration(runner)

        assert adapter._disconnect_calls >= 1, (
            f"exception-during-connect must call adapter.disconnect(); "
            f"got {adapter._disconnect_calls} calls. The except-arm of the "
            "reconnect watcher is one of the three leak paths in #37011."
        )

    @pytest.mark.asyncio
    async def test_dispose_helper_handles_none(self):
        """``_dispose_unused_adapter(None)`` is a no-op (defensive)."""
        await _dispose_unused_adapter(None)  # must not raise

    @pytest.mark.asyncio
    async def test_dispose_helper_swallows_disconnect_exception(self):
        """A disconnect() that itself raises must not abort the watcher loop.

        Half-constructed adapters can raise from disconnect() because
        some of their __init__ state is missing. The watcher loop
        would then die and stop retrying, masking the original
        configuration error as a hard crash.
        """
        disconnect_calls = 0

        class _RaisingAdapter(BasePlatformAdapter):
            def __init__(self):
                super().__init__(
                    PlatformConfig(enabled=True, token="t"),
                    Platform.TELEGRAM,
                )

            async def connect(self, *, is_reconnect: bool = False) -> bool:
                return True

            async def disconnect(self) -> None:
                nonlocal disconnect_calls
                disconnect_calls += 1
                raise RuntimeError("half-constructed; aiohttp app never started")

            async def send(self, chat_id, content, reply_to=None, metadata=None):
                return SendResult(success=True, message_id="1")

            async def send_typing(self, chat_id, metadata=None):
                return None

            async def get_chat_info(self, chat_id):
                return {"id": chat_id}

        await _dispose_unused_adapter(_RaisingAdapter())  # must not raise
        assert disconnect_calls == 1


class TestAPIServerDisconnectClosesResponseStore:
    """The platform-level fix: ``APIServerAdapter.disconnect()`` must close its ResponseStore.

    Without this, the reconnect watcher's dispose call (see the
    test class above) is a no-op for ``APIServerAdapter`` — the
    aiohttp web server stops, but the SQLite ``ResponseStore``
    connection stays open. The DB file plus its WAL sidecar = 2 fds,
    which is the headline leak in #37011.
    """

    def _build_adapter_with_store(self, store: ResponseStore) -> APIServerAdapter:
        """Build an APIServerAdapter with the required internal state.

        We bypass ``__init__`` (which would try to start aiohttp
        immediately) and set just the fields ``disconnect()`` reads.
        """
        adapter = APIServerAdapter.__new__(APIServerAdapter)
        adapter._mark_disconnected = lambda: None  # type: ignore[method-assign]
        adapter._site = None
        adapter._runner = None
        adapter._app = None
        adapter._response_store = store
        adapter.platform = Platform.API_SERVER
        return adapter

    @pytest.mark.asyncio
    async def test_disconnect_closes_response_store(self, tmp_path):
        """Closing the adapter's ResponseStore releases its SQLite connection.

        We point the ``ResponseStore`` at a tmp db so we can verify
        its ``close()`` is called by ``APIServerAdapter.disconnect()``.
        The real ``ResponseStore.__init__`` opens a SQLite connection
        to ``~/.hermes/response_store.db`` (or :memory: as a fallback),
        which is exactly the resource that was leaking pre-fix.
        """
        import sqlite3

        store = ResponseStore(max_size=10, db_path=str(tmp_path / "rs.db"))
        adapter = self._build_adapter_with_store(store)

        await adapter.disconnect()

        # Post-disconnect, the underlying sqlite3 conn should be closed.
        # sqlite3 raises ``ProgrammingError: Cannot operate on a closed
        # database`` for any further operation. We assert on the
        # specific exception type (not bare ``Exception``) so the test
        # only passes when the close actually took effect — a generic
        # ``Exception`` catcher would mask unrelated failures (env
        # issues, AttributeError, etc.).
        with pytest.raises(sqlite3.ProgrammingError):
            store._conn.execute("SELECT 1").fetchone()

    @pytest.mark.asyncio
    async def test_disconnect_swallows_response_store_close_exception(self, tmp_path):
        """A misbehaving ResponseStore.close() must not abort adapter shutdown.

        Real-world failure mode: the SQLite file was unlinked out
        from under us (operator rm'd ``response_store.db`` during a
        disk pressure event). ``close()`` raises. The watcher must
        continue with the aiohttp shutdown, not bail.
        """
        store = ResponseStore(max_size=10, db_path=str(tmp_path / "rs.db"))

        def _boom() -> None:
            raise RuntimeError("sqlite file vanished")

        store.close = _boom  # type: ignore[method-assign]
        adapter = self._build_adapter_with_store(store)

        # Must not raise — disconnect() swallows the close error and
        # continues to the aiohttp teardown (no-op here since we
        # bypassed __init__).
        await adapter.disconnect()
