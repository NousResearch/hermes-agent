"""Photon adapter resilience to transient Spectrum/Envoy upstream overflow.

Covers the three behaviors that let the adapter ride through a Photon
"reset reason: overflow" event instead of degrading delivery and silently
dying (issue #50185):

  1. ``_is_retryable_error`` classifies the Envoy/sidecar overflow strings as
     retryable so ``_send_with_retry`` actually engages its backoff loop.
  2. ``send_typing`` is rate-gated per chat, and ``stop_typing`` resets the
     gate so the next turn's typing indicator fires immediately.
  3. ``_supervise_sidecar`` detects an unexpected sidecar exit and raises a
     ``retryable=True`` fatal so the gateway reconnect watcher revives the
     platform — instead of returning silently and leaving ``_inbound_loop``
     spinning against a dead port.
  4. ``_monitor_sidecar_health`` promotes degraded upstream stream health
     reported by ``/healthz`` into the same retryable reconnect path.

No Node sidecar is spawned and no ports are bound.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.photon.adapter import PhotonAdapter


def _make_adapter(monkeypatch: pytest.MonkeyPatch) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    cfg = PlatformConfig(enabled=True, token="", extra={})
    return PhotonAdapter(cfg)


# -- Gap 1: retryable classification of overflow errors ---------------------

@pytest.mark.parametrize(
    "error",
    [
        "UNAVAILABLE: internal sidecar error",
        "upstream connect error or disconnect/reset before headers",
        "reset reason: overflow",
        # Case-insensitive: real strings arrive with mixed case.
        "Internal Sidecar Error",
    ],
)
def test_overflow_strings_classified_retryable(error: str) -> None:
    assert PhotonAdapter._is_retryable_error(error) is True


def test_unrelated_error_not_retryable() -> None:
    # A genuine permanent failure must NOT be retried.
    assert PhotonAdapter._is_retryable_error("400 bad request: invalid spaceId") is False
    assert PhotonAdapter._is_retryable_error(None) is False


def test_base_network_patterns_still_match() -> None:
    # The override delegates to the base classifier first, so generic
    # network strings keep working.
    assert PhotonAdapter._is_retryable_error("ConnectError: connection refused") is True


# -- Gap 2: typing-indicator cooldown ---------------------------------------

@pytest.mark.asyncio
async def test_typing_cooldown_suppresses_rapid_repeats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    calls: list[Dict[str, Any]] = []

    async def _fake_call(path: str, payload: Dict[str, Any]) -> Any:
        calls.append(payload)
        return {"ok": True}

    monkeypatch.setattr(adapter, "_sidecar_call", _fake_call)

    # First call fires; immediate repeats are suppressed by the cooldown.
    await adapter.send_typing("chat-1")
    await adapter.send_typing("chat-1")
    await adapter.send_typing("chat-1")

    assert len(calls) == 1


@pytest.mark.asyncio
async def test_typing_cooldown_is_per_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    calls: list[str] = []

    async def _fake_call(path: str, payload: Dict[str, Any]) -> Any:
        calls.append(payload["spaceId"])
        return {"ok": True}

    monkeypatch.setattr(adapter, "_sidecar_call", _fake_call)

    # Different chats have independent cooldowns.
    await adapter.send_typing("chat-1")
    await adapter.send_typing("chat-2")

    assert calls == ["chat-1", "chat-2"]


@pytest.mark.asyncio
async def test_stop_typing_resets_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    starts = 0

    async def _fake_call(path: str, payload: Dict[str, Any]) -> Any:
        nonlocal starts
        if payload.get("state") == "start":
            starts += 1
        return {"ok": True}

    monkeypatch.setattr(adapter, "_sidecar_call", _fake_call)

    # A start, then a stop (end of turn), then a start for the next turn must
    # fire immediately — the cooldown only suppresses rapid consecutive starts
    # without an intervening stop.
    await adapter.send_typing("chat-1")
    await adapter.stop_typing("chat-1")
    await adapter.send_typing("chat-1")

    assert starts == 2


# -- Gap 3: sidecar crash detection -----------------------------------------

class _EofStdout:
    """A proc.stdout whose readline() reports immediate EOF (dead sidecar)."""

    def readline(self) -> bytes:
        return b""


class _DeadProc:
    """Minimal subprocess.Popen stand-in for a sidecar that has exited."""

    def __init__(self, exit_code: int = 1) -> None:
        self.stdout = _EofStdout()
        self.stdin = None
        self._exit_code = exit_code

    def poll(self) -> int:
        return self._exit_code


@pytest.mark.asyncio
async def test_unexpected_sidecar_exit_raises_retryable_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    # Simulate a live session whose sidecar then dies underneath it.
    adapter._inbound_running = True

    notified: list[bool] = []

    async def _fake_notify() -> None:
        notified.append(True)

    monkeypatch.setattr(adapter, "_notify_fatal_error", _fake_notify)

    await adapter._supervise_sidecar(_DeadProc(exit_code=137))  # type: ignore[arg-type]

    assert adapter.has_fatal_error is True
    assert adapter.fatal_error_code == "SIDECAR_CRASHED"
    # retryable=True routes the platform into the reconnect watcher rather
    # than crashing the whole gateway.
    assert adapter.fatal_error_retryable is True
    assert adapter._running is False
    # The notification is dispatched onto its own task rather than awaited on
    # the supervisor's stack, so that disconnect() cancelling the supervisor
    # cannot kill the handoff. Let that task run before asserting delivery.
    await _drain_pending_tasks()
    assert notified == [True]


@pytest.mark.asyncio
async def test_clean_shutdown_does_not_raise_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    # disconnect() sets _inbound_running = False before stopping the sidecar,
    # so the detection block must NOT fire on a clean shutdown.
    adapter._inbound_running = False

    notified: list[bool] = []

    async def _fake_notify() -> None:
        notified.append(True)

    monkeypatch.setattr(adapter, "_notify_fatal_error", _fake_notify)

    await adapter._supervise_sidecar(_DeadProc(exit_code=0))  # type: ignore[arg-type]

    assert adapter.has_fatal_error is False
    assert notified == []


@pytest.mark.asyncio
async def test_degraded_stream_health_raises_retryable_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    adapter._inbound_running = True
    adapter._sidecar_health_interval = 0.0

    async def _fake_call(path: str, payload: Dict[str, Any]) -> Any:
        assert path == "/healthz"
        return {
            "ok": True,
            "stream": {
                "ok": False,
                "state": "degraded",
                "degradedForMs": 120000,
                "lastIssue": "[spectrum.stream] stream interrupted; reconnecting",
            },
        }

    notified: list[bool] = []

    async def _fake_notify() -> None:
        notified.append(True)
        adapter._inbound_running = False

    monkeypatch.setattr(adapter, "_sidecar_call", _fake_call)
    monkeypatch.setattr(adapter, "_notify_fatal_error", _fake_notify)

    await adapter._monitor_sidecar_health()

    assert adapter.has_fatal_error is True
    assert adapter.fatal_error_code == "UPSTREAM_STREAM_DEGRADED"
    assert adapter.fatal_error_retryable is True
    # Dispatched detached (see _dispatch_fatal_notification) so the health
    # task's own teardown cannot cancel the handoff; drain before asserting.
    await _drain_pending_tasks()
    assert notified == [True]


async def _drain_pending_tasks(limit: int = 50) -> None:
    """Let detached fatal-notification tasks finish before asserting on them.

    ``_dispatch_fatal_notification`` deliberately does not await the
    notification (that is what kept ``disconnect()`` from cancelling its own
    caller), so a test that drives ``_monitor_sidecar_health`` /
    ``_supervise_sidecar`` directly returns before the notification has run.
    """
    for _ in range(limit):
        pending = [
            t for t in asyncio.all_tasks()
            if t is not asyncio.current_task() and not t.done()
        ]
        if not pending:
            return
        await asyncio.wait(pending, timeout=1.0)


# -- Gap 5: self-cancellation race in _stop_sidecar() (issue #73159) --------
#
# The tests above mock out _notify_fatal_error() entirely, so the real
# integration chain (supervisor task -> _notify_fatal_error() ->
# disconnect() -> _stop_sidecar() -> cancel the supervisor task) is never
# exercised end to end. That chain is exactly where the bug lives: when
# _notify_fatal_error() is a real callback that calls adapter.disconnect(),
# _stop_sidecar() is invoked FROM WITHIN the currently-running supervisor
# task, and cancelling self._sidecar_supervisor_task there cancels the very
# task executing the fatal-error handler -- aborting it (via CancelledError,
# a BaseException the handler's `except Exception` guards don't catch)
# before the Gateway's reconnect-queue step ever runs.

class _FakeStoppedProc:
    """Minimal proc stand-in so _stop_sidecar() reaches its finally block
    (it early-returns entirely when self._sidecar_proc is None) without
    spawning a real subprocess or doing any real I/O."""

    def __init__(self) -> None:
        self.stdin = None

    def wait(self, timeout: float | None = None) -> int:
        return 0


@pytest.mark.asyncio
async def test_supervisor_task_survives_self_triggered_disconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real chain: _supervise_sidecar() (running as the actual
    self._sidecar_supervisor_task) detects the crash and calls a REAL
    _notify_fatal_error() that calls adapter.disconnect() -- which reaches
    _stop_sidecar() from inside the task it's about to try to cancel.

    Before the fix: this raises CancelledError out of _supervise_sidecar(),
    so the task ends up in the "cancelled" state and reconnect_queued below
    is never set (mirroring how the real Gateway's fatal-error handler,
    which runs the reconnect-queue logic AFTER disconnect() returns, never
    gets there either).

    After the fix: disconnect() completes normally, _supervise_sidecar()
    returns normally, and the task is NOT cancelled.
    """
    adapter = _make_adapter(monkeypatch)
    adapter._inbound_running = True
    # A minimal fake proc so _stop_sidecar() reaches its finally block
    # (the real process-management behavior is covered separately by
    # test_sidecar_lifecycle.py) -- this test is purely about the
    # self-cancellation race.
    adapter._sidecar_proc = _FakeStoppedProc()

    reconnect_queued: list[bool] = []

    async def _real_notify_fatal_error() -> None:
        # Stand-in for the Gateway's actual fatal-error handler: it calls
        # adapter.disconnect() (real chain: disconnect -> _stop_sidecar,
        # which used to self-cancel), then -- only if that completes
        # without the CancelledError escaping -- proceeds to queue the
        # platform for background reconnection.
        await adapter.disconnect()
        reconnect_queued.append(True)

    monkeypatch.setattr(adapter, "_notify_fatal_error", _real_notify_fatal_error)

    async def _run_supervisor():
        await adapter._supervise_sidecar(_DeadProc(exit_code=75))

    task = asyncio.ensure_future(_run_supervisor())
    adapter._sidecar_supervisor_task = task

    # Must complete cleanly -- must NOT raise CancelledError out to us.
    await task

    assert task.cancelled() is False, (
        "The supervisor task must not end up cancelled by its own "
        "fatal-error handling chain"
    )
    assert adapter.has_fatal_error is True
    assert adapter.fatal_error_code == "SIDECAR_CRASHED"
    assert reconnect_queued == [True], (
        "The reconnect-queue step (everything after disconnect() returns "
        "in the real Gateway handler) must actually run -- this is the "
        "exact step issue #73159 reports as silently skipped"
    )
    # _stop_sidecar() must have cleared the task reference either way.
    assert adapter._sidecar_supervisor_task is None


@pytest.mark.asyncio
async def test_external_disconnect_still_cancels_supervisor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The OTHER call path -- external cleanup (Gateway shutdown, an
    explicit /platform disconnect) -- calls _stop_sidecar() from a
    DIFFERENT task than the supervisor. That legitimate case must still
    cancel a still-running supervisor task exactly as before."""
    adapter = _make_adapter(monkeypatch)
    adapter._sidecar_proc = _FakeStoppedProc()

    supervisor_ran_forever = asyncio.Event()

    async def _hangs_forever():
        try:
            supervisor_ran_forever.set()
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            raise

    task = asyncio.ensure_future(_hangs_forever())
    adapter._sidecar_supervisor_task = task
    await supervisor_ran_forever.wait()

    # Called from THIS (different) task -- the external-cleanup case.
    await adapter._stop_sidecar()

    # cancel() only schedules the CancelledError; await it so the task
    # actually settles into the cancelled state before asserting.
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert task.cancelled() is True, (
        "External cleanup must still cancel a running supervisor task"
    )
    assert adapter._sidecar_supervisor_task is None
