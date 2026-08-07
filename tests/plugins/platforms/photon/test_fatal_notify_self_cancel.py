"""Photon's fatal-error notification must not be cancelled by its own teardown.

`_monitor_sidecar_health` and `_supervise_sidecar` run as long-lived tasks on
the adapter. When one of them detects a fatal condition, the gateway's handler
tears the adapter down, and `disconnect()` cancels `_sidecar_health_task` and
awaits it. If the notification is awaited inline on the health task's own call
stack, `disconnect()` ends up cancelling its own ultimate caller: the
`task is not asyncio.current_task()` guard in `disconnect()` compares against
the wrapper task the gateway created around `disconnect()`, not the health task
several plain-await frames further up, so the guard passes and the cancel lands.

`CancelledError` stopped subclassing `Exception` in Python 3.8, so the
`except Exception` that used to wrap the inline notify call never saw it. The
health task died silently mid-handoff -- no log line, no retry -- and the
platform stayed stranded until someone restarted the gateway by hand.

PR #69112 hardened the shared dispatch path in `gateway/run.py` against a
cancelled *caller*. These tests cover the Photon-specific self-cancellation one
layer down, which that PR's scope could not reach.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.photon.adapter import PhotonAdapter


def _make_adapter(monkeypatch: pytest.MonkeyPatch) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    return PhotonAdapter(PlatformConfig(enabled=True, token="", extra={}))


class TestFatalNotifyIsDetached:
    """The notification must outlive cancellation of the task that raised it."""

    @pytest.mark.asyncio
    async def test_health_task_cancellation_does_not_kill_notification(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fatal handler that cancels the health task (exactly what
        ``disconnect()`` does) must still see the notification delivered."""
        adapter = _make_adapter(monkeypatch)
        adapter._inbound_running = True
        adapter._sidecar_health_interval = 0
        delivered = asyncio.Event()

        async def fake_notify() -> None:
            # Mirror the real handler: tear the adapter down, cancelling the
            # health task, then finish the handoff.
            await adapter.disconnect()
            delivered.set()

        monkeypatch.setattr(adapter, "_notify_fatal_error", fake_notify)
        monkeypatch.setattr(adapter, "_stop_sidecar", lambda: _noop())

        async def degraded(_path: str, _payload: Dict[str, Any]) -> Dict[str, Any]:
            return {"stream": {"ok": False, "state": "degraded", "degradedForMs": 4000,
                               "lastIssue": "stream persistently failing"}}

        monkeypatch.setattr(adapter, "_sidecar_call", degraded)

        health = asyncio.create_task(adapter._monitor_sidecar_health())
        adapter._sidecar_health_task = health

        await asyncio.wait_for(delivered.wait(), timeout=5.0)
        assert adapter.has_fatal_error
        assert adapter.fatal_error_code == "UPSTREAM_STREAM_DEGRADED"
        assert adapter.fatal_error_retryable

        # With the dispatch detached, the health task reaches its own `break`
        # and returns cleanly instead of being cancelled out from under the
        # handoff. Either way it must not die with an unhandled exception --
        # that was the silent failure that stranded the platform.
        await asyncio.wait_for(asyncio.shield(health), timeout=2.0)
        assert health.done()
        if not health.cancelled():
            assert health.exception() is None

    @pytest.mark.asyncio
    async def test_dispatch_does_not_await_on_caller_stack(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``_dispatch_fatal_notification`` must return without awaiting, so a
        cancel aimed at the calling task cannot reach the handoff."""
        adapter = _make_adapter(monkeypatch)
        started = asyncio.Event()
        finished = asyncio.Event()

        async def slow_notify() -> None:
            started.set()
            await asyncio.sleep(0.05)
            finished.set()

        monkeypatch.setattr(adapter, "_notify_fatal_error", slow_notify)

        async def caller() -> None:
            adapter._dispatch_fatal_notification()  # must not block

        task = asyncio.create_task(caller())
        await task  # returns immediately even though notify sleeps
        await asyncio.wait_for(started.wait(), timeout=2.0)

        task.cancel()  # cancelling the caller must not touch the notification
        await asyncio.wait_for(finished.wait(), timeout=2.0)

    @pytest.mark.asyncio
    async def test_notification_failure_is_logged_not_raised(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A failing notification must warn rather than surface as an
        unretrieved task exception."""
        adapter = _make_adapter(monkeypatch)

        async def boom() -> None:
            raise RuntimeError("gateway unreachable")

        monkeypatch.setattr(adapter, "_notify_fatal_error", boom)

        with caplog.at_level("WARNING"):
            await adapter._notify_fatal_error_logged()

        assert "fatal-error notification failed" in caplog.text


class TestBothCallSitesDetached:
    """Neither fatal path may await the notification inline."""

    def test_no_inline_notify_awaits_remain(self) -> None:
        """Guard against a future edit reintroducing the inline await."""
        import inspect

        from plugins.platforms.photon import adapter as photon_adapter

        for name in ("_monitor_sidecar_health", "_supervise_sidecar"):
            src = inspect.getsource(getattr(photon_adapter.PhotonAdapter, name))
            assert "await self._notify_fatal_error()" not in src, (
                f"{name} awaits _notify_fatal_error inline; use "
                f"_dispatch_fatal_notification() so disconnect() cannot cancel "
                f"its own caller"
            )
            assert "_dispatch_fatal_notification()" in src


async def _noop() -> None:
    return None
