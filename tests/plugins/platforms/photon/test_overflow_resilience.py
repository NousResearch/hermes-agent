"""Tests for Photon adapter overflow resilience (issue #50185).

Covers:
- Typing-indicator cooldown: repeated send_typing calls within the cooldown
  window are suppressed so they don't amplify an upstream overflow.
- Sidecar death detection: when the supervisor's stdout pipe closes (process
  died), the adapter marks the sidecar as dead so subsequent _sidecar_call
  attempts fail fast instead of hanging on a dead port.
"""
from __future__ import annotations

import asyncio
import io
from typing import Any, Dict, List

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.photon import adapter as photon_adapter
from plugins.platforms.photon.adapter import PhotonAdapter


def _make_adapter(monkeypatch: pytest.MonkeyPatch) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    cfg = PlatformConfig(enabled=True, token="", extra={})
    return PhotonAdapter(cfg)


# ---------------------------------------------------------------------------
# Layer 2 — typing-indicator cooldown
# ---------------------------------------------------------------------------

class TestTypingCooldown:
    @pytest.mark.asyncio
    async def test_rapid_typing_calls_are_suppressed(self, monkeypatch):
        """Two send_typing calls within the cooldown window → only one
        sidecar POST."""
        adapter = _make_adapter(monkeypatch)
        calls: List[tuple] = []

        async def fake_sidecar_call(path: str, body: Dict[str, Any]):
            calls.append((path, body))

        monkeypatch.setattr(adapter, "_sidecar_call", fake_sidecar_call)

        await adapter.send_typing("chat-1")
        await adapter.send_typing("chat-1")  # within cooldown → suppressed

        assert len(calls) == 1
        assert calls[0][0] == "/typing"

    @pytest.mark.asyncio
    async def test_different_chats_not_suppressed(self, monkeypatch):
        """Typing in different chats should not be throttled by each
        other's cooldown."""
        adapter = _make_adapter(monkeypatch)
        calls: List[tuple] = []

        async def fake_sidecar_call(path: str, body: Dict[str, Any]):
            calls.append((path, body))

        monkeypatch.setattr(adapter, "_sidecar_call", fake_sidecar_call)

        await adapter.send_typing("chat-A")
        await adapter.send_typing("chat-B")

        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_typing_resumes_after_cooldown(self, monkeypatch):
        """After the cooldown elapses, typing calls go through again."""
        adapter = _make_adapter(monkeypatch)
        calls: List[tuple] = []

        async def fake_sidecar_call(path: str, body: Dict[str, Any]):
            calls.append((path, body))

        monkeypatch.setattr(adapter, "_sidecar_call", fake_sidecar_call)

        # Patch time.time so we can simulate elapsed time.
        fake_now = [1000.0]
        monkeypatch.setattr(photon_adapter.time, "time", lambda: fake_now[0])

        await adapter.send_typing("chat-1")
        fake_now[0] += 10  # well past any cooldown
        await adapter.send_typing("chat-1")

        assert len(calls) == 2


# ---------------------------------------------------------------------------
# Layer 3 — sidecar death detection
# ---------------------------------------------------------------------------

class TestSidecarDeathDetection:
    @pytest.mark.asyncio
    async def test_supervisor_marks_sidecar_dead(self, monkeypatch):
        """When the sidecar process exits, _supervise_sidecar should set
        ``_sidecar_dead = True`` so downstream callers fail fast."""
        adapter = _make_adapter(monkeypatch)

        class _DeadProc:
            returncode = 1
            stdout = io.BytesIO(b"")  # empty → readline returns b"" → death

            @staticmethod
            def poll():
                return 1  # process has exited

        async def fake_run_in_executor(executor, func, *args):
            return func(*args)

        loop = type("L", (), {"run_in_executor": staticmethod(fake_run_in_executor)})()

        monkeypatch.setattr(
            "asyncio.get_event_loop", lambda: loop
        )

        await adapter._supervise_sidecar(_DeadProc())  # type: ignore[arg-type]

        assert adapter._sidecar_dead is True

    @pytest.mark.asyncio
    async def test_sidecar_call_fails_fast_when_dead(self, monkeypatch):
        """_sidecar_call should raise immediately when the sidecar is known
        to be dead — no HTTP attempt to a dead port."""
        adapter = _make_adapter(monkeypatch)
        adapter._sidecar_dead = True
        adapter._http_client = object()  # pretend we're connected

        with pytest.raises(RuntimeError, match="sidecar.*not running|sidecar.*dead"):
            await adapter._sidecar_call("/send", {"spaceId": "x", "text": "hi"})

    @pytest.mark.asyncio
    async def test_supervisor_healthy_does_not_mark_dead(self, monkeypatch):
        """If the supervisor task is cancelled (not a crash), the sidecar
        should NOT be marked dead."""
        adapter = _make_adapter(monkeypatch)

        class _AliveProc:
            returncode = None
            stdout = io.BytesIO(b"")  # will raise CancelledError on readline

            @staticmethod
            def poll():
                return None  # still running

        async def fake_run_in_executor(executor, func, *args):
            # Simulate the supervisor being cancelled mid-readline.
            raise asyncio.CancelledError()

        loop = type("L", (), {"run_in_executor": staticmethod(fake_run_in_executor)})()

        monkeypatch.setattr("asyncio.get_event_loop", lambda: loop)

        with pytest.raises(asyncio.CancelledError):
            await adapter._supervise_sidecar(_AliveProc())  # type: ignore[arg-type]

        assert adapter._sidecar_dead is False


# (asyncio already imported at top of file)
