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

No Node sidecar is spawned and no ports are bound.
"""
from __future__ import annotations

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


@pytest.mark.parametrize(
    "error",
    [
        "Photon sidecar /send returned 500 (auth_failed): photon authentication failed",
        "IMessageError: Invalid credentials",
        "PERMISSION_DENIED: [upstream] Authentication failed.",
    ],
)
def test_auth_failures_are_not_retryable(error: str) -> None:
    assert PhotonAdapter._is_retryable_error(error) is False


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
    assert notified == [True]


@pytest.mark.asyncio
async def test_sidecar_send_preserves_sidecar_retryable_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)

    async def _fake_call(path: str, payload: Dict[str, Any]) -> Any:
        from plugins.platforms.photon.adapter import PhotonSidecarError

        raise PhotonSidecarError(
            path,
            500,
            "photon upstream unavailable",
            code="upstream_unavailable",
            retryable=True,
        )

    monkeypatch.setattr(adapter, "_sidecar_call", _fake_call)

    result = await adapter.send("+15551234567", "hello")

    assert result.success is False
    assert result.retryable is True
    assert "upstream_unavailable" in (result.error or "")


@pytest.mark.asyncio
async def test_sidecar_send_auth_failure_is_permanent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)

    async def _fake_call(path: str, payload: Dict[str, Any]) -> Any:
        from plugins.platforms.photon.adapter import PhotonSidecarError

        raise PhotonSidecarError(
            path,
            500,
            "photon authentication failed",
            code="auth_failed",
            retryable=False,
        )

    monkeypatch.setattr(adapter, "_sidecar_call", _fake_call)

    result = await adapter.send("+15551234567", "hello")

    assert result.success is False
    assert result.retryable is False
    assert PhotonAdapter._is_retryable_error(result.error) is False


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
