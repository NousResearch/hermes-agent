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
  4. ``_monitor_sidecar_health`` promotes a degraded upstream stream reported
     by ``/healthz`` into the same retryable gateway reconnect path.

No Node sidecar is spawned and no ports are bound.
"""
from __future__ import annotations

import types
from pathlib import Path
from typing import Any, Dict

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.photon import adapter as photon_adapter
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


def test_project_policy_failures_are_not_retryable() -> None:
    assert (
        PhotonAdapter._is_retryable_error(
            "Photon sidecar /send returned 500 (target_not_allowed): "
            "target not allowed for this Photon project"
        )
        is False
    )


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
async def test_sidecar_send_project_policy_failure_is_forbidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)

    async def _fake_call(path: str, payload: Dict[str, Any]) -> Any:
        from plugins.platforms.photon.adapter import PhotonSidecarError

        raise PhotonSidecarError(
            path,
            500,
            "target not allowed for this Photon project",
            code="target_not_allowed",
            retryable=False,
        )

    monkeypatch.setattr(adapter, "_sidecar_call", _fake_call)

    result = await adapter.send("+15551234567", "hello")

    assert result.success is False
    assert result.retryable is False
    assert result.error_kind == "forbidden"
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

    monkeypatch.setattr(adapter, "_sidecar_call", _fake_call)
    monkeypatch.setattr(adapter, "_notify_fatal_error", _fake_notify)

    await adapter._monitor_sidecar_health()

    assert adapter.has_fatal_error is True
    assert adapter.fatal_error_code == "UPSTREAM_STREAM_DEGRADED"
    assert adapter.fatal_error_retryable is True
    assert adapter._running is False
    assert notified == [True]


# -- Gap 5: outbound sidecar send recovery ----------------------------------

class _FakeResponse:
    def __init__(
        self,
        status_code: int,
        payload: Dict[str, Any] | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self) -> Dict[str, Any]:
        return self._payload


class _FakeAsyncClient:
    calls: list[tuple[str, Dict[str, Any]]] = []
    responses: list[Any] = []

    def __init__(self, timeout: float | None = None) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def post(
        self,
        url: str,
        json: Dict[str, Any] | None = None,  # noqa: A002 - mirrors httpx
        headers: Dict[str, str] | None = None,  # noqa: ARG002
    ) -> _FakeResponse:
        self.calls.append((url, json or {}))
        if not self.responses:
            raise AssertionError("no fake response queued")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


async def _noop_sleep(delay: float) -> None:  # noqa: ARG001
    return None


@pytest.mark.asyncio
async def test_photon_standalone_send_retries_transient_sidecar_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(photon_adapter, "HTTPX_AVAILABLE", True)
    monkeypatch.setattr(
        photon_adapter,
        "httpx",
        types.SimpleNamespace(AsyncClient=_FakeAsyncClient),
    )
    monkeypatch.setenv("PHOTON_SIDECAR_TOKEN", "token")
    monkeypatch.setattr(photon_adapter.asyncio, "sleep", _noop_sleep)
    _FakeAsyncClient.calls = []
    _FakeAsyncClient.responses = [
        _FakeResponse(500, {"ok": False}, '{"error":"Connection dropped"}'),
        _FakeResponse(200, {"ok": True, "messageId": "msg-ok"}),
    ]

    result = await photon_adapter._standalone_send(
        PlatformConfig(extra={}),
        "any;-;+15551234567",
        "hello",
    )

    assert result == {"success": True, "message_id": "msg-ok"}
    assert len(_FakeAsyncClient.calls) == 2
    assert _FakeAsyncClient.calls[0][1]["spaceId"] == "any;-;+15551234567"


@pytest.mark.asyncio
async def test_photon_standalone_send_uses_sidecar_retryable_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(photon_adapter, "HTTPX_AVAILABLE", True)
    monkeypatch.setattr(
        photon_adapter,
        "httpx",
        types.SimpleNamespace(AsyncClient=_FakeAsyncClient),
    )
    monkeypatch.setenv("PHOTON_SIDECAR_TOKEN", "token")
    monkeypatch.setattr(photon_adapter.asyncio, "sleep", _noop_sleep)
    _FakeAsyncClient.calls = []
    _FakeAsyncClient.responses = [
        _FakeResponse(
            500,
            {
                "ok": False,
                "error": "Photon upstream connection dropped",
                "retryable": True,
            },
        ),
        _FakeResponse(200, {"ok": True, "messageId": "msg-ok"}),
    ]

    result = await photon_adapter._standalone_send(
        PlatformConfig(extra={}),
        "any;-;+15551234567",
        "hello",
    )

    assert result == {"success": True, "message_id": "msg-ok"}
    assert len(_FakeAsyncClient.calls) == 2


@pytest.mark.asyncio
async def test_photon_standalone_send_does_not_retry_non_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(photon_adapter, "HTTPX_AVAILABLE", True)
    monkeypatch.setattr(
        photon_adapter,
        "httpx",
        types.SimpleNamespace(AsyncClient=_FakeAsyncClient),
    )
    monkeypatch.setenv("PHOTON_SIDECAR_TOKEN", "token")
    monkeypatch.setattr(photon_adapter.asyncio, "sleep", _noop_sleep)
    _FakeAsyncClient.calls = []
    _FakeAsyncClient.responses = [
        _FakeResponse(
            500,
            {
                "ok": False,
                "error": "target not allowed for this Photon project",
                "retryable": False,
            },
        ),
        _FakeResponse(200, {"ok": True, "messageId": "should-not-send"}),
    ]

    result = await photon_adapter._standalone_send(
        PlatformConfig(extra={}),
        "+15551234567",
        "hello",
    )

    assert "target not allowed" in result["error"]
    assert len(_FakeAsyncClient.calls) == 1


class _BlankException(Exception):
    def __str__(self) -> str:
        return ""


@pytest.mark.asyncio
async def test_photon_standalone_send_names_blank_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(photon_adapter, "HTTPX_AVAILABLE", True)
    monkeypatch.setattr(
        photon_adapter,
        "httpx",
        types.SimpleNamespace(AsyncClient=_FakeAsyncClient),
    )
    monkeypatch.setenv("PHOTON_SIDECAR_TOKEN", "token")
    _FakeAsyncClient.calls = []
    _FakeAsyncClient.responses = [_BlankException()]

    result = await photon_adapter._standalone_send(
        PlatformConfig(extra={}),
        "+15551234567",
        "hello",
    )

    assert result == {"error": "Photon standalone send failed: _BlankException"}
    assert len(_FakeAsyncClient.calls) == 1


def test_photon_sidecar_resolves_dm_space_before_phone_create() -> None:
    source = Path("plugins/platforms/photon/sidecar/index.mjs").read_text()
    dm_get = source.index("photon-sidecar: DM space.get failed")
    phone_create = source.index("photon-sidecar: phone->DM space.create failed")
    assert dm_get < phone_create
