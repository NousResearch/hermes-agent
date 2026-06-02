"""Tests for the Photon adapter runtime boundary."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from plugins.platforms.photon import adapter as photon_adapter


def _make_adapter(monkeypatch: Any, tmp_path: Path) -> photon_adapter.PhotonAdapter:
    monkeypatch.delenv("PHOTON_PROJECT_ID", raising=False)
    monkeypatch.delenv("PHOTON_PROJECT_SECRET", raising=False)
    monkeypatch.delenv("PHOTON_PROJECT_NAME", raising=False)
    monkeypatch.delenv("PHOTON_OPERATOR_PHONE", raising=False)
    monkeypatch.setattr(photon_adapter, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(
        photon_adapter,
        "load_project_credentials",
        lambda: ("project-id", "project-secret"),
    )
    return photon_adapter.PhotonAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "project_name": "hermes-agent",
                "operator_phone": "+15105550123",
            },
        )
    )


def test_adapter_process_env_exports_canonical_hermes_home(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setattr(photon_adapter, "get_hermes_home", lambda: tmp_path)

    env = photon_adapter._adapter_process_env(
        project_id="project-id",
        project_secret="project-secret",
    )

    assert env["HERMES_HOME"] == str(tmp_path)
    assert env["PHOTON_PROJECT_ID"] == "project-id"
    assert env["PHOTON_PROJECT_SECRET"] == "project-secret"
    assert "PHOTON_SIDECAR_PORT" not in env
    assert "PHOTON_SIDECAR_BIND" not in env
    assert "PHOTON_SIDECAR_TOKEN" not in env
    assert env["PATH"] == "/usr/bin"


def test_env_enablement_hydrates_home_channel(monkeypatch: Any) -> None:
    env = {
        "PHOTON_HOME_CHANNEL": "any;-;+15105550123",
        "PHOTON_HOME_CHANNEL_NAME": "You (iMessage)",
    }

    monkeypatch.setattr(
        photon_adapter,
        "load_project_credentials",
        lambda: ("project-id", "project-secret"),
    )
    monkeypatch.setattr(
        photon_adapter,
        "_get_hermes_env_value",
        lambda key: env.get(key),
    )

    seed = photon_adapter._env_enablement()

    assert seed == {
        "project_id": "project-id",
        "project_secret": "project-secret",
        "home_channel": {
            "chat_id": "any;-;+15105550123",
            "name": "You (iMessage)",
        },
    }


def test_env_enablement_defaults_home_channel_name(monkeypatch: Any) -> None:
    env = {"PHOTON_HOME_CHANNEL": "any;-;+15105550123"}

    monkeypatch.setattr(
        photon_adapter,
        "load_project_credentials",
        lambda: ("project-id", "project-secret"),
    )
    monkeypatch.setattr(
        photon_adapter,
        "_get_hermes_env_value",
        lambda key: env.get(key),
    )

    seed = photon_adapter._env_enablement()

    assert seed is not None
    assert seed["home_channel"] == {
        "chat_id": "any;-;+15105550123",
        "name": "You (iMessage)",
    }


def test_sdk_event_normalizes_to_message_event(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    adapter = _make_adapter(monkeypatch, tmp_path)

    event = adapter._message_event_from_sdk_event(
        {
            "id": "msg-1",
            "timestamp": "2026-05-30T12:00:00Z",
            "space": {"id": "any;-;+15105550123", "name": "Ray"},
            "sender": {"id": "+15105550123", "name": "Ray"},
            "content": {"type": "text", "text": "hello"},
        },
        message_id="msg-1",
    )

    assert isinstance(event, MessageEvent)
    assert event.text == "hello"
    assert event.message_type is MessageType.TEXT
    assert event.message_id == "msg-1"
    assert event.source.chat_id == "any;-;+15105550123"
    assert event.source.chat_type == "dm"
    assert event.source.user_id == "+15105550123"


def test_sdk_event_dedupes_repeated_sdk_message(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    adapter = _make_adapter(monkeypatch, tmp_path)
    events: list[MessageEvent] = []

    async def capture(event: MessageEvent) -> None:
        events.append(event)

    monkeypatch.setattr(adapter, "handle_message", capture)

    payload = {
        "id": "msg-1",
        "space": {"id": "any;-;+15105550123"},
        "sender": {"id": "+15105550123"},
        "content": {"type": "text", "text": "once"},
    }
    asyncio.run(adapter._handle_sdk_event(payload))
    asyncio.run(adapter._handle_sdk_event(payload))

    assert [event.text for event in events] == ["once"]


def test_send_builds_sdk_payload_and_maps_success(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    adapter = _make_adapter(monkeypatch, tmp_path)
    calls: list[tuple[str, dict[str, Any]]] = []

    async def fake_request(command: str, payload: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        calls.append((command, payload))
        return {"messageId": "sent-1", "raw": {"id": "sent-1"}}

    monkeypatch.setattr(adapter, "_sdk_request", fake_request)

    result = asyncio.run(adapter.send("any;-;+15105550123", "hi", reply_to="msg-1"))

    assert result.success is True
    assert result.message_id == "sent-1"
    assert result.raw_response == {"id": "sent-1"}
    assert calls == [
        (
            "send",
            {"spaceId": "any;-;+15105550123", "text": "hi", "replyTo": "msg-1"},
        )
    ]


@pytest.mark.parametrize(
    ("sdk_error", "expected_retryable", "expected_error"),
    [
        (
            photon_adapter.RetryableAdapterError(
                "SDK_SESSION_UNAVAILABLE",
                "session unavailable",
            ),
            True,
            "session unavailable",
        ),
        (
            photon_adapter.AdapterUnavailableError(),
            True,
            "Photon Spectrum adapter is not running",
        ),
        (
            photon_adapter.PermanentAdapterError("BAD_PAYLOAD", "bad payload"),
            False,
            "bad payload",
        ),
    ],
)
def test_send_maps_adapter_errors(
    tmp_path: Path,
    monkeypatch: Any,
    sdk_error: photon_adapter.PhotonAdapterError,
    expected_retryable: bool,
    expected_error: str,
) -> None:
    adapter = _make_adapter(monkeypatch, tmp_path)

    async def fake_request(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise sdk_error

    monkeypatch.setattr(adapter, "_sdk_request", fake_request)

    result = asyncio.run(adapter.send("any;-;+15105550123", "hi"))

    assert result.success is False
    assert result.retryable is expected_retryable
    assert result.error == expected_error


def test_send_rejects_bad_payload_without_retry(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    adapter = _make_adapter(monkeypatch, tmp_path)

    result = asyncio.run(adapter.send("", "hi"))

    assert result.success is False
    assert result.retryable is False
    assert "chat_id" in (result.error or "")


def test_connect_acquires_project_lock_before_sidecar_start(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    adapter = _make_adapter(monkeypatch, tmp_path)
    order: list[str] = []

    monkeypatch.setattr(photon_adapter.shutil, "which", lambda _value: "/usr/bin/node")
    monkeypatch.setattr(
        adapter,
        "_acquire_platform_lock",
        lambda *args: order.append(f"lock:{args[1]}") or True,
    )

    async def fake_start() -> None:
        order.append("start")

    monkeypatch.setattr(adapter, "_start_sdk_sidecar", fake_start)

    assert asyncio.run(adapter.connect()) is True
    assert order == ["lock:project-id", "start"]


def test_connect_lock_failure_prevents_sidecar_start(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    adapter = _make_adapter(monkeypatch, tmp_path)

    monkeypatch.setattr(photon_adapter.shutil, "which", lambda _value: "/usr/bin/node")

    def deny_lock(*_args: Any) -> bool:
        adapter._set_fatal_error(
            "photon_lock",
            "Photon Spectrum project already in use. Stop the other gateway first.",
            retryable=False,
        )
        return False

    monkeypatch.setattr(adapter, "_acquire_platform_lock", deny_lock)
    monkeypatch.setattr(
        adapter,
        "_start_sdk_sidecar",
        lambda: pytest.fail("sidecar must not start when lock acquisition fails"),
    )

    assert asyncio.run(adapter.connect()) is False
    assert adapter._adapter_state == "fatal"
    assert adapter._last_error is not None
    assert adapter._last_error["code"] == "photon_lock"
    assert "Stop the other gateway first" in adapter._last_error["message"]


def test_connect_releases_lock_after_startup_failure(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    adapter = _make_adapter(monkeypatch, tmp_path)
    released: list[bool] = []

    monkeypatch.setattr(photon_adapter.shutil, "which", lambda _value: "/usr/bin/node")
    monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *_args: True)
    monkeypatch.setattr(adapter, "_release_platform_lock", lambda: released.append(True))

    async def fail_start() -> None:
        raise photon_adapter.RetryableAdapterError("SDK_SIDECAR_FAILED", "boom")

    monkeypatch.setattr(adapter, "_start_sdk_sidecar", fail_start)

    assert asyncio.run(adapter.connect()) is False
    assert released == [True]
    assert adapter._adapter_state == "fatal"


def test_disconnect_releases_project_lock(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    adapter = _make_adapter(monkeypatch, tmp_path)
    released: list[bool] = []

    async def fake_stop() -> None:
        return None

    monkeypatch.setattr(adapter, "_stop_sdk_sidecar", fake_stop)
    monkeypatch.setattr(adapter, "_release_platform_lock", lambda: released.append(True))

    asyncio.run(adapter.disconnect())

    assert released == [True]


def test_sidecar_stdout_ignores_invalid_json_and_keeps_reading(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    adapter = _make_adapter(monkeypatch, tmp_path)

    class FakeStdout:
        def __init__(self) -> None:
            self._lines = [
                b"not-json\n",
                b'{"type":"ready","startedAt":"2026-05-30T12:00:00+00:00"}\n',
                b"",
            ]

        async def readline(self) -> bytes:
            return self._lines.pop(0)

    class FakeProc:
        stdout = FakeStdout()

    async def run() -> None:
        adapter._sidecar_ready = asyncio.get_running_loop().create_future()
        await adapter._read_sidecar_stdout(FakeProc())  # type: ignore[arg-type]

    asyncio.run(run())

    assert adapter._adapter_state == "connected"
    assert adapter._sdk_connected is True


def test_stream_error_marks_failed(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    adapter = _make_adapter(monkeypatch, tmp_path)
    adapter._running = True

    asyncio.run(
        adapter._handle_sidecar_message(
            {
                "type": "stream_error",
                "error": {
                    "code": "STREAM_ENDED",
                    "message": "ended",
                    "retryable": True,
                },
            }
        )
    )

    assert adapter._adapter_state == "failed"
    assert adapter._sdk_connected is False
    assert adapter.is_connected is False
    assert adapter._last_error is not None
    assert adapter._last_error["code"] == "STREAM_ENDED"


def test_stream_error_fails_pending_requests(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    adapter = _make_adapter(monkeypatch, tmp_path)

    async def run() -> None:
        pending = asyncio.get_running_loop().create_future()
        adapter._pending_requests["req-1"] = pending
        await adapter._handle_sidecar_message(
            {
                "type": "stream_error",
                "error": {
                    "code": "STREAM_ENDED",
                    "message": "ended",
                    "retryable": True,
                },
            }
        )

        assert pending.done()
        with pytest.raises(photon_adapter.RetryableAdapterError):
            await pending

    asyncio.run(run())


def test_sidecar_exit_marks_failed(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    adapter = _make_adapter(monkeypatch, tmp_path)
    adapter._running = True

    class FakeProc:
        pid = 123
        returncode = 12

        async def wait(self) -> int:
            return 12

    proc = FakeProc()
    adapter._sidecar_proc = proc  # type: ignore[assignment]

    asyncio.run(adapter._watch_sidecar_exit(proc))  # type: ignore[arg-type]

    assert adapter._adapter_state == "failed"
    assert adapter._sdk_connected is False
    assert adapter.is_connected is False
    assert adapter._last_error is not None
    assert adapter._last_error["code"] == "SDK_SIDECAR_EXITED"


def test_sidecar_exit_fails_pending_requests(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    adapter = _make_adapter(monkeypatch, tmp_path)

    class FakeProc:
        pid = 123
        returncode = 12

        async def wait(self) -> int:
            return 12

    async def run() -> None:
        pending = asyncio.get_running_loop().create_future()
        adapter._pending_requests["req-1"] = pending
        proc = FakeProc()
        adapter._sidecar_proc = proc  # type: ignore[assignment]

        await adapter._watch_sidecar_exit(proc)  # type: ignore[arg-type]

        assert pending.done()
        with pytest.raises(photon_adapter.RetryableAdapterError):
            await pending

    asyncio.run(run())


def test_runtime_state_is_current_home_adapter_status(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    adapter = _make_adapter(monkeypatch, tmp_path)

    class FakeProc:
        pid = 456
        returncode = None

    adapter._sidecar_proc = FakeProc()  # type: ignore[assignment]
    adapter._sdk_connected = True
    adapter._adapter_state = "connected"
    adapter._started_at = "2026-05-30T12:00:00+00:00"

    adapter._write_adapter_runtime_state()
    state = photon_adapter.read_adapter_runtime_state()

    assert state["pid"] == 456
    assert state["project_id"] == "project-id"
    assert state["operator_phone"] == "+15105550123"
    assert state["health"]["healthy"] is True
    assert state["health"]["sdk"]["connected"] is True


@pytest.mark.parametrize(
    ("cfg_extra", "chat_id", "message", "use_media", "expected_error"),
    [
        ({}, "any;-;+15105550123", "hi", False, "PHOTON_PROJECT_ID"),
        (
            {"project_id": "p", "project_secret": "s"},
            "",
            "hi",
            False,
            "chat_id is required",
        ),
        (
            {"project_id": "p", "project_secret": "s"},
            "any;-;+15105550123",
            "hi",
            True,
            "media attachments",
        ),
    ],
)
def test_standalone_send_rejects_invalid_inputs(
    tmp_path: Path,
    monkeypatch: Any,
    cfg_extra: dict[str, str],
    chat_id: str,
    message: str,
    use_media: bool,
    expected_error: str,
) -> None:
    monkeypatch.delenv("PHOTON_PROJECT_ID", raising=False)
    monkeypatch.delenv("PHOTON_PROJECT_SECRET", raising=False)
    monkeypatch.setattr(photon_adapter, "load_project_credentials", lambda: (None, None))

    kwargs: dict[str, Any] = {}
    if use_media:
        kwargs["media_files"] = [(str(tmp_path / "image.png"), False)]

    result = asyncio.run(
        photon_adapter._standalone_send(
            PlatformConfig(enabled=True, extra=cfg_extra),
            chat_id,
            message,
            **kwargs,
        )
    )

    assert expected_error in result["error"]


def test_standalone_send_rejects_missing_sidecar_deps(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(photon_adapter.shutil, "which", lambda _value: "/usr/bin/node")
    monkeypatch.setattr(photon_adapter, "_SIDECAR_DIR", tmp_path)

    result = asyncio.run(
        photon_adapter._standalone_send(
            PlatformConfig(enabled=True, extra={"project_id": "p", "project_secret": "s"}),
            "any;-;+15105550123",
            "hi",
        )
    )

    assert "sidecar deps are not installed" in result["error"]


def test_standalone_send_maps_sidecar_success(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    (tmp_path / "node_modules").mkdir()
    calls: list[dict[str, Any]] = []

    monkeypatch.setattr(photon_adapter.shutil, "which", lambda _value: "/usr/bin/node")
    monkeypatch.setattr(photon_adapter, "_SIDECAR_DIR", tmp_path)

    async def fake_send_once(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"messageId": "sent-1", "raw": {"id": "sent-1"}}

    monkeypatch.setattr(photon_adapter, "_send_once_via_sidecar", fake_send_once)

    result = asyncio.run(
        photon_adapter._standalone_send(
            PlatformConfig(enabled=True, extra={"project_id": "p", "project_secret": "s"}),
            "any;-;+15105550123",
            "hello",
        )
    )

    assert result["success"] is True
    assert result["message_id"] == "sent-1"
    assert calls == [
        {
            "node_bin": "/usr/bin/node",
            "project_id": "p",
            "project_secret": "s",
            "chat_id": "any;-;+15105550123",
            "text": "hello",
        }
    ]


def test_standalone_send_truncates_to_photon_limit(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    (tmp_path / "node_modules").mkdir()
    captured: dict[str, Any] = {}

    monkeypatch.setattr(photon_adapter.shutil, "which", lambda _value: "/usr/bin/node")
    monkeypatch.setattr(photon_adapter, "_SIDECAR_DIR", tmp_path)

    async def fake_send_once(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"messageId": "sent-1"}

    monkeypatch.setattr(photon_adapter, "_send_once_via_sidecar", fake_send_once)

    result = asyncio.run(
        photon_adapter._standalone_send(
            PlatformConfig(enabled=True, extra={"project_id": "p", "project_secret": "s"}),
            "any;-;+15105550123",
            "x" * (photon_adapter._MAX_MESSAGE_LENGTH + 10),
        )
    )

    assert result["success"] is True
    assert len(captured["text"]) == photon_adapter._MAX_MESSAGE_LENGTH


def test_sidecar_send_once_mode_is_before_inbound_stream() -> None:
    source = (photon_adapter._SIDECAR_ENTRYPOINT).read_text()

    assert "sendOnceMode" in source
    assert "mode: \"send-once\"" in source
    assert source.index("if (sendOnceMode)") < source.index(
        "for await (const [space, message] of app.messages)"
    )


def test_cron_scheduler_recognizes_photon_home_channel(
    monkeypatch: Any,
) -> None:
    from cron import scheduler
    from gateway.platform_registry import PlatformEntry, platform_registry

    original = platform_registry.get("photon")
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    platform_registry.register(
        PlatformEntry(
            name="photon",
            label="Photon",
            adapter_factory=lambda cfg: photon_adapter.PhotonAdapter(cfg),
            check_fn=lambda: True,
            cron_deliver_env_var="PHOTON_HOME_CHANNEL",
            standalone_sender_fn=photon_adapter._standalone_send,
        )
    )
    try:
        assert scheduler._is_known_delivery_platform("photon") is True
        assert scheduler._resolve_home_env_var("photon") == "PHOTON_HOME_CHANNEL"
    finally:
        platform_registry.unregister("photon")
        if original is not None:
            platform_registry.register(original)
