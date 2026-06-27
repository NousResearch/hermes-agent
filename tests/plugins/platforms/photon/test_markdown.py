"""Markdown handling tests for PhotonAdapter.

Markdown is on by default (the sidecar sends it via spectrum-ts'
``markdown()`` builder and iMessage renders it); ``PHOTON_MARKDOWN=false``
reverts to the stripped-plain-text path.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.photon import adapter as photon_adapter
from plugins.platforms.photon.adapter import PhotonAdapter

_MD = "**bold** and `code`"


def _make_adapter(monkeypatch: pytest.MonkeyPatch) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    cfg = PlatformConfig(enabled=True, token="", extra={})
    return PhotonAdapter(cfg)


def _capture_sidecar(adapter: PhotonAdapter) -> List[Tuple[str, Dict[str, Any]]]:
    calls: List[Tuple[str, Dict[str, Any]]] = []

    async def _fake_call(path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        calls.append((path, body))
        return {"ok": True, "messageId": "msg-123"}

    adapter._sidecar_call = _fake_call  # type: ignore[assignment]
    return calls


def test_format_message_passthrough_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PHOTON_MARKDOWN", raising=False)
    adapter = _make_adapter(monkeypatch)
    assert adapter.format_message(_MD) == _MD


def test_format_message_strips_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHOTON_MARKDOWN", "false")
    adapter = _make_adapter(monkeypatch)
    assert adapter.format_message(_MD) == "bold and code"


def test_supports_code_blocks_mirrors_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PHOTON_MARKDOWN", raising=False)
    assert _make_adapter(monkeypatch).supports_code_blocks is True
    monkeypatch.setenv("PHOTON_MARKDOWN", "false")
    assert _make_adapter(monkeypatch).supports_code_blocks is False


@pytest.mark.asyncio
async def test_sidecar_send_includes_markdown_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PHOTON_MARKDOWN", raising=False)
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    await adapter.send("+15551234567", _MD)

    path, body = calls[0]
    assert path == "/send"
    assert body["format"] == "markdown"
    assert body["text"] == _MD  # passed through unstripped


@pytest.mark.asyncio
async def test_sidecar_send_omits_format_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Old-sidecar compat: the key is absent, not "text", when disabled."""
    monkeypatch.setenv("PHOTON_MARKDOWN", "false")
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    await adapter.send("+15551234567", _MD)

    _, body = calls[0]
    assert "format" not in body
    assert body["text"] == "bold and code"


@pytest.mark.asyncio
async def test_standalone_send_includes_markdown_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PHOTON_MARKDOWN", raising=False)
    monkeypatch.setenv("PHOTON_SIDECAR_TOKEN", "tok")

    posted: List[Tuple[str, Dict[str, Any]]] = []

    class _Resp:
        status_code = 200

        @staticmethod
        def json() -> Dict[str, Any]:
            return {"ok": True, "messageId": "m-9"}

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url: str, json: Dict[str, Any], headers=None):
            posted.append((url, json))
            return _Resp()

    monkeypatch.setattr(photon_adapter.httpx, "AsyncClient", _FakeClient)

    cfg = PlatformConfig(enabled=True, token="", extra={})
    result = await photon_adapter._standalone_send(cfg, "+15551234567", _MD)

    assert result.get("success") is True
    assert posted[0][1]["format"] == "markdown"


@pytest.mark.asyncio
async def test_standalone_send_retries_retryable_sidecar_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PHOTON_MARKDOWN", raising=False)
    monkeypatch.setenv("PHOTON_SIDECAR_TOKEN", "tok")
    monkeypatch.setenv("PHOTON_STANDALONE_RETRY_BASE_DELAY_SECONDS", "0")

    posted: List[Tuple[str, Dict[str, Any]]] = []

    class _Resp:
        status_code = 200

        def __init__(self, payload: Dict[str, Any]) -> None:
            self._payload = payload

        def json(self) -> Dict[str, Any]:
            return self._payload

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url: str, json: Dict[str, Any], headers=None):
            posted.append((url, json))
            if len(posted) == 1:
                return _Resp({"ok": False, "error": "internal sidecar error"})
            return _Resp({"ok": True, "messageId": "m-retry"})

    monkeypatch.setattr(photon_adapter.httpx, "AsyncClient", _FakeClient)

    cfg = PlatformConfig(enabled=True, token="", extra={})
    result = await photon_adapter._standalone_send(cfg, "+15551234567", _MD)

    assert result.get("success") is True
    assert result.get("message_id") == "m-retry"
    assert len(posted) == 2
    assert posted[0][0].endswith("/send")
    assert posted[1][0].endswith("/send")
    assert posted[0][1]["format"] == "markdown"
    assert posted[1][1]["format"] == "markdown"


@pytest.mark.asyncio
async def test_standalone_send_falls_back_to_plain_text_after_markdown_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PHOTON_MARKDOWN", raising=False)
    monkeypatch.setenv("PHOTON_SIDECAR_TOKEN", "tok")
    monkeypatch.setenv("PHOTON_STANDALONE_RETRY_BASE_DELAY_SECONDS", "0")

    posted: List[Tuple[str, Dict[str, Any]]] = []

    class _Resp:
        status_code = 200

        def __init__(self, payload: Dict[str, Any]) -> None:
            self._payload = payload

        def json(self) -> Dict[str, Any]:
            return self._payload

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url: str, json: Dict[str, Any], headers=None):
            posted.append((url, json))
            if len(posted) <= 2:
                return _Resp({"ok": False, "error": "internal sidecar error"})
            return _Resp({"ok": True, "messageId": "m-plain"})

    monkeypatch.setattr(photon_adapter.httpx, "AsyncClient", _FakeClient)

    cfg = PlatformConfig(enabled=True, token="", extra={})
    result = await photon_adapter._standalone_send(cfg, "+15551234567", _MD)

    assert result.get("success") is True
    assert result.get("message_id") == "m-plain"
    assert len(posted) == 3
    assert posted[0][1]["format"] == "markdown"
    assert posted[1][1]["format"] == "markdown"
    assert "format" not in posted[2][1]
    assert posted[2][1]["text"] == "bold and code"
