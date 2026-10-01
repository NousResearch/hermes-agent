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


def test_supports_code_blocks_never_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fenced code blocks are not renderable on iMessage: URL-bearing
    messages fall back to raw text (literal fences), and the markdown path
    renders a fence as inline monospace, not a block. The adapter therefore
    never advertises code-block support — the gateway emits its compact
    one-line tool preview instead (see test_code_block_capability.py)."""
    monkeypatch.delenv("PHOTON_MARKDOWN", raising=False)
    assert _make_adapter(monkeypatch).supports_code_blocks is False
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
async def test_url_bearing_markdown_is_stripped_before_the_text_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``chooseSendFormat`` sends markdown containing a URL through spectrum-ts' verbatim ``text()``
    builder, so the adapter strips first, and a ``[label](url)`` link keeps a tappable bare URL."""
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    await adapter.send("space-1", "**Release 1.2.0** is out\n[Read the notes](https://example.com/notes)")

    path, body = calls[-1]
    assert path == "/send"
    assert body["format"] == "markdown"  # the sidecar owns the builder choice (test_rich_links.py)
    assert "**" not in body["text"] and "](" not in body["text"]
    assert "\nhttps://example.com/notes" in body["text"]
