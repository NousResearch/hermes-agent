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


_MD_WITH_URL = (
    "**Release 1.2.0** is out\n"
    "- **\u20ac5** off this month\n"
    "https://example.com/releases/1.2.0"
)
_MD_WITH_LINK = "**Release 1.2.0** is out\n[Read the notes](https://example.com/releases/1.2.0)"


@pytest.mark.asyncio
async def test_url_bearing_markdown_is_stripped_before_the_text_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A markdown payload the sidecar downgrades must not ship raw markers.

    ``chooseSendFormat`` routes markdown containing a raw URL through
    spectrum-ts' ``text()`` builder, which sends the source verbatim.
    """
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    await adapter.send("space-1", _MD_WITH_URL)

    path, body = calls[-1]
    assert path == "/send"
    # The format key is deliberately preserved: the sidecar owns the builder
    # choice (see test_rich_links.py). Only the payload changes.
    assert body["format"] == "markdown"
    assert "**" not in body["text"]
    # Stripping must not damage the parts iMessage still needs.
    assert "https://example.com/releases/1.2.0" in body["text"]
    assert "\u20ac5" in body["text"]


@pytest.mark.asyncio
async def test_url_free_markdown_still_renders_natively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    await adapter.send("space-1", _MD)

    _, body = calls[-1]
    assert body["format"] == "markdown"
    assert body["text"] == _MD, "URL-free markdown must reach markdown() untouched"


@pytest.mark.asyncio
async def test_plain_fallback_strips_markdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The explicit fallback selects the text builder, so it must strip too."""
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    await adapter._send_plain_fallback("space-1", _MD, reply_to=None, metadata=None)

    _, body = calls[-1]
    assert "format" not in body
    assert "**" not in body["text"]
    assert "`" not in body["text"]


@pytest.mark.asyncio
async def test_markdown_link_keeps_a_tappable_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``[label](url)`` link must not lose its URL on the way to iMessage.

    The URL inside ``](...)`` is enough for ``chooseSendFormat`` to pick the text
    builder, so this payload gets stripped — and iMessage auto-links bare URLs
    only. Dropping the target (the shared default) leaves the link unreachable.
    """
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    await adapter.send("space-1", _MD_WITH_LINK)

    _, body = calls[-1]
    assert body["text"] == (
        "Release 1.2.0 is out\nRead the notes\nhttps://example.com/releases/1.2.0"
    )


@pytest.mark.asyncio
async def test_plain_fallback_keeps_link_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    await adapter._send_plain_fallback("space-1", _MD_WITH_LINK, reply_to=None, metadata=None)

    _, body = calls[-1]
    assert "**" not in body["text"]
    assert "https://example.com/releases/1.2.0" in body["text"]


def test_markdown_disabled_keeps_link_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    """``PHOTON_MARKDOWN=false`` strips through the same iMessage-aware helper."""
    monkeypatch.setenv("PHOTON_MARKDOWN", "false")
    adapter = _make_adapter(monkeypatch)

    assert adapter.format_message(_MD_WITH_LINK) == (
        "Release 1.2.0 is out\nRead the notes\nhttps://example.com/releases/1.2.0"
    )
