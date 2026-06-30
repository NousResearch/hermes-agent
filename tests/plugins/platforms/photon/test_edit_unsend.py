"""Native iMessage edit/unsend tests for PhotonAdapter.

The edit and unsend paths are deliberately feature-gated and restricted to
messages Hermes has sent in the current adapter run. These tests stub
``_sidecar_call`` and do not spawn the Node sidecar or bind ports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.photon.adapter import PhotonAdapter


def _make_adapter(monkeypatch: pytest.MonkeyPatch) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    cfg = PlatformConfig(enabled=True, token="", extra={})
    return PhotonAdapter(cfg)


def _capture_sidecar(adapter: PhotonAdapter) -> List[Tuple[str, Dict[str, Any]]]:
    calls: List[Tuple[str, Dict[str, Any]]] = []

    async def _fake_call(path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        calls.append((path, body))
        return {"ok": True, "messageId": body.get("messageId", "msg-123")}

    adapter._sidecar_call = _fake_call  # type: ignore[assignment]
    return calls


@pytest.mark.asyncio
async def test_edit_sent_message_is_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PHOTON_NATIVE_EDITS", raising=False)
    adapter = _make_adapter(monkeypatch)
    adapter._record_sent_message("bot-msg-1")
    calls = _capture_sidecar(adapter)

    result = await adapter.edit_sent_message("+155****4567", "bot-msg-1", "edited")

    assert result.success is False
    assert "disabled" in (result.error or "")
    assert calls == []


@pytest.mark.asyncio
async def test_edit_sent_message_requires_tracked_hermes_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHOTON_NATIVE_EDITS", "true")
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    result = await adapter.edit_sent_message("+155****4567", "human-msg-1", "edited")

    assert result.success is False
    assert "Hermes-sent" in (result.error or "")
    assert calls == []


@pytest.mark.asyncio
async def test_edit_sent_message_posts_guarded_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHOTON_NATIVE_EDITS", "true")
    adapter = _make_adapter(monkeypatch)
    adapter._record_sent_message("bot-msg-1")
    calls = _capture_sidecar(adapter)

    result = await adapter.edit_sent_message(
        "+155****4567", "bot-msg-1", "**edited**"
    )

    assert result.success is True
    assert result.message_id == "bot-msg-1"
    assert result.raw_response == {"ok": True, "messageId": "bot-msg-1"}
    assert calls == [
        (
            "/edit",
            {
                "spaceId": "+155****4567",
                "messageId": "bot-msg-1",
                "text": "**edited**",
                "hermesSent": True,
            },
        )
    ]


def test_sidecar_edit_uses_text_content_before_advanced_fallback() -> None:
    index = Path("plugins/platforms/photon/sidecar/index.mjs").read_text(
        encoding="utf-8"
    )
    edit_block = index[
        index.index('if (req.url === "/edit")') : index.index(
            'if (req.url === "/unsend")'
        )
    ]

    assert "const content = spectrumText(text);" in edit_block
    assert "spectrumMarkdown(text)" not in edit_block
    assert edit_block.index("await target.edit(content)") < edit_block.index(
        "client.messages.edit"
    )


@pytest.mark.asyncio
async def test_unsend_sent_message_is_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PHOTON_NATIVE_UNSEND", raising=False)
    adapter = _make_adapter(monkeypatch)
    adapter._record_sent_message("bot-msg-1")
    calls = _capture_sidecar(adapter)

    result = await adapter.unsend_sent_message("+155****4567", "bot-msg-1")

    assert result.success is False
    assert "disabled" in (result.error or "")
    assert calls == []


@pytest.mark.asyncio
async def test_unsend_sent_message_requires_tracked_hermes_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHOTON_NATIVE_UNSEND", "true")
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    result = await adapter.unsend_sent_message("+155****4567", "human-msg-1")

    assert result.success is False
    assert "Hermes-sent" in (result.error or "")
    assert calls == []


@pytest.mark.asyncio
async def test_unsend_sent_message_posts_guarded_payload_and_forgets_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHOTON_NATIVE_UNSEND", "true")
    adapter = _make_adapter(monkeypatch)
    adapter._record_sent_message("bot-msg-1")
    calls = _capture_sidecar(adapter)

    result = await adapter.unsend_sent_message("+155****4567", "bot-msg-1")

    assert result.success is True
    assert result.message_id == "bot-msg-1"
    assert result.raw_response == {"ok": True, "messageId": "bot-msg-1"}
    assert "bot-msg-1" not in adapter._sent_message_ids
    assert calls == [
        (
            "/unsend",
            {
                "spaceId": "+155****4567",
                "messageId": "bot-msg-1",
                "hermesSent": True,
            },
        )
    ]
