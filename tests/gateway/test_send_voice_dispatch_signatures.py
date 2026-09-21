"""Every platform adapter's ``send_voice`` must accept the ``is_voice`` keyword the base media
dispatch passes (``BasePlatformAdapter._deliver_media_from_response`` →
``send_voice(..., is_voice=is_voice)``). An explicit signature without it raises ``TypeError`` and
the attachment is silently dropped (#102221, #116776 — Matrix; same class in line/mattermost/weixin)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_ADAPTER_FILES = sorted(
    p for p in list((_ROOT / "plugins" / "platforms").glob("*/adapter.py")) + list((_ROOT / "gateway" / "platforms").glob("*.py"))
    if "async def send_voice" in p.read_text(encoding="utf-8")
)


def _send_voice_defs(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "send_voice"]


@pytest.mark.parametrize("path", _ADAPTER_FILES, ids=lambda p: str(p.relative_to(_ROOT)))
def test_every_send_voice_accepts_is_voice(path: Path) -> None:
    for fn in _send_voice_defs(path):
        names = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
        assert fn.args.kwarg is not None or "is_voice" in names, (
            f"{path}: send_voice() rejects the dispatch's is_voice kwarg — add **kwargs or is_voice"
        )


@pytest.mark.asyncio
async def test_media_dispatch_delivers_audio_through_mattermost_send_voice(tmp_path) -> None:
    """Behavioural pin for the ``**kwargs`` class fix (line/mattermost/weixin): drive the real
    base dispatch, which calls ``send_voice(..., is_voice=False)`` for an audio-ext MEDIA
    attachment. Before the fix the call raised ``TypeError`` inside ``_send_one`` and the
    attachment was recorded as a failed delivery; now the file reaches the adapter's uploader."""
    from unittest.mock import AsyncMock

    from gateway.config import PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.platforms.event import MessageEvent
    from gateway.session import Platform, SessionSource
    from plugins.platforms.mattermost.adapter import MattermostAdapter

    adapter = MattermostAdapter(PlatformConfig(enabled=True, token="mm-token", extra={"url": "https://mm.example"}))
    adapter._send_local_file = AsyncMock(return_value=SendResult(success=True, message_id="post-1"))
    adapter._notify_media_delivery_failure = AsyncMock()
    clip = tmp_path / "clip.mp3"
    clip.write_bytes(b"ID3")
    event = MessageEvent(text="", source=SessionSource(platform=Platform.MATTERMOST, chat_id="chan-1"))
    results = []

    await adapter._deliver_media_attachments(
        event, [(str(clip), False)], [], force_document_attachments=False, human_delay=0,
        metadata={}, record_delivery=results.append)

    assert [r.success for r in results] == [True]
    adapter._notify_media_delivery_failure.assert_not_awaited()
    args, kwargs = adapter._send_local_file.call_args
    assert args[1] == str(clip)
