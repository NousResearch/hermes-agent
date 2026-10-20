"""Cron MEDIA-tag caption delivery (`MEDIA:<path> | <caption>` -> attachment caption).

Split out of ``tests/cron/test_scheduler.py``: that module is over the code-health FILE_LINES cap,
so new behaviour gets its own topical test module.
"""

from concurrent.futures import Future
from unittest.mock import MagicMock, patch

from cron.scheduler_delivery import _send_media_via_adapter


def _safe_media_path(tmp_path, monkeypatch, name, data=b"media"):
    root = tmp_path / "media-cache"
    media_file = root / name
    media_file.parent.mkdir(parents=True, exist_ok=True)
    media_file.write_bytes(data)
    monkeypatch.setattr(
        "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS",
        (root,),
    )
    return media_file.resolve()


def test_enum_platform_bounds_tag_caption(tmp_path, monkeypatch):
    from gateway.config import Platform

    adapter = MagicMock()
    adapter.platform = Platform.DISCORD
    photo_path = _safe_media_path(tmp_path, monkeypatch, "photo.jpg")

    def fake_run_coro(coro, _loop):
        coro.close()
        completed = Future()
        completed.set_result(MagicMock(success=True))
        return completed

    with patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro):
        _send_media_via_adapter(
            adapter, "123", [(str(photo_path), False)], None, MagicMock(), {"id": "j4"},
            media_captions={str(photo_path): "x" * 2500})
    assert len(adapter.send_image_file.call_args.kwargs["caption"]) == 2000
