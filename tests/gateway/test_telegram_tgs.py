"""Tests for bounded Telegram TGS parsing and representative-frame extraction."""

import asyncio
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from plugins.platforms.telegram.tgs import (
    MAX_TGS_JSON_BYTES,
    _read_tgs_json,
    extract_tgs_frame,
)


def _tgs_bytes(**overrides) -> bytes:
    document = {
        "v": "5.7.4",
        "w": 512,
        "h": 512,
        "fr": 60,
        "ip": 0,
        "op": 120,
        "layers": [
            {"ty": 4, "ks": {"o": {"a": 1, "k": [{"t": 0}, {"t": 60}]}}},
            {"ty": 4, "ks": {"o": {"a": 0, "k": 100}}},
        ],
    }
    document.update(overrides)
    return gzip.compress(json.dumps(document).encode())


def test_extract_tgs_returns_metadata_without_optional_renderer(monkeypatch):
    monkeypatch.setattr("plugins.platforms.telegram.tgs.shutil.which", lambda _: None)

    result = extract_tgs_frame(_tgs_bytes())

    assert result.image_bytes is None
    assert result.metadata_description == (
        "a 512x512 animated sticker with 2 visual layers, "
        "lasting about 2.0 seconds, with motion in 1 layer"
    )


def test_extract_tgs_renders_representative_midpoint(monkeypatch):
    png = b"\x89PNG\r\n\x1a\n" + b"frame"
    observed = {}

    def fake_run(args, **kwargs):
        observed["args"] = args
        Path(args[2]).write_bytes(png)

    monkeypatch.setattr(
        "plugins.platforms.telegram.tgs.shutil.which",
        lambda _: "/usr/local/bin/lottie_convert.py",
    )
    monkeypatch.setattr("plugins.platforms.telegram.tgs.subprocess.run", fake_run)

    result = extract_tgs_frame(_tgs_bytes())

    assert result.image_bytes == png
    assert observed["args"][-2:] == ["--frame", "60"]


def test_extract_tgs_does_not_render_unsafe_dimensions(monkeypatch):
    run = Mock()
    monkeypatch.setattr("plugins.platforms.telegram.tgs.subprocess.run", run)
    monkeypatch.setattr(
        "plugins.platforms.telegram.tgs.shutil.which",
        lambda _: "/usr/local/bin/lottie_convert.py",
    )
    result = extract_tgs_frame(_tgs_bytes(w=100_000))

    assert result.image_bytes is None
    run.assert_not_called()


def test_extract_tgs_ignores_non_finite_metadata(monkeypatch):
    monkeypatch.setattr("plugins.platforms.telegram.tgs.shutil.which", lambda _: None)

    result = extract_tgs_frame(_tgs_bytes(w=float("inf"), fr=float("inf")))

    assert "unknown dimensions" in result.metadata_description
    assert "lasting about" not in result.metadata_description


def test_read_tgs_rejects_non_gzip_input():
    with pytest.raises(ValueError, match="gzip-compressed"):
        _read_tgs_json(b'{"layers": []}')


def test_read_tgs_bounds_decompressed_json():
    oversized = gzip.compress(b" " * (MAX_TGS_JSON_BYTES + 1))

    with pytest.raises(ValueError, match="decompressed-size"):
        _read_tgs_json(oversized)


def test_animated_sticker_thumbnail_uses_vision_and_cache():
    from plugins.platforms.telegram.adapter import TelegramAdapter
    from plugins.platforms.telegram.tgs import TGSFrameResult

    tgs_file = SimpleNamespace(
        download_as_bytearray=AsyncMock(return_value=_tgs_bytes())
    )
    thumbnail_file = SimpleNamespace(
        download_as_bytearray=AsyncMock(return_value=b"thumbnail-webp")
    )
    sticker = SimpleNamespace(
        emoji="👋",
        set_name="WavingPack",
        file_unique_id="animated-uid",
        is_animated=True,
        is_video=False,
        get_file=AsyncMock(return_value=tgs_file),
        thumbnail=SimpleNamespace(get_file=AsyncMock(return_value=thumbnail_file)),
    )
    event = SimpleNamespace(text="")
    vision = AsyncMock(
        return_value=json.dumps({"success": True, "analysis": "A cat waving hello"})
    )
    with (
        patch("gateway.sticker_cache.get_cached_description", return_value=None),
        patch("gateway.sticker_cache.cache_sticker_description") as cache_description,
        patch(
            "plugins.platforms.telegram.tgs.extract_tgs_frame",
            return_value=TGSFrameResult(None, "animation metadata"),
        ),
        patch(
            "plugins.platforms.telegram.adapter.cache_image_from_bytes",
            return_value="/tmp/sticker.webp",
        ) as cache_image,
        patch("tools.vision_tools.vision_analyze_tool", new=vision),
    ):
        asyncio.run(
            TelegramAdapter._handle_sticker(
                object.__new__(TelegramAdapter), SimpleNamespace(sticker=sticker), event
            )
        )

    cache_image.assert_called_once_with(b"thumbnail-webp", ext=".webp")
    vision.assert_awaited_once()
    cache_description.assert_called_once_with(
        "animated-uid", "A cat waving hello", "👋", "WavingPack"
    )
    assert 'It shows: "A cat waving hello"' in event.text


@pytest.mark.parametrize(
    "vision_failure",
    [json.dumps({"success": False}), RuntimeError("temporary vision outage")],
)
def test_animated_sticker_vision_failure_does_not_cache_metadata(
    vision_failure, tmp_path
):
    from plugins.platforms.telegram.adapter import TelegramAdapter
    from plugins.platforms.telegram.tgs import TGSFrameResult

    tgs_file = SimpleNamespace(
        download_as_bytearray=AsyncMock(return_value=_tgs_bytes())
    )
    sticker = SimpleNamespace(
        emoji="✨",
        set_name="Sparkles",
        file_unique_id="transient-vision-failure-uid",
        is_animated=True,
        is_video=False,
        get_file=AsyncMock(return_value=tgs_file),
        thumbnail=None,
    )
    event = SimpleNamespace(text="")
    vision = AsyncMock(
        side_effect=[
            vision_failure,
            json.dumps({"success": True, "analysis": "A sparkling star"}),
        ]
    )

    with (
        patch(
            "gateway.sticker_cache.CACHE_PATH",
            tmp_path / "sticker-cache.json",
        ),
        patch(
            "plugins.platforms.telegram.tgs.extract_tgs_frame",
            return_value=TGSFrameResult(
                b"\x89PNG\r\n\x1a\nframe",
                "a 512x512 animation with 2 layers",
            ),
        ) as extract_frame,
        patch(
            "plugins.platforms.telegram.adapter.cache_image_from_bytes",
            return_value="/tmp/sticker.png",
        ),
        patch(
            "tools.vision_tools.vision_analyze_tool",
            new=vision,
        ),
    ):
        asyncio.run(
            TelegramAdapter._handle_sticker(
                object.__new__(TelegramAdapter), SimpleNamespace(sticker=sticker), event
            )
        )
        retry_event = SimpleNamespace(text="")
        asyncio.run(
            TelegramAdapter._handle_sticker(
                object.__new__(TelegramAdapter),
                SimpleNamespace(sticker=sticker),
                retry_event,
            )
        )
        cached_event = SimpleNamespace(text="")
        asyncio.run(
            TelegramAdapter._handle_sticker(
                object.__new__(TelegramAdapter),
                SimpleNamespace(sticker=sticker),
                cached_event,
            )
        )

    assert "a 512x512 animation with 2 layers" in event.text
    assert vision.await_count == 2
    assert extract_frame.call_count == 2
    assert 'It shows: "A sparkling star"' in retry_event.text
    assert 'It shows: "A sparkling star"' in cached_event.text


def test_animated_sticker_metadata_fallback_is_not_cached(tmp_path):
    from plugins.platforms.telegram.adapter import TelegramAdapter
    from plugins.platforms.telegram.tgs import TGSFrameResult

    tgs_file = SimpleNamespace(
        download_as_bytearray=AsyncMock(return_value=_tgs_bytes())
    )
    sticker = SimpleNamespace(
        emoji="✨",
        set_name="Sparkles",
        file_unique_id="metadata-uid",
        is_animated=True,
        is_video=False,
        get_file=AsyncMock(return_value=tgs_file),
        thumbnail=None,
    )
    event = SimpleNamespace(text="")

    with (
        patch(
            "gateway.sticker_cache.CACHE_PATH",
            tmp_path / "sticker-cache.json",
        ),
        patch(
            "plugins.platforms.telegram.tgs.extract_tgs_frame",
            return_value=TGSFrameResult(None, "a 512x512 animation with 2 layers"),
        ) as extract_frame,
    ):
        asyncio.run(
            TelegramAdapter._handle_sticker(
                object.__new__(TelegramAdapter), SimpleNamespace(sticker=sticker), event
            )
        )
        retry_event = SimpleNamespace(text="")
        asyncio.run(
            TelegramAdapter._handle_sticker(
                object.__new__(TelegramAdapter),
                SimpleNamespace(sticker=sticker),
                retry_event,
            )
        )

    assert extract_frame.call_count == 2
    assert "a 512x512 animation with 2 layers" in event.text
    assert "a 512x512 animation with 2 layers" in retry_event.text


def test_video_sticker_vision_failure_keeps_emoji_fallback():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    thumbnail_file = SimpleNamespace(
        download_as_bytearray=AsyncMock(return_value=b"video-thumbnail")
    )
    sticker = SimpleNamespace(
        emoji="🎬",
        set_name="MoviePack",
        file_unique_id="video-uid",
        is_animated=False,
        is_video=True,
        thumbnail=SimpleNamespace(get_file=AsyncMock(return_value=thumbnail_file)),
    )
    event = SimpleNamespace(text="")

    with (
        patch("gateway.sticker_cache.get_cached_description", return_value=None),
        patch(
            "plugins.platforms.telegram.adapter.cache_image_from_bytes",
            return_value="/tmp/video-sticker.webp",
        ),
        patch(
            "tools.vision_tools.vision_analyze_tool",
            new=AsyncMock(return_value=json.dumps({"success": False})),
        ),
    ):
        asyncio.run(
            TelegramAdapter._handle_sticker(
                object.__new__(TelegramAdapter), SimpleNamespace(sticker=sticker), event
            )
        )

    assert "animated sticker 🎬" in event.text
    assert "emoji suggests: 🎬" in event.text
