from __future__ import annotations

import pytest

from agent import provider_media


class _Response:
    def __init__(self, content_type: str, chunks: list[bytes]):
        self.headers = {"Content-Type": content_type}
        self._chunks = chunks

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size: int):
        del chunk_size
        return iter(self._chunks)


def _save_video(url: str, **kwargs):
    return provider_media.save_url(
        "videos",
        url,
        prefix="provider-test",
        timeout=30,
        max_bytes=1024,
        chunk_size=64,
        content_types={"video/mp4": "mp4"},
        url_extensions=("mp4",),
        default_extension="mp4",
        label="Video",
        empty_error="empty: {url}",
        **kwargs,
    )


def test_save_url_forwards_headers_and_streams_to_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return _Response("video/mp4", [b"clip"])

    monkeypatch.setattr("requests.get", fake_get)

    path = _save_video(
        "https://api.example/videos/job/content",
        headers={"Authorization": "Bearer test"},
        require_known_content_type=True,
    )

    assert path.read_bytes() == b"clip"
    assert path.suffix == ".mp4"
    assert calls == [
        (
            "https://api.example/videos/job/content",
            {
                "headers": {"Authorization": "Bearer test"},
                "timeout": 30,
                "stream": True,
            },
        )
    ]


def test_save_url_strict_content_type_rejects_non_video(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(
        "requests.get",
        lambda *_args, **_kwargs: _Response("text/html", [b"not a video"]),
    )

    with pytest.raises(ValueError, match="unexpected Content-Type text/html"):
        _save_video(
            "https://api.example/videos/job/content",
            require_known_content_type=True,
        )

    assert not list(provider_media.cache_dir("videos").iterdir())
