from __future__ import annotations

from pathlib import Path

import pytest
import requests

from agent.image_gen_provider import save_url_image
from agent.video_gen_provider import save_url_video


class _Response:
    headers = {"Content-Type": "application/octet-stream"}

    def __init__(self, chunks: list[bytes], error: BaseException | None = None):
        self._chunks = chunks
        self._error = error

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        del chunk_size
        yield from self._chunks
        if self._error is not None:
            raise self._error


@pytest.fixture(params=[save_url_image, save_url_video], ids=["image", "video"])
def url_saver(request):
    return request.param


def _cache_files(home: Path) -> list[Path]:
    return [path for path in home.rglob("*") if path.is_file()]


def test_url_cache_publishes_only_complete_download(tmp_path, monkeypatch, url_saver):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: _Response([b"first", b"second"]))

    result = url_saver("https://cdn.example/media.bin")

    assert result.read_bytes() == b"firstsecond"
    assert _cache_files(tmp_path) == [result]
    assert not result.name.startswith(".")


def test_url_cache_removes_partial_after_stream_failure(tmp_path, monkeypatch, url_saver):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    stream_error = RuntimeError("connection reset")
    monkeypatch.setattr(
        requests,
        "get",
        lambda *args, **kwargs: _Response([b"partial"], error=stream_error),
    )

    with pytest.raises(RuntimeError, match="connection reset"):
        url_saver("https://cdn.example/media.bin")

    assert _cache_files(tmp_path) == []


def test_url_cache_removes_partial_after_write_failure(tmp_path, monkeypatch, url_saver):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: _Response([b"first", b"second"]))
    original_open = Path.open

    class _FailingWriter:
        def __init__(self, raw):
            self._raw = raw
            self._writes = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self._raw.close()

        def write(self, chunk: bytes):
            self._writes += 1
            if self._writes == 2:
                raise OSError("disk full")
            return self._raw.write(chunk)

    def _open(path: Path, *args, **kwargs):
        raw = original_open(path, *args, **kwargs)
        if args and args[0] == "wb" and path.name.endswith(".part"):
            return _FailingWriter(raw)
        return raw

    monkeypatch.setattr(Path, "open", _open)

    with pytest.raises(OSError, match="disk full"):
        url_saver("https://cdn.example/media.bin")

    assert _cache_files(tmp_path) == []


@pytest.mark.parametrize(
    ("chunks", "max_bytes", "message"),
    [([], 10, "0 bytes"), ([b"too large"], 2, "exceeds")],
    ids=["empty", "oversize"],
)
def test_url_cache_rejections_leave_no_artifact(
    tmp_path, monkeypatch, url_saver, chunks, max_bytes, message
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: _Response(chunks))

    with pytest.raises(ValueError, match=message):
        url_saver("https://cdn.example/media.bin", max_bytes=max_bytes)

    assert _cache_files(tmp_path) == []
