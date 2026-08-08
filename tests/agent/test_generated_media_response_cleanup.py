import requests
import pytest

from agent import image_gen_provider, video_gen_provider


class _RecordingResponse:
    def __init__(self, chunks, *, content_type, http_error=False):
        self.chunks = chunks
        self.close_calls = 0
        self.headers = {"Content-Type": content_type}
        self.http_error = http_error

    def raise_for_status(self):
        if self.http_error:
            raise RuntimeError("HTTP 500")

    def iter_content(self, *, chunk_size):
        del chunk_size
        yield from self.chunks

    def close(self):
        self.close_calls += 1


@pytest.fixture(
    params=[
        (
            image_gen_provider,
            "save_url_image",
            "_images_cache_dir",
            "image/png",
        ),
        (
            video_gen_provider,
            "save_url_video",
            "_videos_cache_dir",
            "video/mp4",
        ),
    ],
    ids=["image", "video"],
)
def media_saver(request, monkeypatch, tmp_path):
    module, helper_name, cache_name, content_type = request.param
    monkeypatch.setattr(module, cache_name, lambda: tmp_path)
    return getattr(module, helper_name), content_type


def test_generated_media_download_closes_successful_response(
    media_saver,
    monkeypatch,
):
    save, content_type = media_saver
    response = _RecordingResponse([b"asset-bytes"], content_type=content_type)
    monkeypatch.setattr(requests, "get", lambda *_args, **_kwargs: response)

    path = save("https://example.test/asset", max_bytes=100)

    assert path.read_bytes() == b"asset-bytes"
    assert response.close_calls == 1


def test_generated_media_download_closes_oversized_response(
    media_saver,
    monkeypatch,
):
    save, content_type = media_saver
    response = _RecordingResponse([b"oversized"], content_type=content_type)
    monkeypatch.setattr(requests, "get", lambda *_args, **_kwargs: response)

    with pytest.raises(ValueError, match="exceeds"):
        save("https://example.test/asset", max_bytes=4)

    assert response.close_calls == 1


def test_generated_media_download_closes_http_error(
    media_saver,
    monkeypatch,
):
    save, content_type = media_saver
    response = _RecordingResponse(
        [],
        content_type=content_type,
        http_error=True,
    )
    monkeypatch.setattr(requests, "get", lambda *_args, **_kwargs: response)

    with pytest.raises(RuntimeError, match="HTTP 500"):
        save("https://example.test/asset")

    assert response.close_calls == 1
