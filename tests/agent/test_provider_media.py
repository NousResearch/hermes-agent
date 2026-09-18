from __future__ import annotations

import pytest

from agent import provider_media


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


def _stub_fetch(monkeypatch, handler):
    """Route save_url's fetch through an httpx MockTransport and stub the URL
    policy check — SSRF behavior has dedicated tests in test_save_url_image.py;
    these pin the streaming/content-type/headers contract."""
    import httpx

    monkeypatch.setattr("tools.url_safety.is_safe_url", lambda url: True)
    monkeypatch.setattr(
        "tools.url_safety.create_ssrf_safe_client",
        lambda **kw: httpx.Client(transport=httpx.MockTransport(handler), **kw),
    )


def test_save_url_forwards_headers_and_streams_to_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    calls = []

    import httpx

    def handler(request):
        calls.append(request)
        return httpx.Response(200, headers={"Content-Type": "video/mp4"}, content=b"clip")

    _stub_fetch(monkeypatch, handler)

    path = _save_video(
        "https://api.example/videos/job/content",
        headers={"Authorization": "Bearer test"},
        require_known_content_type=True,
    )

    assert path.read_bytes() == b"clip"
    assert path.suffix == ".mp4"
    assert len(calls) == 1
    assert str(calls[0].url) == "https://api.example/videos/job/content"
    assert calls[0].headers["Authorization"] == "Bearer test"


def test_save_url_strict_content_type_rejects_non_video(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    import httpx

    _stub_fetch(
        monkeypatch,
        lambda request: httpx.Response(200, headers={"Content-Type": "text/html"}, content=b"not a video"),
    )

    with pytest.raises(ValueError, match="unexpected Content-Type text/html"):
        _save_video(
            "https://api.example/videos/job/content",
            require_known_content_type=True,
        )

    assert not list(provider_media.cache_dir("videos").iterdir())


def test_save_url_redirect_does_not_forward_caller_headers(monkeypatch, tmp_path):
    """Caller auth headers (provider base_url fetches) go to the first hop only —
    a redirect target must never receive them."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    calls = []

    import httpx

    def handler(request):
        calls.append(request)
        if request.url.path == "/start":
            return httpx.Response(302, headers={"Location": "/final"})
        return httpx.Response(200, headers={"Content-Type": "video/mp4"}, content=b"clip")

    _stub_fetch(monkeypatch, handler)

    path = _save_video(
        "https://api.example/start",
        headers={"Authorization": "Bearer test"},
        require_known_content_type=True,
    )

    assert path.read_bytes() == b"clip"
    assert len(calls) == 2
    assert calls[0].headers.get("Authorization") == "Bearer test"
    assert calls[1].headers.get("Authorization") is None
