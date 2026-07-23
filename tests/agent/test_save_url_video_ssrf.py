"""SSRF invariants for agent.video_gen_provider.save_url_video."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_PROFILE", raising=False)


def _final_response(content_type: str = "video/mp4") -> MagicMock:
    resp = MagicMock()
    resp.headers = {"Content-Type": content_type}
    resp.raise_for_status = MagicMock()
    resp.is_redirect = False
    resp.iter_content = MagicMock(return_value=[b"\x00\x00\x00\x18ftypmp42"])
    resp.close = MagicMock()
    return resp


def _redirect_response(location: str) -> MagicMock:
    resp = MagicMock()
    resp.headers = {"Location": location}
    resp.is_redirect = True
    resp.close = MagicMock()
    return resp


def test_save_url_video_blocks_loopback():
    from agent.video_gen_provider import save_url_video

    with pytest.raises(ValueError, match="private or internal"):
        save_url_video("http://127.0.0.1:8080/clip.mp4")


def test_save_url_video_blocks_cloud_metadata():
    from agent.video_gen_provider import save_url_video

    with pytest.raises(ValueError, match="private or internal"):
        save_url_video("http://169.254.169.254/latest/meta-data/")


def test_save_url_video_blocks_redirect_to_metadata():
    from agent.video_gen_provider import save_url_video

    public = "https://cdn.example.com/clip.mp4"
    evil = "http://169.254.169.254/latest/meta-data/"

    with patch("tools.url_safety.is_safe_url", side_effect=lambda u: u == public), patch(
        "requests.get", return_value=_redirect_response(evil)
    ) as mock_get:
        with pytest.raises(ValueError, match="private or internal"):
            save_url_video(public)
        mock_get.assert_called()


def test_save_url_video_allows_public_url(tmp_path, monkeypatch):
    from agent.video_gen_provider import save_url_video

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    public = "https://cdn.example.com/clip.mp4"

    with patch("tools.url_safety.is_safe_url", return_value=True), patch(
        "requests.get", return_value=_final_response()
    ):
        path = save_url_video(public, prefix="test")
    assert path.exists()
    assert path.stat().st_size > 0
