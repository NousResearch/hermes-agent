"""Tests for save_url_image SSRF guard in agent.image_gen_provider."""
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    """Point HERMES_HOME at a temp dir so cache writes land safely."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_PROFILE", raising=False)


def test_save_url_image_blocks_private_loopback():
    """SSRF guard: save_url_image must reject loopback URLs."""
    from agent.image_gen_provider import save_url_image

    with pytest.raises(ValueError, match="private or internal"):
        save_url_image("http://127.0.0.1:8080/secret")


def test_save_url_image_blocks_cloud_metadata():
    """SSRF guard: save_url_image must reject cloud metadata endpoints."""
    from agent.image_gen_provider import save_url_image

    with pytest.raises(ValueError, match="private or internal"):
        save_url_image("http://169.254.169.254/latest/meta-data/")


def test_save_url_image_blocks_internal_host():
    """SSRF guard: save_url_image must reject internal network addresses."""
    from agent.image_gen_provider import save_url_image

    with pytest.raises(ValueError, match="private or internal"):
        save_url_image("http://10.0.0.1/admin")


def test_save_url_image_allows_public_url():
    """SSRF guard: save_url_image should proceed with valid public URLs."""
    from agent.image_gen_provider import save_url_image

    mock_response = MagicMock()
    mock_response.headers = {"Content-Type": "image/png"}
    mock_response.raise_for_status = MagicMock()
    mock_response.iter_content = MagicMock(return_value=[b"\x89PNG\r\n"])

    with patch("requests.get", return_value=mock_response),          patch("tools.url_safety.is_safe_url", return_value=True):
        path = save_url_image("https://api.x.ai/v1/images/abc.png")
        assert path.exists()
        assert path.name.endswith(".png")
