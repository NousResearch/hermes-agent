"""Tests for save_url_image SSRF guard in agent.image_gen_provider."""
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    """Point HERMES_HOME at a temp dir so cache writes land safely."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_PROFILE", raising=False)


def _mock_final_response(content_type="image/png"):
    """Create a mock non-redirect response."""
    resp = MagicMock()
    resp.headers = {"Content-Type": content_type}
    resp.raise_for_status = MagicMock()
    resp.iter_content = MagicMock(return_value=[b"\x89PNG\r\n"])
    resp.is_redirect = False
    resp.close = MagicMock()
    return resp


def _mock_redirect_response(location, content_type="image/png"):
    """Create a mock redirect response with a Location header."""
    resp = MagicMock()
    resp.headers = {"Content-Type": content_type, "Location": location}
    resp.raise_for_status = MagicMock()
    resp.is_redirect = True
    resp.close = MagicMock()
    return resp


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

    with patch("requests.get", return_value=_mock_final_response()), \
         patch("tools.url_safety.is_safe_url", return_value=True):
        path = save_url_image("https://api.x.ai/v1/images/abc.png")
        assert path.exists()
        assert path.name.endswith(".png")


def test_save_url_image_blocks_redirect_to_private():
    """SSRF guard: redirect from public URL to private must be blocked
    BEFORE the request to the private address is made."""
    from agent.image_gen_provider import save_url_image

    redirect_resp = _mock_redirect_response("http://127.0.0.1:8080/exfil")

    def _is_safe(url: str) -> bool:
        return "127.0.0.1" not in url

    with patch("requests.get", return_value=redirect_resp) as mock_get, \
         patch("tools.url_safety.is_safe_url", side_effect=_is_safe):
        with pytest.raises(ValueError, match="redirect target"):
            save_url_image("https://cdn.example.com/image.png")

        # Only ONE request should have been made (the initial one).
        # The private redirect target must NOT be fetched.
        assert mock_get.call_count == 1
        mock_get.assert_called_with(
            "https://cdn.example.com/image.png",
            timeout=60.0, stream=True, allow_redirects=False,
        )


def test_save_url_image_allows_redirect_to_public():
    """SSRF guard: redirect to another public URL should proceed."""
    from agent.image_gen_provider import save_url_image

    redirect_resp = _mock_redirect_response("https://cdn2.example.com/final.png")
    final_resp = _mock_final_response()

    with patch("requests.get", side_effect=[redirect_resp, final_resp]), \
         patch("tools.url_safety.is_safe_url", return_value=True):
        path = save_url_image("https://cdn1.example.com/image.png")
        assert path.exists()


def test_save_url_image_multi_hop_redirect_validates_each_hop():
    """SSRF guard: each hop in a redirect chain must be validated."""
    from agent.image_gen_provider import save_url_image

    hop1 = _mock_redirect_response("https://cdn2.example.com/step2")
    hop2 = _mock_redirect_response("http://10.0.0.1/internal")

    def _is_safe(url: str) -> bool:
        return "10.0.0.1" not in url

    with patch("requests.get", side_effect=[hop1, hop2]) as mock_get, \
         patch("tools.url_safety.is_safe_url", side_effect=_is_safe):
        with pytest.raises(ValueError, match="redirect target.*10\\.0\\.0\\.1"):
            save_url_image("https://cdn.example.com/image.png")

        # hop2's Location was private → third request must NOT happen
        # (only 2 requests: initial + safe redirect)
        assert mock_get.call_count == 2
        # The private URL (10.0.0.1) must never appear in any request
        for c in mock_get.call_args_list:
            assert "10.0.0.1" not in c.args[0]


def test_save_url_image_too_many_redirects_raises():
    """SSRF guard: redirect loop exceeding budget must raise ValueError."""
    from agent.image_gen_provider import save_url_image

    # 11 safe redirects (exceeds _MAX_REDIRECTS=10) — loop exhausts
    redirects = [
        _mock_redirect_response(f"https://cdn{i}.example.com/next")
        for i in range(11)
    ]

    with patch("requests.get", side_effect=redirects), \
         patch("tools.url_safety.is_safe_url", return_value=True):
        with pytest.raises(ValueError, match="too many redirects"):
            save_url_image("https://cdn.example.com/image.png")

    # All 11 redirect responses should have been closed
    for r in redirects:
        r.close.assert_called()
