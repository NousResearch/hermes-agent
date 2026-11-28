"""redact_cdp_url() masks a token that a CDP gateway carries in the URL path. Regression for #158758."""

import pytest
import requests
from urllib3 import HTTPSConnectionPool
from urllib3.exceptions import MaxRetryError

from agent.redact import redact_cdp_url

TOKEN = "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8s9T0u1V2w3X4y5Z6a7B8c9D0e1F2"
TARGET_ID = "0a1b2c3d-4e5f-4789-abcd-ef0123456789"
URL = f"https://cdp.example.com/{TOKEN}"


def test_masks_token_path_segment():
    assert redact_cdp_url(f"{URL}/json/version") == "https://cdp.example.com/***/json/version"


@pytest.mark.parametrize(
    "text",
    [
        f"Cannot connect to host {URL} ssl:default",
        f"Failed to reach {URL}: 401",
        f"({URL}).",
        f"{URL}==",
        f"HTTPS://CDP.EXAMPLE.COM/{TOKEN}",
        f"https://cdp.example.com/{TOKEN}?session=1",
        f"https://cdp.example.com//{TOKEN}",
    ],
)
def test_masks_token_in_surrounding_text(text):
    assert TOKEN not in redact_cdp_url(text)


def test_masks_token_in_requests_connection_error():
    # urllib3 names only the path of the failed request, without the scheme or host.
    pool = HTTPSConnectionPool("cdp.example.com", 443)
    exc = requests.exceptions.ConnectionError(MaxRetryError(pool, f"/{TOKEN}/json/version", "refused"))
    out = redact_cdp_url(str(exc))
    assert TOKEN not in out
    assert "url: /***/json/version" in out


def test_masks_token_when_redaction_is_disabled(monkeypatch):
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
    assert redact_cdp_url(URL) == "https://cdp.example.com/***"


def test_masks_token_before_devtools_target_id():
    out = redact_cdp_url(f"wss://cdp.example.com/{TOKEN}/devtools/browser/{TARGET_ID}")
    assert out == f"wss://cdp.example.com/***/devtools/browser/{TARGET_ID}"


@pytest.mark.parametrize("segment", [TOKEN, "0" * 34, "-" * 36])
def test_masks_token_in_place_of_devtools_target_id(segment):
    assert redact_cdp_url(f"wss://cdp.example.com/devtools/browser/{segment}") == "wss://cdp.example.com/devtools/browser/***"


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:9222",
        "http://127.0.0.1:9222/json/version",
        f"ws://localhost:9222/devtools/browser/{TARGET_ID}",
        "ws://127.0.0.1:9222/devtools/page/8F1A2B3C4D5E6F708192A3B4C5D6E7F8",
        "https://cdp.example.com/" + "a" * 31,
        "wss://chrome-1234567890.us-east-1.cdp.example.com/json/version",
    ],
)
def test_keeps_local_chrome_and_short_urls(url):
    assert redact_cdp_url(url) == url
