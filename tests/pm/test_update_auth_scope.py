"""Index credentials stay on their exact HTTPS origin, including redirects."""
from __future__ import annotations

from email.message import Message
import io
import urllib.request
import urllib.response

import pytest

from pm import update


class IndexTransport(urllib.request.HTTPHandler, urllib.request.HTTPSHandler):
    def __init__(self, redirects=None):
        super().__init__()
        self.redirects = redirects or {}
        self.sent = []

    def http_open(self, request):
        self.sent.append((request.full_url, dict(request.header_items())))
        headers = Message()
        destination = self.redirects.get(request.full_url)
        if destination:
            headers["Location"] = destination
        response = urllib.response.addinfourl(
            io.BytesIO(b'{}'), headers, request.full_url, 302 if destination else 200,
        )
        response.msg = "Found" if destination else "OK"
        return response

    https_open = http_open


@pytest.mark.parametrize("url,token", [
    ("https://api.github.com/repos/example/tool/releases", "dummy-gh"),
    ("https://api.github.com:443/repos/example/tool/releases", "dummy-gh"),
    ("https://huggingface.co/api/buckets/example/tool/tree", "dummy-hf"),
    ("https://nodejs.org/dist/index.json", None),
    ("https://registry.npmjs.org/npm", None),
    ("https://evilhuggingface.co/index", None),
    ("https://api.github.com.evil.invalid/index", None),
    ("https://api.github.com:444/index", None),
    ("http://api.github.com/index", None),
])
def test_index_credentials_match_the_exact_origin(monkeypatch, url, token):
    transport = IndexTransport()
    monkeypatch.setattr(urllib.request, "_opener", urllib.request.build_opener(transport))
    monkeypatch.setenv("GH_TOKEN", "dummy-gh")
    monkeypatch.setenv("HF_TOKEN", "dummy-hf")

    assert update._get_json(url) == {}

    headers = {key.lower(): value for key, value in transport.sent[-1][1].items()}
    assert headers.get("authorization") == (f"Bearer {token}" if token else None)
    assert headers["user-agent"] == "hermes-pm"


@pytest.mark.parametrize("reader", [update._get_json, update._get_text], ids=["json", "text"])
@pytest.mark.parametrize("destination,credential_survives", [
    ("https://api.github.com/second", True),
    ("https://unrelated.invalid/second", False),
    ("https://api.github.com:444/second", False),
    ("http://api.github.com/second", False),
])
def test_index_redirects_use_the_credential_safe_opener(monkeypatch, reader, destination, credential_survives):
    origin = "https://api.github.com/first"
    transport = IndexTransport({origin: destination})
    monkeypatch.setattr(urllib.request, "_opener", urllib.request.build_opener(transport))
    monkeypatch.setenv("GH_TOKEN", "dummy-gh")

    reader(origin)

    assert [url for url, _ in transport.sent] == [origin, destination]
    initial, redirected = [{key.lower(): value for key, value in headers.items()} for _, headers in transport.sent]
    assert initial["authorization"] == "Bearer dummy-gh"
    assert (redirected.get("authorization") == "Bearer dummy-gh") is credential_survives
