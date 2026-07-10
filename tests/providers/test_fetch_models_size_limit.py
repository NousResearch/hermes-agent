"""Regression tests for ProviderProfile.fetch_models response-size bounding.

fetch_models() probes a provider's /models endpoint. The endpoint URL comes
from operator/config (base_url/models_url, e.g. a self-hosted or community
relay), so a misconfigured or hostile catalog endpoint must not be able to
force the process to buffer an unbounded body into memory. A timeout bounds
wall-clock, not size — the response itself has to be capped.
"""

import json

import providers.base as pbase
from providers.base import ProviderProfile


class _FakeResp:
    """Minimal stand-in for an http.client.HTTPResponse context manager."""

    def __init__(self, body: bytes, content_length=None):
        self._body = body
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)

    def read(self, n=-1):
        if n is None or n < 0:
            return self._body
        return self._body[:n]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _patch_urlopen(monkeypatch, resp):
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda req, timeout=None: resp
    )


def _profile():
    return ProviderProfile(name="test", base_url="https://api.example.test/v1")


def test_small_response_returns_model_ids(monkeypatch):
    body = json.dumps({"data": [{"id": "m1"}, {"id": "m2"}]}).encode()
    _patch_urlopen(monkeypatch, _FakeResp(body))
    assert _profile().fetch_models() == ["m1", "m2"]


def test_oversized_body_is_rejected(monkeypatch):
    # Cap to a tiny value so we don't allocate megabytes in the test.
    monkeypatch.setattr(pbase, "_MAX_MODELS_RESPONSE_BYTES", 32)
    body = json.dumps({"data": [{"id": "x" * 200}]}).encode()
    assert len(body) > 32
    # No Content-Length header → the post-read length guard must catch it.
    _patch_urlopen(monkeypatch, _FakeResp(body))
    assert _profile().fetch_models() is None


def test_oversized_content_length_header_is_rejected(monkeypatch):
    monkeypatch.setattr(pbase, "_MAX_MODELS_RESPONSE_BYTES", 32)
    # Small body, but the server advertises a huge Content-Length.
    _patch_urlopen(
        monkeypatch, _FakeResp(b'{"data": []}', content_length=10_000_000)
    )
    assert _profile().fetch_models() is None
