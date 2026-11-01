"""Regression for #123132: a configured npm registry may be plain http.

The plan gate refused every non-https candidate before any transfer, so `hermes pm install`
could not provision npm from the very mirrors the registry routing was added for: closed
networks that serve a plain-http registry inside their perimeter, which the network-mirror
E2E cell only ever simulates as a loopback bridge.
"""
from urllib.request import Request

import pytest

from pm.downloader import Download, DownloadError, DownloadTransportError, Source, _HttpsRedirectHandler


def _plan(tmp_path, *sources):
    return Download(list(sources), partials_dir=tmp_path / "partials", connections=1)


def test_a_configured_registry_may_serve_a_plain_http_url(tmp_path, monkeypatch):
    """The origin is one the user configured; the lock's SHA256 still verifies the bytes."""
    source = Source("http://mirror.corp.example/npm/-/npm-12.0.2.tgz",
                    tmp_path / "npm.tgz", "a" * 64, allow_plain_http=True)
    probed: list[str] = []

    def probe(self, url):
        probed.append(url)
        raise DownloadTransportError(url, OSError("mirror unreachable"))

    monkeypatch.setattr(Download, "_probe", probe)
    with pytest.raises(DownloadTransportError):
        _plan(tmp_path, source).run()
    assert probed == [source.url]


def test_the_plain_http_exemption_does_not_leak(tmp_path):
    """Only the first opted-in candidate is exempt: unmarked URLs and fallbacks stay https-only."""
    unmarked = Source("http://mirror.corp.example/npm-12.0.2.tgz",
                      tmp_path / "unmarked.tgz", "a" * 64)
    with pytest.raises(ValueError, match="refusing non-https url"):
        _plan(tmp_path, unmarked).run()

    with_fallback = Source("https://registry.npmjs.org/npm/-/npm-12.0.2.tgz",
                           tmp_path / "fallback.tgz", "a" * 64,
                           fallbacks=("http://elsewhere.example/npm-12.0.2.tgz",),
                           allow_plain_http=True)
    with pytest.raises(ValueError, match="refusing non-https url"):
        _plan(tmp_path, with_fallback).run()


def test_a_plain_http_origin_keeps_plain_http_redirects():
    """A mirror redirecting inside its own perimeter works; https never downgrades."""
    handler = _HttpsRedirectHandler()
    followed = handler.redirect_request(
        Request("http://mirror.corp.example/npm-12.0.2.tgz"), None, 302, "Found", {},
        "http://mirror.corp.example/files/npm-12.0.2.tgz")
    assert followed.full_url == "http://mirror.corp.example/files/npm-12.0.2.tgz"

    with pytest.raises(DownloadError, match="refusing redirect to non-https url"):
        handler.redirect_request(
            Request("https://mirror.corp.example/npm-12.0.2.tgz"), None, 302, "Found", {},
            "http://mirror.corp.example/files/npm-12.0.2.tgz")
