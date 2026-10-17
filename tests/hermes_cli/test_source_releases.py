"""The release-channel HTTP reader's credential handling.

stable/canary resolution reads api.github.com anonymously by default; behind a
shared exit the 60/hour per-IP budget 403s and the failure surfaces as "No
published stable release could be verified". The reader must send the configured
credential on GitHub API requests only, and fall back to anonymous on a 401.
"""
import urllib.error
import urllib.request

import pytest

from hermes_cli import source_releases


class _FakeResponse:
    def __init__(self, body=b""):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, _n=-1):
        return self._body


def _capture(monkeypatch, responses):
    """Route urlopen through a recorder that captures each request's headers."""
    calls = []

    def urlopen(request, *args, **kwargs):
        is_req = isinstance(request, urllib.request.Request)
        calls.append({
            "url": request.full_url if is_req else request,
            "headers": dict(request.headers) if is_req else {},
        })
        response = responses[len(calls) - 1]
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return calls


def test_github_api_headers_carry_the_token():
    headers = source_releases._github_api_headers("https://api.github.com/repos/x/y", "tok")
    assert headers["Authorization"] == "Bearer tok"


def test_asset_host_headers_never_carry_the_token():
    headers = source_releases._github_api_headers(
        "https://hermes-assets.nousresearch.com/releases/stable/index.html", "tok")
    assert "Authorization" not in headers


def test_anonymous_github_headers_have_no_authorization():
    headers = source_releases._github_api_headers("https://api.github.com/repos/x/y", None)
    assert "Authorization" not in headers


def test_read_authenticates_github_but_not_the_asset_host(monkeypatch):
    monkeypatch.setattr(source_releases, "github_token", lambda: "tok")
    calls = _capture(monkeypatch, [_FakeResponse(b"{}"), _FakeResponse(b"ok")])
    assert source_releases._read("https://api.github.com/repos/x/y") == "{}"
    assert source_releases._read("https://hermes-assets.nousresearch.com/releases/stable/index.html") == "ok"
    assert calls[0]["headers"].get("Authorization") == "Bearer tok"
    assert "Authorization" not in calls[1]["headers"]


def test_read_falls_back_to_anonymous_when_token_is_rejected(monkeypatch):
    monkeypatch.setattr(source_releases, "github_token", lambda: "stale-token")
    rejected = urllib.error.HTTPError("https://api.github.com/repos/x/y", 401, "Unauthorized", {}, None)
    calls = _capture(monkeypatch, [rejected, _FakeResponse(b"ok")])
    assert source_releases._read("https://api.github.com/repos/x/y") == "ok"
    assert calls[0]["headers"].get("Authorization") == "Bearer stale-token"
    assert "Authorization" not in calls[1]["headers"]


def test_read_never_resolves_a_token_for_the_asset_host(monkeypatch):
    def boom():
        pytest.fail("must not resolve a token for the asset host")

    monkeypatch.setattr(source_releases, "github_token", boom)
    calls = _capture(monkeypatch, [_FakeResponse(b"ok")])
    assert source_releases._read("https://hermes-assets.nousresearch.com/releases/stable/index.html") == "ok"
    assert "Authorization" not in calls[0]["headers"]
