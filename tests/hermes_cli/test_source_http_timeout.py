"""Source release reads recover from a single transport timeout (#139176)."""
import io
import json
from urllib.error import HTTPError, URLError

import pytest

from hermes_cli import github_api, source_check, source_releases

pytestmark = pytest.mark.real_release_channels


@pytest.mark.parametrize("wrapped", [False, True])
def test_stable_lookup_recovers_from_cold_timeout(monkeypatch, wrapped):
    calls = []
    sha = "a" * 40

    def open_request(request, timeout):
        calls.append((request.full_url, timeout, request.get_header("Authorization")))
        if len(calls) == 1:
            error = TimeoutError("handshake timed out")
            raise URLError(error) if wrapped else error
        payload = ({"tag_name": "v1.2.3", "draft": False, "prerelease": False}
                   if request.full_url.endswith("/releases/latest") else {"sha": sha})
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(github_api, "github_token", lambda: "fixture-token")
    monkeypatch.setattr(source_check.urllib.request, "urlopen", open_request)
    try:
        result = source_releases.resolve_source_target("stable")
    except ValueError as exc:
        pytest.fail(f"a single cold timeout aborted the stable lookup: {exc}")
    assert result.commit == sha
    assert len(calls) == 3
    assert calls[0] == calls[1]
    assert all(call[2] == "Bearer fixture-token" for call in calls)


@pytest.mark.parametrize("code", [401, 403, 429, 500])
def test_http_failures_keep_existing_auth_fallback(monkeypatch, code):
    calls = []
    error = HTTPError("https://api.github.com/fixture", code, "fixture", {}, None)

    def open_request(request, timeout):
        calls.append(request.get_header("Authorization"))
        raise error

    monkeypatch.setattr(github_api, "github_token", lambda: "fixture-token")
    monkeypatch.setattr(source_check.urllib.request, "urlopen", open_request)
    with pytest.raises(HTTPError) as caught:
        source_check._request("https://api.github.com/fixture")
    assert caught.value is error
    assert calls == (["Bearer fixture-token", None] if code == 401 else ["Bearer fixture-token"])
    assert error.hermes_authenticated is (code != 401)


@pytest.mark.parametrize("wrapped", [False, True])
def test_exhausted_timeout_keeps_failure_and_matching_budget(monkeypatch, wrapped):
    calls = []
    fault = TimeoutError("fixture")
    error = URLError(fault) if wrapped else fault
    monkeypatch.setattr(github_api, "GITHUB_REQUEST_TIMEOUT_SECONDS", 37)
    monkeypatch.setattr(github_api, "github_token", lambda: None)

    def open_request(request, timeout):
        calls.append(timeout)
        raise error

    monkeypatch.setattr(source_check.urllib.request, "urlopen", open_request)
    with pytest.raises(type(error)) as caught:
        source_check._request("https://api.github.com/fixture")
    assert caught.value is error
    assert calls == [37, 37]
    message = github_api.describe_github_failure(error, authenticated=False)
    assert "per-attempt timeout: 37 seconds" in message


@pytest.mark.parametrize("body", [b"", b"\xef\xbb\xbf result \n"])
def test_success_preserves_body_decoding_and_closes_response(monkeypatch, body):
    response = io.BytesIO(body)
    calls = []

    def open_request(request, timeout):
        calls.append(timeout)
        return response

    monkeypatch.setattr(source_check.urllib.request, "urlopen", open_request)
    assert source_check._request_with("https://api.github.com/fixture", "application/json", None) == body.decode("utf-8-sig").strip()
    assert len(calls) == 1
    assert response.closed


def test_non_timeout_transport_error_does_not_retry(monkeypatch):
    calls = []
    error = URLError(ConnectionRefusedError("fixture"))

    def open_request(request, timeout):
        calls.append(request)
        raise error

    monkeypatch.setattr(source_check.urllib.request, "urlopen", open_request)
    with pytest.raises(URLError) as caught:
        source_check._request_with("https://api.github.com/fixture", "application/json", None)
    assert caught.value is error
    assert len(calls) == 1
