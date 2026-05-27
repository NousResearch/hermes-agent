"""Regression coverage for xAI OAuth PKCE token exchange (issue #26990).

Issue [#26990] reported that ``hermes auth add xai-oauth`` succeeds at the
browser-side authorize step but fails at the token endpoint with
``code_challenge is required`` — the symptom of an OAuth server that
re-validates PKCE at the token step instead of relying purely on
state captured during the authorize redirect.

The fix in ``hermes_cli/auth.py`` extracts the token POST into
:func:`_xai_oauth_exchange_code_for_tokens` and:

* Sends ``code_verifier`` (RFC 7636 §4.5 requirement).
* **Also** echoes ``code_challenge`` and ``code_challenge_method``
  in the request body as defense-in-depth — strictly compliant
  servers ignore extras at the token endpoint, but xAI's server
  needs them.
* Refuses to fire the POST locally when ``code_verifier`` is empty
  (avoids leaking the auth code to a server that can't redeem it).
* Surfaces the HTTP status code prominently in the error message so
  users / maintainers can tell a 400 (bad request) from a 403
  (entitlement denied) at a glance.

These tests pin all three behaviors so the fix can't silently regress.
"""

from __future__ import annotations

from typing import Any, Dict, List
from urllib.parse import parse_qs

import httpx
import pytest

from hermes_cli.auth import (
    AuthError,
    XAI_OAUTH_CLIENT_ID,
    _xai_oauth_exchange_code_for_tokens,
)


# ---------------------------------------------------------------------------
# httpx.post recorder
# ---------------------------------------------------------------------------


class _PostRecorder:
    """Capture every ``httpx.post`` call without touching the network."""

    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, url, *, headers=None, data=None, timeout=None, **kw):
        self.calls.append(
            {"url": url, "headers": headers or {}, "data": data or {},
             "timeout": timeout, "extra": kw}
        )
        return self.response


def _ok_response(payload: dict) -> httpx.Response:
    return httpx.Response(200, json=payload)


def _err_response(status: int, body: str) -> httpx.Response:
    return httpx.Response(status, text=body)


@pytest.fixture
def post_recorder(monkeypatch):
    """Default: 200 response with a full xAI token payload."""
    recorder = _PostRecorder(
        _ok_response(
            {
                "access_token": "AT-fresh",
                "refresh_token": "RT-fresh",
                "id_token": "ID",
                "expires_in": 3600,
                "token_type": "Bearer",
            }
        )
    )
    monkeypatch.setattr("hermes_cli.auth.httpx.post", recorder)
    return recorder


# ---------------------------------------------------------------------------
# Core contract: which fields go on the wire?
# ---------------------------------------------------------------------------


def test_token_exchange_includes_code_verifier(post_recorder):
    """RFC 7636 §4.5 — ``code_verifier`` MUST be sent."""
    _xai_oauth_exchange_code_for_tokens(
        token_endpoint="https://auth.x.ai/oauth2/token",
        code="AUTHCODE",
        redirect_uri="http://127.0.0.1:56121/callback",
        code_verifier="theVerifier_43_to_128_chars_____________________",
        code_challenge="aBcDeF",
    )
    sent = post_recorder.calls[-1]["data"]
    assert sent["code_verifier"] == "theVerifier_43_to_128_chars_____________________"


def test_token_exchange_also_echoes_code_challenge_for_xai(post_recorder):
    """Defense-in-depth for #26990 — xAI re-validates the challenge
    at the token endpoint, not just at authorize.  Without this echo
    we get ``code_challenge is required`` even though we send a valid
    ``code_verifier``."""
    _xai_oauth_exchange_code_for_tokens(
        token_endpoint="https://auth.x.ai/oauth2/token",
        code="AUTHCODE",
        redirect_uri="http://127.0.0.1:56121/callback",
        code_verifier="v" * 64,
        code_challenge="aBcDeF",
    )
    sent = post_recorder.calls[-1]["data"]
    assert sent["code_challenge"] == "aBcDeF"
    assert sent["code_challenge_method"] == "S256"


def test_token_exchange_uses_correct_grant_and_client(post_recorder):
    """Lock the static fields too — a future refactor must not flip
    these to ``client_credentials`` or drop ``client_id``."""
    _xai_oauth_exchange_code_for_tokens(
        token_endpoint="https://auth.x.ai/oauth2/token",
        code="AUTHCODE",
        redirect_uri="http://127.0.0.1:56121/callback",
        code_verifier="v" * 64,
        code_challenge="c" * 43,
    )
    sent = post_recorder.calls[-1]["data"]
    assert sent["grant_type"] == "authorization_code"
    assert sent["code"] == "AUTHCODE"
    assert sent["redirect_uri"] == "http://127.0.0.1:56121/callback"
    assert sent["client_id"] == XAI_OAUTH_CLIENT_ID


def test_token_exchange_uses_form_urlencoded_content_type(post_recorder):
    """xAI's token endpoint expects ``application/x-www-form-urlencoded``."""
    _xai_oauth_exchange_code_for_tokens(
        token_endpoint="https://auth.x.ai/oauth2/token",
        code="AUTHCODE",
        redirect_uri="http://127.0.0.1:56121/callback",
        code_verifier="v" * 64,
        code_challenge="c" * 43,
    )
    headers = post_recorder.calls[-1]["headers"]
    assert headers["Content-Type"] == "application/x-www-form-urlencoded"
    assert headers["Accept"] == "application/json"


def test_token_exchange_targets_the_supplied_endpoint(post_recorder):
    """Some test fixtures sniff the discovered token endpoint dynamically.
    We must POST to the URL the caller passed, not a hard-coded constant."""
    _xai_oauth_exchange_code_for_tokens(
        token_endpoint="https://auth.x.ai/some/other/token/path",
        code="AUTHCODE",
        redirect_uri="http://127.0.0.1:56121/callback",
        code_verifier="v" * 64,
        code_challenge="c" * 43,
    )
    assert post_recorder.calls[-1]["url"] == "https://auth.x.ai/some/other/token/path"


def test_token_exchange_passes_timeout_through(post_recorder):
    """Operators on slow networks pass a higher ``timeout_seconds``;
    the helper must forward it (and bump the floor to 20s)."""
    _xai_oauth_exchange_code_for_tokens(
        token_endpoint="https://auth.x.ai/oauth2/token",
        code="AUTHCODE",
        redirect_uri="http://127.0.0.1:56121/callback",
        code_verifier="v" * 64,
        code_challenge="c" * 43,
        timeout_seconds=45.0,
    )
    assert post_recorder.calls[-1]["timeout"] == 45.0


def test_token_exchange_floor_timeout_is_20s(post_recorder):
    _xai_oauth_exchange_code_for_tokens(
        token_endpoint="https://auth.x.ai/oauth2/token",
        code="AUTHCODE",
        redirect_uri="http://127.0.0.1:56121/callback",
        code_verifier="v" * 64,
        code_challenge="c" * 43,
        timeout_seconds=2.0,
    )
    assert post_recorder.calls[-1]["timeout"] == 20.0


# ---------------------------------------------------------------------------
# Sanity guard: refuse to POST with an empty code_verifier
# ---------------------------------------------------------------------------


def test_empty_code_verifier_raises_without_posting(post_recorder):
    """If ``code_verifier`` is somehow lost upstream, we must refuse to
    send the request — leaking an authorization code to xAI without a
    verifier is worse than failing locally with an actionable error."""
    with pytest.raises(AuthError) as exc_info:
        _xai_oauth_exchange_code_for_tokens(
            token_endpoint="https://auth.x.ai/oauth2/token",
            code="AUTHCODE",
            redirect_uri="http://127.0.0.1:56121/callback",
            code_verifier="",
            code_challenge="c" * 43,
        )
    assert exc_info.value.code == "xai_pkce_verifier_missing"
    assert "26990" in str(exc_info.value)
    # And critically: nothing was sent.
    assert post_recorder.calls == []


def test_missing_code_challenge_omits_echo_but_still_sends_verifier(post_recorder):
    """``code_challenge`` is defensive — if a caller doesn't have it
    handy, we must still send the standards-compliant request rather
    than refusing.  This keeps RFC-compliant servers happy."""
    _xai_oauth_exchange_code_for_tokens(
        token_endpoint="https://auth.x.ai/oauth2/token",
        code="AUTHCODE",
        redirect_uri="http://127.0.0.1:56121/callback",
        code_verifier="v" * 64,
        code_challenge="",
    )
    sent = post_recorder.calls[-1]["data"]
    assert sent["code_verifier"] == "v" * 64
    assert "code_challenge" not in sent
    assert "code_challenge_method" not in sent


# ---------------------------------------------------------------------------
# Error surfacing
# ---------------------------------------------------------------------------


def test_non_200_response_surfaces_status_and_body(monkeypatch):
    """When xAI returns a 4xx, the operator needs both the HTTP status
    code (to tell 400 from 401 from 403 at a glance) and the response
    body (the actual server-side reason)."""
    recorder = _PostRecorder(
        _err_response(400, '{"error":"invalid_grant","error_description":"code_challenge is required"}')
    )
    monkeypatch.setattr("hermes_cli.auth.httpx.post", recorder)
    with pytest.raises(AuthError) as exc_info:
        _xai_oauth_exchange_code_for_tokens(
            token_endpoint="https://auth.x.ai/oauth2/token",
            code="AUTHCODE",
            redirect_uri="http://127.0.0.1:56121/callback",
            code_verifier="v" * 64,
            code_challenge="c" * 43,
        )
    msg = str(exc_info.value)
    assert "HTTP 400" in msg, (
        "Status code must be in the error so callers can disambiguate "
        "tier-denied (403) from bad-request (400) without inspecting "
        "exc.code."
    )
    assert "code_challenge is required" in msg
    assert exc_info.value.code == "xai_token_exchange_failed"


def test_transport_error_wraps_as_auth_error(monkeypatch):
    """A connection failure must come back as ``AuthError`` so the
    surrounding ``format_auth_error`` UI mapping fires correctly."""

    def _boom(*args, **kwargs):
        raise httpx.ConnectError("dns failure")

    monkeypatch.setattr("hermes_cli.auth.httpx.post", _boom)
    with pytest.raises(AuthError) as exc_info:
        _xai_oauth_exchange_code_for_tokens(
            token_endpoint="https://auth.x.ai/oauth2/token",
            code="AUTHCODE",
            redirect_uri="http://127.0.0.1:56121/callback",
            code_verifier="v" * 64,
            code_challenge="c" * 43,
        )
    assert exc_info.value.code == "xai_token_exchange_failed"
    assert "dns failure" in str(exc_info.value)


def test_non_dict_payload_raises_invalid_json(monkeypatch):
    """xAI returning ``[]`` or a string at 200 is a server bug — fail
    with a precise error rather than crashing later in token storage."""
    recorder = _PostRecorder(_ok_response([1, 2, 3]))  # type: ignore[arg-type]
    monkeypatch.setattr("hermes_cli.auth.httpx.post", recorder)
    with pytest.raises(AuthError) as exc_info:
        _xai_oauth_exchange_code_for_tokens(
            token_endpoint="https://auth.x.ai/oauth2/token",
            code="AUTHCODE",
            redirect_uri="http://127.0.0.1:56121/callback",
            code_verifier="v" * 64,
            code_challenge="c" * 43,
        )
    assert exc_info.value.code == "xai_token_exchange_invalid"


def test_success_returns_full_payload_dict(post_recorder):
    """200 happy path: the parsed JSON dict comes back verbatim so the
    caller can pluck ``access_token`` / ``refresh_token`` etc."""
    out = _xai_oauth_exchange_code_for_tokens(
        token_endpoint="https://auth.x.ai/oauth2/token",
        code="AUTHCODE",
        redirect_uri="http://127.0.0.1:56121/callback",
        code_verifier="v" * 64,
        code_challenge="c" * 43,
    )
    assert out["access_token"] == "AT-fresh"
    assert out["refresh_token"] == "RT-fresh"


# ---------------------------------------------------------------------------
# Wire-format guard: httpx must serialise ``data`` as form-urlencoded
# ---------------------------------------------------------------------------


def test_wire_format_is_form_urlencoded_with_all_pkce_fields(monkeypatch):
    """End-to-end check on the actual bytes httpx puts on the wire.
    If anyone ever swaps ``data=`` for ``json=`` or refactors the dict,
    xAI will start rejecting again — this catches it locally."""

    captured: Dict[str, Any] = {}

    class _Transport(httpx.BaseTransport):
        def handle_request(self, request):
            captured["body"] = bytes(request.read())
            captured["content_type"] = request.headers.get("content-type", "")
            return httpx.Response(
                200,
                json={"access_token": "AT", "refresh_token": "RT",
                      "id_token": "", "expires_in": 60, "token_type": "Bearer"},
            )

    real_post = httpx.post

    def _post(*args, **kwargs):
        with httpx.Client(transport=_Transport()) as c:
            return c.post(*args, **kwargs)

    monkeypatch.setattr("hermes_cli.auth.httpx.post", _post)

    _xai_oauth_exchange_code_for_tokens(
        token_endpoint="https://auth.x.ai/oauth2/token",
        code="AUTHCODE",
        redirect_uri="http://127.0.0.1:56121/callback",
        code_verifier="theVerifier_43+",
        code_challenge="theChallenge_43+",
    )

    assert "application/x-www-form-urlencoded" in captured["content_type"]
    parsed = parse_qs(captured["body"].decode())
    assert parsed["grant_type"] == ["authorization_code"]
    assert parsed["code"] == ["AUTHCODE"]
    assert parsed["redirect_uri"] == ["http://127.0.0.1:56121/callback"]
    assert parsed["client_id"] == [XAI_OAUTH_CLIENT_ID]
    assert parsed["code_verifier"] == ["theVerifier_43+"]
    assert parsed["code_challenge"] == ["theChallenge_43+"]
    assert parsed["code_challenge_method"] == ["S256"]
