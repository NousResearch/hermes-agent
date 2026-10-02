"""Compatibility of the Google OAuth dependency chain after the oauthlib update."""

from urllib.parse import parse_qs, urlsplit

import requests
from google_auth_oauthlib.flow import Flow
from requests_oauthlib import OAuth2Session


def test_authorization_code_flow_preserves_state_and_pkce():
    flow = Flow.from_client_config(
        {"web": {
            "client_id": "compat-client",
            "client_secret": "synthetic-unused-secret",
            "auth_uri": "https://identity.invalid/authorize",
            "token_uri": "https://identity.invalid/token",
            "redirect_uris": ["https://callback.invalid/oauth"],
        }},
        scopes=["openid", "email"],
        redirect_uri="https://callback.invalid/oauth",
        code_verifier="v" * 64,
        autogenerate_code_verifier=False,
    )
    session = flow.oauth2session
    authorization, state = flow.authorization_url(
        state="compat-state",
        code_challenge="compat-challenge",
        code_challenge_method="S256",
    )
    params = parse_qs(urlsplit(authorization).query)
    assert state == "compat-state"
    assert params["state"] == [state]
    assert params["redirect_uri"] == [session.redirect_uri]
    assert params["response_type"] == ["code"]
    assert params["code_challenge"] == ["compat-challenge"]
    assert params["code_challenge_method"] == ["S256"]


def test_token_exchange_and_refresh_use_oauth2_protocol(monkeypatch):
    session = OAuth2Session(client_id="compat-client", redirect_uri="https://callback.invalid/oauth")
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        response = requests.Response()
        response.status_code = 200
        response._content = (
            b'{"access_token":"synthetic-access","refresh_token":"synthetic-refresh",'
            b'"token_type":"Bearer","expires_in":3600}'
        )
        response.request = requests.Request(
            method, url, headers=kwargs.get("headers"), data=kwargs.get("data")
        ).prepare()
        return response

    monkeypatch.setattr(session, "request", request)
    token = session.fetch_token("https://identity.invalid/token", code="synthetic-code")
    assert token["token_type"] == "Bearer"
    assert calls[-1][0] == "POST"
    assert calls[-1][2]["data"]["grant_type"] == "authorization_code"
    assert calls[-1][2]["data"]["code"] == "synthetic-code"
    refreshed = session.refresh_token("https://identity.invalid/token")
    assert refreshed["refresh_token"] == token["refresh_token"]
    refresh_body = calls[-1][2]["data"]
    refresh_params = parse_qs(refresh_body) if isinstance(refresh_body, str) else refresh_body
    assert refresh_params["grant_type"] in ("refresh_token", ["refresh_token"])
    assert refresh_params["refresh_token"] in (token["refresh_token"], [token["refresh_token"]])
