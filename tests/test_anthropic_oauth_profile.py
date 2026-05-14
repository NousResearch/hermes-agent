"""Tests for the overridable Anthropic OAuth client profile.

Covers:
- ``get_anthropic_oauth_client_profile()`` env-var resolution (defaults,
  per-field overrides, empty-string + whitespace fallback, fresh read on
  every call).
- Back-compat module-level ``_OAUTH_*`` aliases match the defaults.
- ``run_hermes_oauth_login_pure()`` (PKCE login) builds the authorize URL
  and token-exchange POST from the profile.
- ``refresh_anthropic_oauth_pure()`` puts the configured token URL first
  while preserving the historical fallbacks.
"""
from __future__ import annotations

import json
from io import BytesIO
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from agent import anthropic_adapter
from agent.anthropic_adapter import (
    AnthropicOAuthClientProfile,
    _DEFAULT_OAUTH_AUTHORIZE_URL,
    _DEFAULT_OAUTH_CLIENT_ID,
    _DEFAULT_OAUTH_REDIRECT_URI,
    _DEFAULT_OAUTH_SCOPES,
    _DEFAULT_OAUTH_TOKEN_URL,
    _OAUTH_CLIENT_ID,
    _OAUTH_REDIRECT_URI,
    _OAUTH_SCOPES,
    _OAUTH_TOKEN_URL,
    get_anthropic_oauth_client_profile,
    refresh_anthropic_oauth_pure,
    run_hermes_oauth_login_pure,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_ENV_VARS = (
    "HERMES_ANTHROPIC_OAUTH_CLIENT_ID",
    "HERMES_ANTHROPIC_OAUTH_AUTHORIZE_URL",
    "HERMES_ANTHROPIC_OAUTH_TOKEN_URL",
    "HERMES_ANTHROPIC_OAUTH_REDIRECT_URI",
    "HERMES_ANTHROPIC_OAUTH_SCOPES",
)


@pytest.fixture(autouse=True)
def _clean_oauth_env(monkeypatch):
    """Strip every HERMES_ANTHROPIC_OAUTH_* var so each test starts clean."""
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _fake_token_response(payload: dict) -> MagicMock:
    """Build a ``urllib.request.urlopen``-shaped context manager."""
    body = json.dumps(payload).encode()
    handle = MagicMock()
    handle.read.return_value = body
    cm = MagicMock()
    cm.__enter__.return_value = handle
    cm.__exit__.return_value = False
    return cm


# ---------------------------------------------------------------------------
# Resolver — env-var resolution
# ---------------------------------------------------------------------------


class TestResolverDefaults:
    def test_returns_namedtuple_with_five_fields(self):
        profile = get_anthropic_oauth_client_profile()
        assert isinstance(profile, AnthropicOAuthClientProfile)
        assert profile._fields == (
            "client_id",
            "authorize_url",
            "token_url",
            "redirect_uri",
            "scopes",
        )

    def test_clean_env_returns_all_defaults(self):
        profile = get_anthropic_oauth_client_profile()
        assert profile.client_id == _DEFAULT_OAUTH_CLIENT_ID
        assert profile.authorize_url == _DEFAULT_OAUTH_AUTHORIZE_URL
        assert profile.token_url == _DEFAULT_OAUTH_TOKEN_URL
        assert profile.redirect_uri == _DEFAULT_OAUTH_REDIRECT_URI
        assert profile.scopes == _DEFAULT_OAUTH_SCOPES

    def test_default_values_match_hermes_registered_client(self):
        # Pin the upstream identifiers — anyone changing these is changing
        # which OAuth client Hermes ships to authorise as.
        assert _DEFAULT_OAUTH_CLIENT_ID == "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
        assert _DEFAULT_OAUTH_AUTHORIZE_URL == "https://claude.ai/oauth/authorize"
        assert _DEFAULT_OAUTH_TOKEN_URL == "https://console.anthropic.com/v1/oauth/token"
        assert _DEFAULT_OAUTH_REDIRECT_URI == "https://console.anthropic.com/oauth/code/callback"
        assert _DEFAULT_OAUTH_SCOPES == "org:create_api_key user:profile user:inference"


class TestResolverOverrides:
    @pytest.mark.parametrize(
        "env_name,field,override",
        [
            ("HERMES_ANTHROPIC_OAUTH_CLIENT_ID", "client_id", "custom-client-id"),
            ("HERMES_ANTHROPIC_OAUTH_AUTHORIZE_URL", "authorize_url", "https://example.test/authorize"),
            ("HERMES_ANTHROPIC_OAUTH_TOKEN_URL", "token_url", "https://example.test/token"),
            ("HERMES_ANTHROPIC_OAUTH_REDIRECT_URI", "redirect_uri", "https://example.test/cb"),
            ("HERMES_ANTHROPIC_OAUTH_SCOPES", "scopes", "scope:a scope:b"),
        ],
    )
    def test_individual_override(self, monkeypatch, env_name, field, override):
        monkeypatch.setenv(env_name, override)
        profile = get_anthropic_oauth_client_profile()
        assert getattr(profile, field) == override
        # Other fields stay on their defaults.
        for other_field in set(profile._fields) - {field}:
            default_name = f"_DEFAULT_OAUTH_{other_field.upper()}"
            assert getattr(profile, other_field) == getattr(anthropic_adapter, default_name)

    def test_all_five_overrides_together(self, monkeypatch):
        monkeypatch.setenv("HERMES_ANTHROPIC_OAUTH_CLIENT_ID", "cid")
        monkeypatch.setenv("HERMES_ANTHROPIC_OAUTH_AUTHORIZE_URL", "https://example.test/a")
        monkeypatch.setenv("HERMES_ANTHROPIC_OAUTH_TOKEN_URL", "https://example.test/t")
        monkeypatch.setenv("HERMES_ANTHROPIC_OAUTH_REDIRECT_URI", "https://example.test/r")
        monkeypatch.setenv("HERMES_ANTHROPIC_OAUTH_SCOPES", "s1 s2")
        profile = get_anthropic_oauth_client_profile()
        assert profile == AnthropicOAuthClientProfile(
            client_id="cid",
            authorize_url="https://example.test/a",
            token_url="https://example.test/t",
            redirect_uri="https://example.test/r",
            scopes="s1 s2",
        )

    @pytest.mark.parametrize("value", ["", "   ", "\t", "\n"])
    def test_empty_or_whitespace_env_falls_through(self, monkeypatch, value):
        monkeypatch.setenv("HERMES_ANTHROPIC_OAUTH_CLIENT_ID", value)
        profile = get_anthropic_oauth_client_profile()
        assert profile.client_id == _DEFAULT_OAUTH_CLIENT_ID

    def test_resolver_rereads_env_on_every_call(self, monkeypatch):
        # Call once with default, then set override, then call again. The
        # second call must pick up the new value (no caching).
        first = get_anthropic_oauth_client_profile()
        assert first.client_id == _DEFAULT_OAUTH_CLIENT_ID

        monkeypatch.setenv("HERMES_ANTHROPIC_OAUTH_CLIENT_ID", "rotated-cid")
        second = get_anthropic_oauth_client_profile()
        assert second.client_id == "rotated-cid"

        # And unset → fall back to default again.
        monkeypatch.delenv("HERMES_ANTHROPIC_OAUTH_CLIENT_ID", raising=False)
        third = get_anthropic_oauth_client_profile()
        assert third.client_id == _DEFAULT_OAUTH_CLIENT_ID


# ---------------------------------------------------------------------------
# Backward-compat aliases (used by hermes_cli/web_server.py imports)
# ---------------------------------------------------------------------------


class TestModuleAliases:
    """The four ``_OAUTH_*`` names are kept as module attributes because
    ``hermes_cli.web_server`` imports them by name. They reflect the values
    present at module import time. The conftest's hermetic-env fixture
    runs before the test module imports — so on a clean test run the
    aliases equal the defaults.
    """

    def test_client_id_alias(self):
        assert _OAUTH_CLIENT_ID == _DEFAULT_OAUTH_CLIENT_ID

    def test_token_url_alias(self):
        assert _OAUTH_TOKEN_URL == _DEFAULT_OAUTH_TOKEN_URL

    def test_redirect_uri_alias(self):
        assert _OAUTH_REDIRECT_URI == _DEFAULT_OAUTH_REDIRECT_URI

    def test_scopes_alias(self):
        assert _OAUTH_SCOPES == _DEFAULT_OAUTH_SCOPES


# ---------------------------------------------------------------------------
# PKCE login flow — run_hermes_oauth_login_pure()
# ---------------------------------------------------------------------------


class TestRunHermesOAuthLoginPure:
    """End-to-end login flow with input/urlopen/webbrowser mocked.

    The function:
      1. generates a PKCE pair
      2. builds an authorize URL and prints it
      3. opens the browser (best-effort)
      4. reads the user's pasted ``<code>#<state>`` value
      5. POSTs an authorization-code grant to the token URL
      6. returns the parsed token response
    """

    def _patched_login(
        self,
        monkeypatch,
        capsys,
        *,
        env_overrides: dict | None = None,
        pasted_code: str = "abc123#state-value",
        token_response: dict | None = None,
    ):
        """Run ``run_hermes_oauth_login_pure`` with safe mocks. Returns
        the captured ``urllib.request.Request`` plus the printed authorize URL."""
        if env_overrides:
            for k, v in env_overrides.items():
                monkeypatch.setenv(k, v)
        if token_response is None:
            token_response = {
                "access_token": "sk-ant-oat-fake",
                "refresh_token": "sk-ant-ort-fake",
                "expires_in": 3600,
            }
        captured: dict = {}

        def _capture_request(url, data=None, headers=None, method=None):
            req = MagicMock()
            req.full_url = url
            req.data = data
            req.headers = headers or {}
            req.method = method
            captured["request"] = req
            return req

        monkeypatch.setattr(
            "agent.anthropic_adapter.webbrowser.open",
            lambda url: True,
            raising=False,
        ) if False else None  # webbrowser is imported inside the function

        # Patch the symbols that get imported INSIDE the function body.
        with patch("webbrowser.open", return_value=True), \
             patch("builtins.input", return_value=pasted_code), \
             patch("urllib.request.Request", side_effect=_capture_request) as req_factory, \
             patch("urllib.request.urlopen", return_value=_fake_token_response(token_response)):
            result = run_hermes_oauth_login_pure()

        printed = capsys.readouterr().out
        captured["result"] = result
        captured["printed"] = printed
        captured["req_factory_calls"] = req_factory.call_args_list
        return captured

    def test_defaults_use_upstream_endpoints(self, monkeypatch, capsys):
        bag = self._patched_login(monkeypatch, capsys)

        # Authorize URL printed to the terminal must be on the default host.
        assert _DEFAULT_OAUTH_AUTHORIZE_URL in bag["printed"]

        # Parse the printed authorize URL and check key params.
        # The URL appears on its own line — pluck it out.
        printed_lines = [ln.strip() for ln in bag["printed"].splitlines()]
        auth_lines = [ln for ln in printed_lines if ln.startswith(_DEFAULT_OAUTH_AUTHORIZE_URL)]
        assert auth_lines, "authorize URL not printed"
        parsed = urlparse(auth_lines[0])
        qs = parse_qs(parsed.query)
        assert qs["client_id"] == [_DEFAULT_OAUTH_CLIENT_ID]
        assert qs["redirect_uri"] == [_DEFAULT_OAUTH_REDIRECT_URI]
        assert qs["scope"] == [_DEFAULT_OAUTH_SCOPES]
        assert qs["code_challenge_method"] == ["S256"]
        assert "code_challenge" in qs and qs["code_challenge"]
        assert "state" in qs and qs["state"]

        # Token exchange must POST to the default token URL with the
        # default client_id + redirect_uri in a JSON body.
        req = bag["request"]
        assert req.full_url == _DEFAULT_OAUTH_TOKEN_URL
        assert req.method == "POST"
        body = json.loads(req.data.decode())
        assert body["grant_type"] == "authorization_code"
        assert body["client_id"] == _DEFAULT_OAUTH_CLIENT_ID
        assert body["redirect_uri"] == _DEFAULT_OAUTH_REDIRECT_URI
        assert body["code"] == "abc123"
        assert body["state"] == "state-value"
        assert body["code_verifier"]  # PKCE verifier present

        # And the function returns the parsed token payload bits we care about.
        assert bag["result"]["access_token"] == "sk-ant-oat-fake"

    def test_overrides_flow_through_to_authorize_url_and_token_post(self, monkeypatch, capsys):
        bag = self._patched_login(
            monkeypatch,
            capsys,
            env_overrides={
                "HERMES_ANTHROPIC_OAUTH_CLIENT_ID": "override-cid",
                "HERMES_ANTHROPIC_OAUTH_AUTHORIZE_URL": "https://example.test/authorize",
                "HERMES_ANTHROPIC_OAUTH_TOKEN_URL": "https://example.test/token",
                "HERMES_ANTHROPIC_OAUTH_REDIRECT_URI": "https://example.test/cb",
                "HERMES_ANTHROPIC_OAUTH_SCOPES": "scope:custom",
            },
        )

        # Authorize URL printed uses the overridden host and params.
        assert "https://example.test/authorize?" in bag["printed"]
        auth_line = next(
            ln.strip()
            for ln in bag["printed"].splitlines()
            if ln.strip().startswith("https://example.test/authorize")
        )
        qs = parse_qs(urlparse(auth_line).query)
        assert qs["client_id"] == ["override-cid"]
        assert qs["redirect_uri"] == ["https://example.test/cb"]
        assert qs["scope"] == ["scope:custom"]

        # Token-exchange POST goes to the overridden token URL with the
        # overridden client_id + redirect_uri.
        req = bag["request"]
        assert req.full_url == "https://example.test/token"
        body = json.loads(req.data.decode())
        assert body["client_id"] == "override-cid"
        assert body["redirect_uri"] == "https://example.test/cb"

    def test_empty_paste_returns_none_without_calling_token_endpoint(self, monkeypatch, capsys):
        with patch("webbrowser.open", return_value=True), \
             patch("builtins.input", return_value=""), \
             patch("urllib.request.urlopen") as mock_urlopen:
            result = run_hermes_oauth_login_pure()
        assert result is None
        assert mock_urlopen.call_count == 0


# ---------------------------------------------------------------------------
# Refresh flow — refresh_anthropic_oauth_pure()
# ---------------------------------------------------------------------------


class TestRefreshAnthropicOAuthPure:
    """Refresh tries ``profile.token_url`` first, then keeps the historical
    upstream endpoints as fallbacks. Order matters — we assert on the
    exact sequence of URLs passed to ``urllib.request.Request``."""

    def _run_refresh(
        self,
        monkeypatch,
        *,
        env_overrides: dict | None = None,
        urlopen_side_effect=None,
        token_response: dict | None = None,
    ):
        if env_overrides:
            for k, v in env_overrides.items():
                monkeypatch.setenv(k, v)
        if token_response is None:
            token_response = {
                "access_token": "sk-ant-oat-new",
                "refresh_token": "sk-ant-ort-new",
                "expires_in": 3600,
            }
        url_calls: list = []

        def _capture_request(url, data=None, headers=None, method=None):
            url_calls.append(url)
            req = MagicMock()
            req.full_url = url
            req.data = data
            req.headers = headers or {}
            req.method = method
            return req

        if urlopen_side_effect is None:
            urlopen_side_effect = [_fake_token_response(token_response)]

        with patch("urllib.request.Request", side_effect=_capture_request), \
             patch("urllib.request.urlopen", side_effect=urlopen_side_effect):
            try:
                result = refresh_anthropic_oauth_pure("sk-ant-ort-old", use_json=False)
            except Exception as exc:  # propagate so tests can assert on failure
                result = exc
        return url_calls, result

    def test_default_endpoint_called_first_and_returns_on_success(self, monkeypatch):
        # The refresh loop exits on the first successful endpoint, so only
        # one URL is exercised when nothing fails. That URL is the
        # configured profile.token_url (== upstream default in this case).
        urls, result = self._run_refresh(monkeypatch)
        assert urls == [_DEFAULT_OAUTH_TOKEN_URL]
        assert result["access_token"] == "sk-ant-oat-new"

    def test_full_endpoint_sequence_when_all_endpoints_fail(self, monkeypatch):
        """With every endpoint failing, the refresh loop iterates through
        the entire deduped list. Asserts dedupe + order: configured URL
        first (equal to upstream default, so deduped), then the historical
        platform.claude.com fallback. No duplicate console URL."""
        urls, result = self._run_refresh(
            monkeypatch,
            urlopen_side_effect=[
                ConnectionError("ep1 dead"),
                ConnectionError("ep2 dead"),
            ],
        )
        assert urls == [
            _DEFAULT_OAUTH_TOKEN_URL,
            "https://platform.claude.com/v1/oauth/token",
        ]
        # All endpoints failed → function returns the dict from no path,
        # which means the loop fell through; the implementation raises in
        # that case. Either way ``result`` is an exception here.
        assert isinstance(result, Exception)

    def test_override_puts_custom_url_first_with_historical_fallbacks_intact(self, monkeypatch):
        """With override + every endpoint failing, the full sequence is
        the overridden URL first, then both historical fallbacks in
        their original order, no dedupe needed because override ≠ upstream."""
        urls, result = self._run_refresh(
            monkeypatch,
            env_overrides={"HERMES_ANTHROPIC_OAUTH_TOKEN_URL": "https://example.test/token"},
            urlopen_side_effect=[
                ConnectionError("ep1 dead"),
                ConnectionError("ep2 dead"),
                ConnectionError("ep3 dead"),
            ],
        )
        assert urls == [
            "https://example.test/token",
            "https://platform.claude.com/v1/oauth/token",
            "https://console.anthropic.com/v1/oauth/token",
        ]
        assert isinstance(result, Exception)

    def test_override_used_first_on_happy_path(self, monkeypatch):
        """Sanity check the override path: when the overridden endpoint
        succeeds on its first attempt, that's the only URL called."""
        urls, result = self._run_refresh(
            monkeypatch,
            env_overrides={"HERMES_ANTHROPIC_OAUTH_TOKEN_URL": "https://example.test/token"},
        )
        assert urls == ["https://example.test/token"]
        assert result["access_token"] == "sk-ant-oat-new"

    def test_falls_back_when_first_endpoint_fails(self, monkeypatch):
        """If the configured URL errors, refresh moves on to the next one."""
        successful = _fake_token_response(
            {"access_token": "sk-ant-oat-fb", "refresh_token": "sk-ant-ort-fb", "expires_in": 1200}
        )
        urls, result = self._run_refresh(
            monkeypatch,
            env_overrides={"HERMES_ANTHROPIC_OAUTH_TOKEN_URL": "https://example.test/token"},
            urlopen_side_effect=[
                ConnectionError("first endpoint dead"),
                successful,
            ],
        )
        assert urls[0] == "https://example.test/token"
        assert urls[1] == "https://platform.claude.com/v1/oauth/token"
        assert result["access_token"] == "sk-ant-oat-fb"

    def test_uses_profile_client_id_in_refresh_body(self, monkeypatch):
        """The body's ``client_id`` field follows the profile, not the
        old hardcoded constant."""
        captured_bodies: list = []

        def _capture_request(url, data=None, headers=None, method=None):
            captured_bodies.append((url, data))
            req = MagicMock()
            req.full_url = url
            req.data = data
            return req

        monkeypatch.setenv("HERMES_ANTHROPIC_OAUTH_CLIENT_ID", "custom-cid")

        with patch("urllib.request.Request", side_effect=_capture_request), \
             patch(
                 "urllib.request.urlopen",
                 return_value=_fake_token_response(
                     {"access_token": "x", "refresh_token": "y", "expires_in": 60}
                 ),
             ):
            refresh_anthropic_oauth_pure("some-refresh-token", use_json=True)

        # use_json=True → body is JSON. Decode and check client_id.
        body = json.loads(captured_bodies[0][1].decode())
        assert body["client_id"] == "custom-cid"
        assert body["grant_type"] == "refresh_token"
        assert body["refresh_token"] == "some-refresh-token"

    def test_empty_refresh_token_raises(self):
        with pytest.raises(ValueError, match="refresh_token is required"):
            refresh_anthropic_oauth_pure("", use_json=False)
