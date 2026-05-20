"""Tests for the Sendblue onboarding helpers in hermes_cli.setup."""
from __future__ import annotations

import io
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import setup as setup_mod


class TestValidateE164List:
    def test_accepts_valid_numbers(self):
        result = setup_mod._validate_e164_list("+15551234567,+447911123456")
        assert result == ["+15551234567", "+447911123456"]

    def test_strips_whitespace(self):
        result = setup_mod._validate_e164_list("  +15551234567 , +447911123456  ")
        assert result == ["+15551234567", "+447911123456"]

    def test_drops_invalid_and_keeps_valid(self, capsys):
        result = setup_mod._validate_e164_list("+15551234567,bad,+0invalid,+15559876543")
        assert result == ["+15551234567", "+15559876543"]
        # Each invalid entry surfaces a warning so the user knows what was dropped.
        captured = capsys.readouterr()
        assert "bad" in captured.out
        assert "+0invalid" in captured.out

    def test_empty_and_blank_entries_skipped(self):
        assert setup_mod._validate_e164_list("") == []
        assert setup_mod._validate_e164_list(",,,") == []
        assert setup_mod._validate_e164_list(" , , ") == []

    def test_rejects_missing_plus_prefix(self):
        # E.164 requires the leading "+"; raw national format must be rejected.
        assert setup_mod._validate_e164_list("15551234567") == []

    def test_rejects_leading_zero_country_code(self):
        # E.164 country codes start with [1-9].
        assert setup_mod._validate_e164_list("+05551234567") == []


class TestValidateSendblueCredentials:
    def _patch_urlopen(self, returner):
        return patch.object(setup_mod, "__name__", setup_mod.__name__), patch(
            "urllib.request.urlopen", side_effect=returner
        )

    def test_success(self):
        resp = MagicMock()
        resp.__enter__ = lambda self: self
        resp.__exit__ = lambda *a: False
        with patch("urllib.request.urlopen", return_value=resp):
            ok, msg = setup_mod._validate_sendblue_credentials("kid", "secret")
        assert ok is True
        assert msg == ""

    def test_401_returns_friendly_message(self):
        err = urllib.error.HTTPError(
            url="https://api.sendblue.com/accounts/me",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=io.BytesIO(b""),
        )
        with patch("urllib.request.urlopen", side_effect=err):
            ok, msg = setup_mod._validate_sendblue_credentials("kid", "wrong")
        assert ok is False
        assert "401" in msg
        assert "rejected" in msg.lower()

    def test_other_http_error_surfaces_code(self):
        err = urllib.error.HTTPError(
            url="https://api.sendblue.com/accounts/me",
            code=503,
            msg="Service Unavailable",
            hdrs=None,
            fp=io.BytesIO(b""),
        )
        with patch("urllib.request.urlopen", side_effect=err):
            ok, msg = setup_mod._validate_sendblue_credentials("kid", "secret")
        assert ok is False
        assert "503" in msg

    def test_network_error(self):
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("DNS fail")):
            ok, msg = setup_mod._validate_sendblue_credentials("kid", "secret")
        assert ok is False
        assert "Could not reach Sendblue" in msg

    def test_os_error_surfaces(self):
        with patch("urllib.request.urlopen", side_effect=OSError("ECONNREFUSED")):
            ok, msg = setup_mod._validate_sendblue_credentials("kid", "secret")
        assert ok is False
        assert "Could not reach Sendblue" in msg

    def test_sends_required_headers_and_hits_right_endpoint(self):
        """The User-Agent header is mandatory — Sendblue's Cloudflare layer
        returns HTTP 403 (error 1010) to the default urllib UA. Endpoint
        must be the documented webhook-list path (verified live against
        api.sendblue.com 2026-05-20)."""
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["headers"] = dict(req.headers)
            captured["url"] = req.full_url
            resp = MagicMock()
            resp.__enter__ = lambda self: self
            resp.__exit__ = lambda *a: False
            return resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            setup_mod._validate_sendblue_credentials("kid-123", "secret-xyz")

        assert captured["url"] == "https://api.sendblue.com/api/account/webhooks"
        headers = {k.lower(): v for k, v in captured["headers"].items()}
        assert headers["sb-api-key-id"] == "kid-123"
        assert headers["sb-api-secret-key"] == "secret-xyz"
        assert headers["user-agent"] == "hermes-agent setup"


class TestPrintSendblueReverseProxySnippets:
    def test_extracts_hostname_from_url(self, capsys):
        setup_mod._print_sendblue_reverse_proxy_snippets(
            "https://example.com/sendblue-gateway/receive", 8665
        )
        out = capsys.readouterr().out
        assert "example.com {" in out
        assert "127.0.0.1:8665" in out

    def test_uses_custom_port(self, capsys):
        setup_mod._print_sendblue_reverse_proxy_snippets(
            "https://hermes.example.org/sendblue-gateway/receive", 9000
        )
        out = capsys.readouterr().out
        assert "127.0.0.1:9000" in out
        assert "hermes.example.org" in out

    def test_falls_back_when_hostname_missing(self, capsys):
        # A path-only string has no hostname — fall back rather than crash.
        setup_mod._print_sendblue_reverse_proxy_snippets("/sendblue-gateway/receive", 8665)
        out = capsys.readouterr().out
        assert "yourdomain.com" in out

    def test_includes_nginx_block(self, capsys):
        setup_mod._print_sendblue_reverse_proxy_snippets(
            "https://example.com/sendblue-gateway/receive", 8665
        )
        out = capsys.readouterr().out
        assert "nginx" in out
        assert "proxy_pass http://127.0.0.1:8665/sendblue-gateway/" in out


class TestPrintSendblueTunnelGuidance:
    def test_lists_all_four_options(self, capsys):
        setup_mod._print_sendblue_tunnel_guidance()
        out = capsys.readouterr().out
        assert "Cloudflare Tunnel" in out
        assert "Tailscale Funnel" in out
        assert "DuckDNS" in out
        assert "ngrok" in out

    def test_mentions_le_ip_limitation(self, capsys):
        setup_mod._print_sendblue_tunnel_guidance()
        out = capsys.readouterr().out
        assert "Let's Encrypt" in out

    def test_references_webhook_path(self, capsys):
        setup_mod._print_sendblue_tunnel_guidance()
        out = capsys.readouterr().out
        assert "/sendblue-gateway/receive" in out


class TestSendblueDispatch:
    def test_builtin_setup_fn_resolves_sendblue(self):
        from hermes_cli import gateway as gateway_mod

        fn = gateway_mod._builtin_setup_fn("sendblue")
        assert fn is setup_mod._setup_sendblue

    def test_sendblue_in_platforms_registry(self):
        from hermes_cli.platforms import PLATFORMS

        assert "sendblue" in PLATFORMS
        assert PLATFORMS["sendblue"].default_toolset == "hermes-sendblue"

    def test_sendblue_in_platforms_list(self):
        from hermes_cli.gateway import _PLATFORMS

        entries = [p for p in _PLATFORMS if p["key"] == "sendblue"]
        assert len(entries) == 1
        assert entries[0]["token_var"] == "SENDBLUE_API_KEY_ID"
        # Bespoke flow — must not have a declarative vars schema.
        assert "vars" not in entries[0]

    def test_sendblue_toolset_registered(self):
        from toolsets import TOOLSETS

        assert "hermes-sendblue" in TOOLSETS
        # Composite mirrors hermes-bluebubbles shape.
        assert TOOLSETS["hermes-sendblue"]["tools"] == TOOLSETS["hermes-bluebubbles"]["tools"]
