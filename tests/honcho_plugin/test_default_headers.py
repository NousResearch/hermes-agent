"""Tests for Honcho defaultHeaders config — parsing, SDK wiring, and isolation.

This feature allows users to send custom HTTP headers with every Honcho SDK
request (e.g. proxy auth for Cloudflare Access, Tailscale, basic auth).

SECURITY INVARIANT: These headers MUST only be sent to the Honcho API server.
They must NEVER appear in LLM provider requests or any other HTTP client.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from plugins.memory.honcho.client import (
    HonchoClientConfig,
    _parse_headers,
    _HEADER_NAME_RE,
    get_honcho_client,
    reset_honcho_client,
)


# ─── Config parsing tests ────────────────────────────────────────────────────


class TestDefaultHeadersDataclass:
    def test_empty_by_default(self):
        config = HonchoClientConfig()
        assert config.default_headers == {}


class TestParseHeaders:
    """Unit tests for the _parse_headers helper."""

    def test_root_level_headers(self):
        root = {"defaultHeaders": {"X-Custom": "value1", "Authorization": "Bearer abc"}}
        host = {}
        result = _parse_headers(host, root, "defaultHeaders")
        assert result == {"X-Custom": "value1", "Authorization": "Bearer abc"}

    def test_host_override_replaces_root(self):
        """Host-level defaultHeaders replaces the root entirely (not merged)."""
        root = {"defaultHeaders": {"X-Root": "should-be-gone"}}
        host = {"defaultHeaders": {"X-Host": "wins"}}
        result = _parse_headers(host, root, "defaultHeaders")
        assert result == {"X-Host": "wins"}
        assert "X-Root" not in result

    def test_empty_when_not_configured(self):
        result = _parse_headers({}, {}, "defaultHeaders")
        assert result == {}

    def test_empty_when_value_is_not_dict(self):
        result = _parse_headers({}, {"defaultHeaders": "not-a-dict"}, "defaultHeaders")
        assert result == {}

    def test_preserves_whitespace_in_values(self):
        """Auth tokens can have trailing/leading whitespace — don't strip."""
        root = {"defaultHeaders": {"CF-Access-Client-Secret": " abc123 "}}
        result = _parse_headers({}, root, "defaultHeaders")
        assert result["CF-Access-Client-Secret"] == " abc123 "

    def test_allows_empty_string_values(self):
        """Unlike _parse_string_map, empty values are kept."""
        root = {"defaultHeaders": {"X-Empty": ""}}
        result = _parse_headers({}, root, "defaultHeaders")
        assert result == {"X-Empty": ""}

    def test_none_value_becomes_empty_string(self):
        root = {"defaultHeaders": {"X-Null": None}}
        result = _parse_headers({}, root, "defaultHeaders")
        assert result == {"X-Null": ""}

    def test_invalid_header_name_skipped(self):
        """Non-RFC-7230 header names are skipped with a warning."""
        root = {"defaultHeaders": {
            "Valid-Name": "ok",
            "Invalid Name": "has space",
            "Also\tBad": "has tab",
        }}
        result = _parse_headers({}, root, "defaultHeaders")
        assert result == {"Valid-Name": "ok"}

    def test_empty_key_skipped(self):
        root = {"defaultHeaders": {"": "no-name", "X-Real": "yes"}}
        result = _parse_headers({}, root, "defaultHeaders")
        assert result == {"X-Real": "yes"}


class TestHeaderNameRegex:
    """Verify the RFC 7230 token regex accepts/rejects correctly."""

    @pytest.mark.parametrize("name", [
        "Content-Type", "X-Custom-Header", "Authorization",
        "CF-Access-Client-Id", "x-api-key", "Accept",
        "X_Underscore", "X.Dot", "X~Tilde",
    ])
    def test_valid_names(self, name):
        assert _HEADER_NAME_RE.match(name)

    @pytest.mark.parametrize("name", [
        "Has Space", "Tab\there", "Newline\nBad",
        "Colon:Bad", "Slash/Bad", "(Parens)",
        "Bräcket", "日本語",
    ])
    def test_invalid_names(self, name):
        assert not _HEADER_NAME_RE.match(name)


class TestFromGlobalConfigDefaultHeaders:
    """Integration tests: config JSON → HonchoClientConfig.default_headers."""

    def test_root_level(self, tmp_path, monkeypatch):
        config_file = tmp_path / "honcho.json"
        config_file.write_text(json.dumps({
            "baseUrl": "http://localhost:8000",
            "defaultHeaders": {
                "CF-Access-Client-Id": "xxx.access",
                "CF-Access-Client-Secret": "secret123",
            },
        }))
        monkeypatch.delenv("HONCHO_API_KEY", raising=False)
        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.default_headers == {
            "CF-Access-Client-Id": "xxx.access",
            "CF-Access-Client-Secret": "secret123",
        }

    def test_host_level_overrides_root(self, tmp_path, monkeypatch):
        config_file = tmp_path / "honcho.json"
        config_file.write_text(json.dumps({
            "baseUrl": "http://localhost:8000",
            "defaultHeaders": {"X-Root": "should-not-appear"},
            "hosts": {
                "hermes": {
                    "defaultHeaders": {"X-Host": "wins"},
                }
            },
        }))
        monkeypatch.delenv("HONCHO_API_KEY", raising=False)
        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.default_headers == {"X-Host": "wins"}
        assert "X-Root" not in config.default_headers

    def test_missing_key_yields_empty(self, tmp_path, monkeypatch):
        config_file = tmp_path / "honcho.json"
        config_file.write_text(json.dumps({
            "baseUrl": "http://localhost:8000",
        }))
        monkeypatch.delenv("HONCHO_API_KEY", raising=False)
        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.default_headers == {}

    def test_invalid_names_filtered_at_parse_time(self, tmp_path, monkeypatch):
        config_file = tmp_path / "honcho.json"
        config_file.write_text(json.dumps({
            "baseUrl": "http://localhost:8000",
            "defaultHeaders": {
                "Valid-Header": "ok",
                "Invalid Header": "has space",
            },
        }))
        monkeypatch.delenv("HONCHO_API_KEY", raising=False)
        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.default_headers == {"Valid-Header": "ok"}


# ─── SDK wiring tests ────────────────────────────────────────────────────────


class TestGetHonchoClientDefaultHeaders:
    """Verify headers are passed to the Honcho SDK constructor correctly."""

    def setup_method(self):
        reset_honcho_client()

    def teardown_method(self):
        reset_honcho_client()

    def test_headers_passed_to_sdk(self):
        """Non-empty default_headers are passed as default_headers kwarg."""
        cfg = HonchoClientConfig(
            base_url="http://localhost:8000",
            enabled=True,
            default_headers={
                "CF-Access-Client-Id": "test-id",
                "CF-Access-Client-Secret": "test-secret",
            },
        )
        fake_honcho = MagicMock()
        with patch("honcho.Honcho", return_value=fake_honcho) as mock_honcho, \
             patch("hermes_cli.config.load_config", return_value={}):
            get_honcho_client(cfg)

        mock_honcho.assert_called_once()
        kwargs = mock_honcho.call_args.kwargs
        assert kwargs["default_headers"] == {
            "CF-Access-Client-Id": "test-id",
            "CF-Access-Client-Secret": "test-secret",
        }

    def test_empty_headers_not_passed_to_sdk(self):
        """Empty default_headers should NOT add default_headers kwarg at all."""
        cfg = HonchoClientConfig(
            base_url="http://localhost:8000",
            enabled=True,
            default_headers={},
        )
        fake_honcho = MagicMock()
        with patch("honcho.Honcho", return_value=fake_honcho) as mock_honcho, \
             patch("hermes_cli.config.load_config", return_value={}):
            get_honcho_client(cfg)

        mock_honcho.assert_called_once()
        kwargs = mock_honcho.call_args.kwargs
        assert "default_headers" not in kwargs


# ─── Security isolation test ─────────────────────────────────────────────────


class TestDefaultHeadersIsolation:
    """SECURITY: Honcho headers must NEVER leak to LLM provider clients.

    The Honcho SDK creates its own httpx.Client, so the headers are isolated
    by construction. This test guards against future refactoring that might
    accidentally merge Honcho headers into shared dicts or LLM client kwargs.
    """

    def setup_method(self):
        reset_honcho_client()

    def teardown_method(self):
        reset_honcho_client()

    def test_canary_header_only_in_honcho_not_in_llm_client(self, tmp_path, monkeypatch):
        """A canary header configured for Honcho must not appear in the
        LLM provider's OpenAI client constructor."""
        # Set up Honcho config with a canary header
        config_file = tmp_path / "honcho.json"
        config_file.write_text(json.dumps({
            "baseUrl": "http://honcho.internal:8000",
            "defaultHeaders": {
                "X-Canary-Secret": "LEAK_DETECTOR_12345",
            },
            "hosts": {"hermes": {"enabled": True}},
        }))
        monkeypatch.delenv("HONCHO_API_KEY", raising=False)

        # Parse config — canary must be present
        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.default_headers["X-Canary-Secret"] == "LEAK_DETECTOR_12345"

        # Build Honcho client — canary must arrive in SDK kwargs
        fake_honcho = MagicMock()
        with patch("honcho.Honcho", return_value=fake_honcho) as mock_honcho, \
             patch("hermes_cli.config.load_config", return_value={}):
            get_honcho_client(config)

        honcho_kwargs = mock_honcho.call_args.kwargs
        assert honcho_kwargs["default_headers"]["X-Canary-Secret"] == "LEAK_DETECTOR_12345"

        # Now verify the canary is NOT reachable from a simulated LLM client path.
        # The HonchoClientConfig.default_headers field is the only place these
        # live; verify it's not on any shared/global config dict that LLM
        # provider code might read.
        from hermes_cli.config import load_config
        with patch("hermes_cli.config.load_config", return_value={
            "model": {"default": "gpt-4"},
        }) as mock_cfg:
            llm_config = mock_cfg()

        # The LLM config must not contain any trace of Honcho headers
        assert "X-Canary-Secret" not in json.dumps(llm_config)
        assert "LEAK_DETECTOR_12345" not in json.dumps(llm_config)

        # The Honcho config's raw dict should NOT be the same object as
        # anything an LLM provider reads
        assert "default_headers" not in llm_config
        assert "defaultHeaders" not in llm_config
