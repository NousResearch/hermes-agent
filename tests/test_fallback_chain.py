"""Tests for the provider fallback chain module."""

import os
import sys
from io import StringIO
from unittest.mock import patch, MagicMock

import pytest

from agent.fallback_chain import (
    FallbackChain,
    FallbackEntry,
    select_auto,
    select_interactive,
    should_trigger_fallback,
    FALLBACK_STATUS_CODES,
)


# =============================================================================
# FallbackEntry
# =============================================================================

class TestFallbackEntry:
    def test_display_name_plain(self):
        e = FallbackEntry(provider="openrouter")
        assert e.display_name == "openrouter"

    def test_display_name_local(self):
        e = FallbackEntry(provider="lmstudio", base_url="http://localhost:1234/v1")
        assert "localhost:1234" in e.display_name

    def test_display_name_remote(self):
        """Remote URLs should NOT show host in display name."""
        e = FallbackEntry(provider="openrouter", base_url="https://openrouter.ai/api/v1")
        assert e.display_name == "openrouter"

    def test_resolve_api_key_explicit_env(self):
        e = FallbackEntry(provider="custom", api_key_env="MY_CUSTOM_KEY")
        with patch.dict("os.environ", {"MY_CUSTOM_KEY": "secret123"}):
            assert e.resolve_api_key() == "secret123"

    def test_resolve_api_key_openrouter(self):
        e = FallbackEntry(provider="openrouter")
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-key"}):
            assert e.resolve_api_key() == "sk-or-key"

    def test_resolve_api_key_local(self):
        """Local providers should work even without a key."""
        e = FallbackEntry(provider="lmstudio")
        with patch.dict("os.environ", {}, clear=True):
            key = e.resolve_api_key()
            assert key == "not-needed"

    def test_build_client_kwargs_openrouter(self):
        e = FallbackEntry(provider="openrouter")
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-key"}):
            kwargs = e.build_client_kwargs()
            assert "openrouter" in kwargs["base_url"]
            assert kwargs["api_key"] == "sk-or-key"
            assert "default_headers" in kwargs

    def test_build_client_kwargs_local(self):
        e = FallbackEntry(provider="lmstudio", base_url="http://localhost:1234/v1")
        kwargs = e.build_client_kwargs()
        assert kwargs["base_url"] == "http://localhost:1234/v1"
        assert "default_headers" not in kwargs


# =============================================================================
# FallbackChain
# =============================================================================

class TestFallbackChain:
    def test_from_config_empty(self):
        chain = FallbackChain.from_config({})
        assert chain.has_fallbacks() is False
        # Empty chain is technically exhausted (nothing to try)
        assert chain.is_exhausted() is True

    def test_from_config_with_entries(self):
        config = {
            "fallback": {
                "mode": "interactive",
                "timeout": 15,
                "chain": [
                    {"provider": "openrouter", "model": "anthropic/claude-opus-4.6"},
                    {"provider": "lmstudio", "base_url": "http://localhost:1234/v1", "model": "qwen3-30b"},
                ],
            }
        }
        chain = FallbackChain.from_config(config)
        assert chain.has_fallbacks() is True
        assert len(chain.entries) == 2
        assert chain.mode == "interactive"
        assert chain.timeout == 15
        assert chain.entries[0].provider == "openrouter"
        assert chain.entries[0].model == "anthropic/claude-opus-4.6"
        assert chain.entries[1].provider == "lmstudio"
        assert chain.entries[1].base_url == "http://localhost:1234/v1"

    def test_from_config_skips_invalid_entries(self):
        config = {
            "fallback": {
                "chain": [
                    {"provider": "openrouter"},
                    "not-a-dict",
                    {"provider": ""},  # empty provider
                    {"provider": "lmstudio"},
                ],
            }
        }
        chain = FallbackChain.from_config(config)
        assert len(chain.entries) == 2
        assert chain.entries[0].provider == "openrouter"
        assert chain.entries[1].provider == "lmstudio"

    def test_build_legacy_chain_with_key(self):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-key"}):
            chain = FallbackChain.build_legacy_chain("claude-opus-4-20250514")
            assert chain.has_fallbacks() is True
            assert len(chain.entries) == 1
            assert chain.entries[0].provider == "openrouter"
            assert chain.entries[0].model == "anthropic/claude-opus-4-20250514"
            assert chain.mode == "auto"  # Legacy mode = auto for backward compat

    def test_build_legacy_chain_without_key(self):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": ""}):
            chain = FallbackChain.build_legacy_chain("some-model")
            assert chain.has_fallbacks() is False

    def test_next_available(self):
        chain = FallbackChain(entries=[
            FallbackEntry(provider="a"),
            FallbackEntry(provider="b"),
            FallbackEntry(provider="c"),
        ])
        assert chain.next_available().provider == "a"
        chain.mark_failed(chain.entries[0])
        assert chain.next_available().provider == "b"
        chain.mark_failed(chain.entries[1])
        assert chain.next_available().provider == "c"
        chain.mark_failed(chain.entries[2])
        assert chain.next_available() is None
        assert chain.is_exhausted() is True

    def test_available_entries(self):
        chain = FallbackChain(entries=[
            FallbackEntry(provider="a"),
            FallbackEntry(provider="b"),
            FallbackEntry(provider="c"),
        ])
        chain.mark_failed(chain.entries[1])
        available = chain.available_entries()
        assert len(available) == 2
        assert available[0][1].provider == "a"
        assert available[1][1].provider == "c"

    def test_select_by_index(self):
        chain = FallbackChain(entries=[
            FallbackEntry(provider="a"),
            FallbackEntry(provider="b"),
        ])
        assert chain.select_by_index(0).provider == "a"
        assert chain.select_by_index(1).provider == "b"
        assert chain.select_by_index(2) is None
        assert chain.select_by_index(-1) is None

    def test_reset(self):
        chain = FallbackChain(entries=[
            FallbackEntry(provider="a"),
            FallbackEntry(provider="b"),
        ])
        chain.mark_failed(chain.entries[0])
        chain.mark_failed(chain.entries[1])
        assert chain.is_exhausted() is True
        chain.reset()
        assert chain.is_exhausted() is False
        assert chain.next_available().provider == "a"


# =============================================================================
# select_auto
# =============================================================================

class TestSelectAuto:
    def test_returns_first_available(self):
        chain = FallbackChain(entries=[
            FallbackEntry(provider="openrouter", model="claude-opus"),
            FallbackEntry(provider="lmstudio", model="qwen3"),
        ])
        entry = select_auto(chain, "rate limited")
        assert entry.provider == "openrouter"

    def test_skips_exhausted(self):
        chain = FallbackChain(entries=[
            FallbackEntry(provider="openrouter"),
            FallbackEntry(provider="lmstudio"),
        ])
        chain.mark_failed(chain.entries[0])
        entry = select_auto(chain, "rate limited")
        assert entry.provider == "lmstudio"

    def test_returns_none_when_exhausted(self):
        chain = FallbackChain(entries=[
            FallbackEntry(provider="openrouter"),
        ])
        chain.mark_failed(chain.entries[0])
        entry = select_auto(chain, "rate limited")
        assert entry is None

    def test_calls_notify_on_switch(self):
        chain = FallbackChain(entries=[
            FallbackEntry(provider="openrouter", model="claude"),
        ])
        notify = MagicMock()
        entry = select_auto(chain, "error", notify=notify)
        assert entry is not None
        notify.assert_called_once()
        call_msg = notify.call_args[0][0]
        assert "openrouter" in call_msg.lower()

    def test_calls_notify_on_exhaust(self):
        chain = FallbackChain(entries=[
            FallbackEntry(provider="openrouter"),
        ])
        chain.mark_failed(chain.entries[0])
        notify = MagicMock()
        select_auto(chain, "error", notify=notify)
        notify.assert_called_once()
        assert "exhausted" in notify.call_args[0][0].lower()


# =============================================================================
# should_trigger_fallback
# =============================================================================

class TestShouldTriggerFallback:
    def test_rate_limit(self):
        err = Exception("rate limit exceeded")
        assert should_trigger_fallback(429, err) is True

    def test_overload(self):
        err = Exception("server overloaded")
        assert should_trigger_fallback(529, err) is True

    def test_service_unavailable(self):
        err = Exception("service unavailable")
        assert should_trigger_fallback(503, err) is True

    def test_auth_error(self):
        err = Exception("unauthorized")
        assert should_trigger_fallback(401, err) is True

    def test_bad_request_not_triggered(self):
        err = Exception("invalid request")
        assert should_trigger_fallback(400, err) is False

    def test_connection_error(self):
        class ConnectError(Exception):
            pass
        err = ConnectError("connection refused")
        assert should_trigger_fallback(None, err) is True

    def test_overloaded_in_message(self):
        err = Exception("The server is overloaded right now")
        assert should_trigger_fallback(None, err) is True

    def test_generic_error_not_triggered(self):
        err = Exception("something weird happened")
        assert should_trigger_fallback(None, err) is False


# =============================================================================
# select_interactive (limited testing — mocking stdin)
# =============================================================================

class TestSelectInteractive:
    def test_returns_none_when_exhausted(self):
        chain = FallbackChain(entries=[
            FallbackEntry(provider="openrouter"),
        ])
        chain.mark_failed(chain.entries[0])
        result = select_interactive(chain, "error")
        assert result is None

    def test_auto_selects_on_non_tty(self):
        """When stdin is not a tty, should auto-select after timeout."""
        chain = FallbackChain(entries=[
            FallbackEntry(provider="openrouter", model="claude"),
        ], timeout=1)  # 1 second timeout for fast test

        # Mock stdin to not be a tty
        with patch.object(sys, "stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            entry = select_interactive(chain, "error")
            assert entry is not None
            assert entry.provider == "openrouter"

    def test_user_selects_option(self):
        """Simulate user typing '2' to select second option."""
        chain = FallbackChain(entries=[
            FallbackEntry(provider="openrouter", model="claude"),
            FallbackEntry(provider="lmstudio", model="qwen3"),
        ], timeout=30)

        # Mock stdin as tty with select returning ready + user input "2"
        mock_stdin = StringIO("2\n")
        mock_stdin.isatty = lambda: True

        with (
            patch.object(sys, "stdin", mock_stdin),
            patch("agent.fallback_chain.select.select", return_value=([mock_stdin], [], [])),
        ):
            entry = select_interactive(chain, "error")
            assert entry is not None
            assert entry.provider == "lmstudio"

    def test_user_skips(self):
        """Simulate user typing 's' to skip."""
        chain = FallbackChain(entries=[
            FallbackEntry(provider="openrouter"),
        ], timeout=30)

        mock_stdin = StringIO("s\n")
        mock_stdin.isatty = lambda: True

        with (
            patch.object(sys, "stdin", mock_stdin),
            patch("agent.fallback_chain.select.select", return_value=([mock_stdin], [], [])),
        ):
            entry = select_interactive(chain, "error")
            assert entry is None


# =============================================================================
# Integration: config -> chain -> entry -> client kwargs
# =============================================================================

class TestIntegration:
    def test_config_to_chain_to_kwargs(self):
        """Full round-trip: config dict -> chain -> entry -> OpenAI client kwargs."""
        config = {
            "fallback": {
                "mode": "auto",
                "timeout": 10,
                "chain": [
                    {"provider": "openrouter", "model": "anthropic/claude-opus-4.6"},
                    {
                        "provider": "lmstudio",
                        "base_url": "http://localhost:1234/v1",
                        "model": "qwen3-30b-a3b",
                    },
                ],
            }
        }
        chain = FallbackChain.from_config(config)

        # First entry: openrouter
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-key"}):
            entry1 = chain.next_available()
            kwargs1 = entry1.build_client_kwargs()
            assert "openrouter" in kwargs1["base_url"]
            assert kwargs1["api_key"] == "sk-or-key"
            assert "default_headers" in kwargs1

        # Mark first as failed, get second
        chain.mark_failed(entry1)
        entry2 = chain.next_available()
        kwargs2 = entry2.build_client_kwargs()
        assert kwargs2["base_url"] == "http://localhost:1234/v1"
        assert entry2.model == "qwen3-30b-a3b"
        assert "default_headers" not in kwargs2  # Not openrouter

        # Chain exhausted
        chain.mark_failed(entry2)
        assert chain.is_exhausted() is True
        assert chain.next_available() is None

    def test_multi_provider_cascade_auto(self):
        """Auto mode cascades through entries without interaction."""
        chain = FallbackChain(entries=[
            FallbackEntry(provider="openrouter", model="claude"),
            FallbackEntry(provider="lmstudio", model="qwen3"),
            FallbackEntry(provider="ollama", model="llama3"),
        ], mode="auto")

        # First call -> openrouter
        entry = select_auto(chain, "429 rate limited")
        assert entry.provider == "openrouter"
        chain.mark_failed(entry)

        # Second call -> lmstudio
        entry = select_auto(chain, "openrouter also failed")
        assert entry.provider == "lmstudio"
        chain.mark_failed(entry)

        # Third call -> ollama
        entry = select_auto(chain, "lmstudio down")
        assert entry.provider == "ollama"
        chain.mark_failed(entry)

        # Exhausted
        entry = select_auto(chain, "everything failed")
        assert entry is None
        assert chain.is_exhausted() is True
