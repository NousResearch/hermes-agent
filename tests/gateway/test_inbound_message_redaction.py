"""Test inbound message credential redaction (issue #64351).

When a user pastes credentials into a chat (e.g., a Google OAuth client_secret
paste), the gateway logs the inbound message body at INFO level. Without this
fix, credentials are written to logs in cleartext and later collected by
`hermes debug share`, potentially leaking to a public paste service.

This fix applies `redact_sensitive_text(force=True)` to message preview and
reply_text fields before logging, redacting well-known credential patterns
(sk-, ghp_, AIza, ya29., GOCSPX-, etc.) while preserving non-sensitive context
for debugging.

Credential fixtures are synthetic (benign prefix + run of X's) to avoid false
positives in secret scanners, matching the redactor regexes so tests stay
meaningful without containing real keys.
"""

import logging
import sys
import types
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, SessionSource

# Synthetic, scanner-safe credential fixtures. Each matches redactor regexes.
_FAKE_GHP = "ghp_" + "X" * 36
_FAKE_OPENAI = "sk-proj-" + "X" * 40
_FAKE_JWT = "eyJ" + "X" * 20 + "." + "eyJ" + "X" * 24 + "." + "X" * 30
_FAKE_GOOGLE_SECRET = "GOCSPX-" + "X" * 28
_FAKE_YANDEX = "ya29." + "X" * 80


def _bootstrap(monkeypatch, tmp_path):
    """Minimal GatewayRunner setup."""
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    config = GatewayConfig()
    runner = gateway_run.GatewayRunner(config)
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._handle_active_session_busy_message = AsyncMock(return_value=False)
    runner._session_db = MagicMock()
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._cache_session_source = lambda _key, _source: None
    runner._is_session_run_current = lambda _key, _gen: True
    runner._begin_session_run_generation = lambda _key: 1
    runner._reply_anchor_for_event = lambda _event: None
    runner._get_guild_id = lambda _event: None
    runner._should_send_voice_reply = lambda *_a, **_kw: False
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()

    runner.session_store = MagicMock()
    runner.session_store.load_transcript.return_value = []
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.has_platform_message_id.return_value = False

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"}
    )
    return runner


class TestInboundMessageRedaction:
    """Regression tests for inbound message logging credential redaction."""

    def test_redacts_github_pat_in_message_text(self, monkeypatch, tmp_path, caplog):
        """Message text containing a GitHub PAT must be redacted."""
        import sys

        runner = _bootstrap(monkeypatch, tmp_path)

        event = MessageEvent(
            text=f"help me use this token {_FAKE_GHP}",
            source=SessionSource(
                platform=Platform.TELEGRAM,
                chat_id="123",
                chat_type="private",
                user_id="user123",
            ),
            message_id="msg-1",
        )

        with caplog.at_level(logging.INFO):
            # Call the handler (bypass sentinel guard for test)
            import asyncio

            async def _test():
                return await runner._handle_message_with_agent(event, event.source, "test", 1)

            asyncio.run(_test())

        # Verify credential was redacted from log
        assert any(_FAKE_GHP not in record.message for record in caplog.records)
        # Verify context preserved
        assert any("telegram" in record.message for record in caplog.records)

    def test_redacts_openai_key_in_reply_text(self, monkeypatch, tmp_path, caplog):
        """Reply text containing an OpenAI key must be redacted."""
        import sys

        runner = _bootstrap(monkeypatch, tmp_path)

        event = MessageEvent(
            text="check this",
            source=SessionSource(
                platform=Platform.TELEGRAM,
                chat_id="123",
                chat_type="private",
                user_id="user123",
            ),
            message_id="msg-1",
        )
        event.reply_to_message_id = 456
        event.reply_to_text = f"use this key {_FAKE_OPENAI}"

        with caplog.at_level(logging.INFO):
            import asyncio

            async def _test():
                return await runner._handle_message_with_agent(event, event.source, "test", 1)

            asyncio.run(_test())

        # Verify credential was redacted
        assert any(_FAKE_OPENAI not in record.message for record in caplog.records)
        # Verify reply metadata preserved
        assert any("reply_to_id=456" in record.message for record in caplog.records)

    def test_redacts_google_oauth_client_secret(self, monkeypatch, tmp_path, caplog):
        """Google OAuth client_secret must be redacted (issue #64351 case)."""
        import sys

        runner = _bootstrap(monkeypatch, tmp_path)

        event = MessageEvent(
            text=f'{{"client_secret": "{_FAKE_GOOGLE_SECRET}", "client_id": "test"}}',
            source=SessionSource(
                platform=Platform.TELEGRAM,
                chat_id="123",
                chat_type="private",
                user_id="user123",
            ),
            message_id="msg-1",
        )

        with caplog.at_level(logging.INFO):
            import asyncio

            async def _test():
                return await runner._handle_message_with_agent(event, event.source, "test", 1)

            asyncio.run(_test())

        # Verify client_secret was redacted
        assert any(_FAKE_GOOGLE_SECRET not in record.message for record in caplog.records)

    def test_clean_message_passes_through(self, monkeypatch, tmp_path, caplog):
        """Non-sensitive message text must pass through unchanged."""
        import sys

        runner = _bootstrap(monkeypatch, tmp_path)

        event = MessageEvent(
            text="hello world",
            source=SessionSource(
                platform=Platform.TELEGRAM,
                chat_id="123",
                chat_type="private",
                user_id="user123",
            ),
            message_id="msg-1",
        )

        with caplog.at_level(logging.INFO):
            import asyncio

            async def _test():
                return await runner._handle_message_with_agent(event, event.source, "test", 1)

            asyncio.run(_test())

        # Verify full message preserved
        assert any("hello world" in record.message for record in caplog.records)

    def test_forces_redaction_when_globally_disabled(self, monkeypatch, tmp_path, caplog):
        """force=True must redact even if security.redact_secrets is off."""
        import sys

        runner = _bootstrap(monkeypatch, tmp_path)

        event = MessageEvent(
            text=f"token {_FAKE_GHP}",
            source=SessionSource(
                platform=Platform.TELEGRAM,
                chat_id="123",
                chat_type="private",
                user_id="user123",
            ),
            message_id="msg-1",
        )

        # Disable global redaction
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False, raising=False)

        with caplog.at_level(logging.INFO):
            import asyncio

            async def _test():
                return await runner._handle_message_with_agent(event, event.source, "test", 1)

            asyncio.run(_test())

        # force=True bypasses global setting
        assert any(_FAKE_GHP not in record.message for record in caplog.records)