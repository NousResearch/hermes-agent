"""Tests for #109904 — inbound message log preview leaks credentials pasted as
assignment lines.

The preview for ``inbound message: …`` is truncated to 80 chars and folded to a
single line. ``redact_sensitive_text``'s YAML/ENV passes are line-anchored
(``^…password: value`` with ``re.MULTILINE``), so folding first moves the key
mid-line and the value reaches ``agent.log`` / ``gateway.log`` verbatim.
Redaction must run BEFORE truncation/folding.
"""

import logging
import sys
import types
from unittest.mock import AsyncMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource


def _bootstrap(monkeypatch):
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    runner = gateway_run.GatewayRunner(GatewayConfig())
    # The log line under test fires before session resolution; bail out there.
    runner._hmwa_resolve_session = AsyncMock(return_value=None)
    return runner


def _event(text, reply_to_text=None):
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1001",
            chat_type="group",
            user_id="12345",
        ),
        message_id="msg-42",
        reply_to_message_id="msg-41",
        reply_to_text=reply_to_text,
    )


@pytest.mark.asyncio
async def test_multiline_assignment_password_masked_in_inbound_log(monkeypatch, caplog):
    runner = _bootstrap(monkeypatch)
    body = "For Renpho:\nUser: someone@example.com\nPassword: Xy9zAbc123\n"

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        await runner._handle_message_with_agent(
            _event(body, reply_to_text="Password: SecretReply99"), _event(body).source,
            "agent:main:telegram:group:-1001:12345", 1,
        )

    inbound = [r for r in caplog.records if "inbound message" in r.getMessage()]
    assert inbound, "expected the inbound-message log line to fire"
    line = inbound[0].getMessage()

    assert "Xy9zAbc123" not in line
    assert "SecretReply99" not in line
    assert "Password: ***" in line
    # Folding is preserved: the preview stays a single line.
    assert "\n" not in line


@pytest.mark.asyncio
async def test_prefix_token_still_masked_in_inbound_log(monkeypatch, caplog):
    """The prefix-shaped pass (ghp_…) must keep working after the reorder."""
    runner = _bootstrap(monkeypatch)
    body = "For Renpho: ghp_abcdef1234567890abcdef1234567890abcdef12"

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        await runner._handle_message_with_agent(
            _event(body), _event(body).source,
            "agent:main:telegram:group:-1001:12345", 1,
        )

    inbound = [r for r in caplog.records if "inbound message" in r.getMessage()]
    line = inbound[0].getMessage()
    assert "ghp_abcdef1234567890abcdef1234567890abcdef12" not in line


@pytest.mark.asyncio
async def test_benign_multiline_message_logs_folded_preview(monkeypatch, caplog):
    runner = _bootstrap(monkeypatch)
    body = "line one\nline two\nline three"

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        await runner._handle_message_with_agent(
            _event(body), _event(body).source,
            "agent:main:telegram:group:-1001:12345", 1,
        )

    inbound = [r for r in caplog.records if "inbound message" in r.getMessage()]
    line = inbound[0].getMessage()
    assert "line one line two line three" in line
