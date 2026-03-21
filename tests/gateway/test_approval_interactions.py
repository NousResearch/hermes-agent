"""Tests for gateway-native approval inline button interactions."""

import asyncio
import hashlib
import importlib
import time
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import GATEWAY_NO_RESPONSE, MessageEvent, SendResult
from gateway.session import SessionEntry, SessionSource


class ApprovalCaptureAdapter:
    def __init__(self):
        self.sent = []

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append({
            "chat_id": chat_id,
            "content": content,
            "reply_to": reply_to,
            "metadata": metadata,
        })
        return SendResult(success=True, message_id=f"msg-{len(self.sent)}")

    async def send_typing(self, chat_id, metadata=None):
        pass

    async def edit_message(self, chat_id, message_id, content):
        return SendResult(success=True, message_id=message_id)


def _make_source(platform=Platform.TELEGRAM) -> SessionSource:
    return SessionSource(
        platform=platform,
        user_id="u1",
        user_name="Alice",
        chat_id="c1",
        chat_type="dm",
    )


def _make_runner(adapter, platform=Platform.TELEGRAM):
    gateway_run = importlib.import_module("gateway.run")
    GatewayRunner = gateway_run.GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={platform: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {platform: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._smart_model_routing = {}
    runner._session_db = None
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._pending_clarifications = {}
    runner._honcho_managers = {}
    runner._honcho_configs = {}
    runner._show_reasoning = False
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:telegram:dm:c1",
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=platform,
        chat_type="dm",
    )
    return runner


SESSION_KEY = "agent:main:telegram:dm:c1"


class TestBuildApprovalControls:
    def test_produces_correct_structure(self):
        runner = _make_runner(ApprovalCaptureAdapter())
        controls, key_hash = runner._build_approval_controls(SESSION_KEY)

        expected_hash = hashlib.sha256(SESSION_KEY.encode()).hexdigest()[:10]
        assert key_hash == expected_hash

        buttons = controls["buttons"]
        assert len(buttons) == 1  # single row with 3 buttons
        row = buttons[0]
        assert len(row) == 3
        assert row[0]["label"] == "Allow Once"
        assert row[0]["action"] == f"approve-{key_hash}-once"
        assert row[1]["label"] == "Always Allow"
        assert row[1]["action"] == f"approve-{key_hash}-always"
        assert row[2]["label"] == "Deny"
        assert row[2]["action"] == f"approve-{key_hash}-deny"

    def test_hash_is_deterministic(self):
        runner = _make_runner(ApprovalCaptureAdapter())
        _, h1 = runner._build_approval_controls(SESSION_KEY)
        _, h2 = runner._build_approval_controls(SESSION_KEY)
        assert h1 == h2


class TestParseApprovalAction:
    def test_valid_once(self):
        gateway_run = importlib.import_module("gateway.run")
        m = gateway_run.GatewayRunner._APPROVAL_ACTION_RE.fullmatch("approve-abcdef0123-once")
        assert m is not None
        assert m.group(1) == "abcdef0123"
        assert m.group(2) == "once"

    def test_valid_always(self):
        gateway_run = importlib.import_module("gateway.run")
        m = gateway_run.GatewayRunner._APPROVAL_ACTION_RE.fullmatch("approve-abcdef0123-always")
        assert m is not None
        assert m.group(2) == "always"

    def test_valid_deny(self):
        gateway_run = importlib.import_module("gateway.run")
        m = gateway_run.GatewayRunner._APPROVAL_ACTION_RE.fullmatch("approve-abcdef0123-deny")
        assert m is not None
        assert m.group(2) == "deny"

    def test_invalid_action(self):
        gateway_run = importlib.import_module("gateway.run")
        m = gateway_run.GatewayRunner._APPROVAL_ACTION_RE.fullmatch("clarify-abc-choice-0")
        assert m is None

    def test_invalid_hash(self):
        gateway_run = importlib.import_module("gateway.run")
        # Non-hex characters in hash
        m = gateway_run.GatewayRunner._APPROVAL_ACTION_RE.fullmatch("approve-ZZZZZZZZZZ-once")
        assert m is None


@pytest.mark.asyncio
async def test_approval_allow_once():
    """Tapping 'Allow Once' executes the command and pops the approval."""
    runner = _make_runner(ApprovalCaptureAdapter())
    _, key_hash = runner._build_approval_controls(SESSION_KEY)
    runner._pending_approvals[SESSION_KEY] = {
        "command": "rm -rf /tmp/test",
        "pattern_keys": ["exec:rm"],
        "key_hash": key_hash,
        "timestamp": time.time(),
    }

    event = MessageEvent(
        text="",
        source=_make_source(),
        message_id="cb-1",
        metadata={"interaction": {"kind": "button", "action": f"approve-{key_hash}-once"}},
    )

    with patch("tools.approval.approve_session") as mock_session, \
         patch("tools.terminal_tool.terminal_tool", return_value="done") as mock_term:
        result = await runner._handle_pending_approval(event, SESSION_KEY)

    assert "approved and executed" in result
    assert SESSION_KEY not in runner._pending_approvals
    mock_session.assert_called_once_with(SESSION_KEY, "exec:rm")
    mock_term.assert_called_once_with(command="rm -rf /tmp/test", force=True)


@pytest.mark.asyncio
async def test_approval_always_allow():
    """Tapping 'Always Allow' calls approve_permanent."""
    runner = _make_runner(ApprovalCaptureAdapter())
    _, key_hash = runner._build_approval_controls(SESSION_KEY)
    runner._pending_approvals[SESSION_KEY] = {
        "command": "docker restart web",
        "pattern_keys": ["exec:docker"],
        "key_hash": key_hash,
        "timestamp": time.time(),
    }

    event = MessageEvent(
        text="",
        source=_make_source(),
        message_id="cb-2",
        metadata={"interaction": {"kind": "button", "action": f"approve-{key_hash}-always"}},
    )

    with patch("tools.approval.approve_permanent") as mock_perm, \
         patch("tools.terminal_tool.terminal_tool", return_value="ok"):
        result = await runner._handle_pending_approval(event, SESSION_KEY)

    assert "permanently" in result
    mock_perm.assert_called_once_with("exec:docker")


@pytest.mark.asyncio
async def test_approval_deny():
    """Tapping 'Deny' cancels without executing."""
    runner = _make_runner(ApprovalCaptureAdapter())
    _, key_hash = runner._build_approval_controls(SESSION_KEY)
    runner._pending_approvals[SESSION_KEY] = {
        "command": "rm -rf /",
        "pattern_keys": ["exec:rm"],
        "key_hash": key_hash,
        "timestamp": time.time(),
    }

    event = MessageEvent(
        text="",
        source=_make_source(),
        message_id="cb-3",
        metadata={"interaction": {"kind": "button", "action": f"approve-{key_hash}-deny"}},
    )

    result = await runner._handle_pending_approval(event, SESSION_KEY)
    assert "denied" in result.lower()
    assert SESSION_KEY not in runner._pending_approvals


@pytest.mark.asyncio
async def test_stale_approval_returns_error():
    """An expired approval callback returns an error message."""
    runner = _make_runner(ApprovalCaptureAdapter())
    _, key_hash = runner._build_approval_controls(SESSION_KEY)
    runner._pending_approvals[SESSION_KEY] = {
        "command": "rm -rf /tmp/test",
        "pattern_keys": ["exec:rm"],
        "key_hash": key_hash,
        "timestamp": time.time() - 600,  # 10 minutes ago, past the 5-min timeout
    }

    event = MessageEvent(
        text="",
        source=_make_source(),
        message_id="cb-stale",
        metadata={"interaction": {"kind": "button", "action": f"approve-{key_hash}-once"}},
    )

    result = await runner._handle_pending_approval(event, SESSION_KEY)
    assert "expired" in result.lower()
    assert SESSION_KEY not in runner._pending_approvals


@pytest.mark.asyncio
async def test_unknown_approval_callback_returns_inactive():
    """A callback with no matching pending approval returns inactive message."""
    runner = _make_runner(ApprovalCaptureAdapter())
    event = MessageEvent(
        text="",
        source=_make_source(),
        message_id="cb-unknown",
        metadata={"interaction": {"kind": "button", "action": "approve-0000000000-once"}},
    )

    result = await runner._handle_pending_approval(event, SESSION_KEY)
    assert "no longer active" in result.lower()


@pytest.mark.asyncio
async def test_non_approval_callback_returns_none():
    """A non-approval interaction returns None (fall through)."""
    runner = _make_runner(ApprovalCaptureAdapter())
    event = MessageEvent(
        text="hello",
        source=_make_source(),
        message_id="m1",
        metadata={},
    )

    result = await runner._handle_pending_approval(event, SESSION_KEY)
    assert result is None


@pytest.mark.asyncio
async def test_stop_cancels_pending_approval():
    """The /stop command should pop pending approvals."""
    runner = _make_runner(ApprovalCaptureAdapter())
    runner._pending_approvals[SESSION_KEY] = {
        "command": "rm /tmp/test",
        "pattern_keys": ["exec:rm"],
        "key_hash": "abc",
        "timestamp": time.time(),
    }
    runner._running_agents[SESSION_KEY] = MagicMock()

    await runner._handle_stop_command(
        MessageEvent(text="/stop", source=_make_source(), message_id="m-stop")
    )

    assert SESSION_KEY not in runner._pending_approvals


@pytest.mark.asyncio
async def test_cross_session_approval_rejected():
    """A user from a different session cannot approve another user's command."""
    runner = _make_runner(ApprovalCaptureAdapter())
    _, key_hash = runner._build_approval_controls(SESSION_KEY)
    runner._pending_approvals[SESSION_KEY] = {
        "command": "rm -rf /data",
        "pattern_keys": ["exec:rm"],
        "key_hash": key_hash,
        "timestamp": time.time(),
    }

    event = MessageEvent(
        text="",
        source=_make_source(),
        message_id="cb-cross",
        metadata={"interaction": {"kind": "button", "action": f"approve-{key_hash}-once"}},
    )

    # Call with a DIFFERENT session key (simulating User B in a group)
    other_session_key = "agent:main:telegram:group:c1:user-b"
    result = await runner._handle_pending_approval(event, other_session_key)

    assert "no longer active" in result.lower()
    # The original approval must NOT have been consumed
    assert SESSION_KEY in runner._pending_approvals
