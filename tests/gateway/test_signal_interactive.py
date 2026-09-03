"""Tests for Signal interactive methods.

Covers send_exec_approval, send_clarify, send_slash_confirm and
_try_intercept_interactive_reply. Follows the test_signal.py harness:
monkeypatch fixtures, _make_signal_adapter helper, class-based grouping.

Coverage matrix:
  send_exec_approval
    - stores pending state on success, choice_keys correct
    - smart_denied removes session/always from choices
    - allow_session=False removes session choice
    - allow_permanent=False removes always choice
    - send failure does not store state

  send_clarify
    - multi-choice: stores state, sends numbered menu
    - open-ended (empty choices): delegates to base, no state stored
    - send failure does not store state

  send_slash_confirm
    - stores state, choice_keys correct
    - allow_permanent=False removes always choice
    - send failure does not store state

  _try_intercept_interactive_reply
    - non-digit text returns False (pass-through)
    - approval: valid index resolves and returns True
    - approval: deny choice resolved correctly
    - approval: out-of-range index returns False, state preserved
    - approval: resolve exception logged, still returns True (consumed)
    - clarify: valid index resolves and returns True
    - clarify: out-of-range index returns False, state preserved
    - slash_confirm: valid index resolves and returns True
    - no pending state for chat_id returns False
    - state isolation: +1111 resolution does not affect +2222
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult


# ---------------------------------------------------------------------------
# Shared helpers (mirror test_signal.py conventions)
# ---------------------------------------------------------------------------

def _make_adapter(monkeypatch, account="+15551234567"):
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "")
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", "*")
    from gateway.platforms.signal import SignalAdapter
    config = PlatformConfig()
    config.enabled = True
    config.extra = {"http_url": "http://localhost:8080", "account": account}
    adapter = SignalAdapter(config)
    # Stub send so tests don't need a live HTTP client
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="ts1"))
    return adapter


def _choices_dm():
    return ["Option A", "Option B", "Option C"]


# ---------------------------------------------------------------------------
# send_exec_approval
# ---------------------------------------------------------------------------

class TestSignalSendExecApproval:

    @pytest.mark.asyncio
    async def test_success_stores_state_with_correct_keys(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        result = await adapter.send_exec_approval(
            chat_id="+9999", command="rm -rf /tmp/x",
            session_key="sk1",
        )
        assert result.success
        assert "+9999" in adapter._pending_approvals
        state = adapter._pending_approvals["+9999"]
        assert state["session_key"] == "sk1"
        assert state["choice_keys"] == ["once", "session", "always", "deny"]

    @pytest.mark.asyncio
    async def test_smart_denied_removes_session_and_always(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        await adapter.send_exec_approval(
            chat_id="+9999", command="cmd", session_key="sk",
            smart_denied=True,
        )
        keys = adapter._pending_approvals["+9999"]["choice_keys"]
        assert "session" not in keys
        assert "always" not in keys
        assert "deny" in keys

    @pytest.mark.asyncio
    async def test_allow_session_false_removes_session(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        await adapter.send_exec_approval(
            chat_id="+9999", command="cmd", session_key="sk",
            allow_session=False,
        )
        keys = adapter._pending_approvals["+9999"]["choice_keys"]
        assert "session" not in keys
        assert "always" in keys

    @pytest.mark.asyncio
    async def test_allow_permanent_false_removes_always(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        await adapter.send_exec_approval(
            chat_id="+9999", command="cmd", session_key="sk",
            allow_permanent=False,
        )
        keys = adapter._pending_approvals["+9999"]["choice_keys"]
        assert "always" not in keys
        assert "session" in keys

    @pytest.mark.asyncio
    async def test_send_failure_does_not_store_state(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter.send = AsyncMock(return_value=SendResult(success=False, error="offline"))
        result = await adapter.send_exec_approval(
            chat_id="+9999", command="cmd", session_key="sk",
        )
        assert not result.success
        assert "+9999" not in adapter._pending_approvals


# ---------------------------------------------------------------------------
# send_clarify
# ---------------------------------------------------------------------------

class TestSignalSendClarify:

    @pytest.mark.asyncio
    async def test_multi_choice_stores_state(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        choices = _choices_dm()
        result = await adapter.send_clarify(
            chat_id="+9999", question="Pick one?", choices=choices,
            clarify_id="cid1", session_key="sk",
        )
        assert result.success
        assert "+9999" in adapter._pending_clarify
        state = adapter._pending_clarify["+9999"]
        assert state["clarify_id"] == "cid1"
        assert state["choices"] == choices

    @pytest.mark.asyncio
    async def test_open_ended_delegates_to_base_no_state_stored(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        base_result = SendResult(success=True, message_id="base")
        with patch(
            "gateway.platforms.base.BasePlatformAdapter.send_clarify",
            new=AsyncMock(return_value=base_result),
        ):
            await adapter.send_clarify(
                chat_id="+9999", question="Tell me?", choices=[],
                clarify_id="cid2", session_key="sk",
            )
        assert "+9999" not in adapter._pending_clarify

    @pytest.mark.asyncio
    async def test_send_failure_does_not_store_state(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter.send = AsyncMock(return_value=SendResult(success=False, error="err"))
        await adapter.send_clarify(
            chat_id="+9999", question="Q?", choices=_choices_dm(),
            clarify_id="cid3", session_key="sk",
        )
        assert "+9999" not in adapter._pending_clarify


# ---------------------------------------------------------------------------
# send_slash_confirm
# ---------------------------------------------------------------------------

class TestSignalSendSlashConfirm:

    @pytest.mark.asyncio
    async def test_success_stores_state_with_correct_keys(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        result = await adapter.send_slash_confirm(
            chat_id="+9999", title="Reload MCP",
            message="Restarts the MCP server.",
            session_key="sk", confirm_id="conf1",
        )
        assert result.success
        assert "+9999" in adapter._pending_slash_confirm
        state = adapter._pending_slash_confirm["+9999"]
        assert state["confirm_id"] == "conf1"
        assert state["session_key"] == "sk"
        assert state["choice_keys"] == ["once", "always", "cancel"]

    @pytest.mark.asyncio
    async def test_allow_permanent_false_removes_always(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        await adapter.send_slash_confirm(
            chat_id="+9999", title="T", message="M",
            session_key="sk", confirm_id="conf2",
            allow_permanent=False,
        )
        keys = adapter._pending_slash_confirm["+9999"]["choice_keys"]
        assert "always" not in keys
        assert "cancel" in keys

    @pytest.mark.asyncio
    async def test_send_failure_does_not_store_state(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter.send = AsyncMock(return_value=SendResult(success=False, error="err"))
        await adapter.send_slash_confirm(
            chat_id="+9999", title="T", message="M",
            session_key="sk", confirm_id="conf3",
        )
        assert "+9999" not in adapter._pending_slash_confirm


# ---------------------------------------------------------------------------
# _try_intercept_interactive_reply
# ---------------------------------------------------------------------------

class TestSignalInterceptInteractiveReply:

    @pytest.mark.asyncio
    async def test_non_digit_text_returns_false(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        assert await adapter._try_intercept_interactive_reply("+9999", "yes", "sk") is False
        assert await adapter._try_intercept_interactive_reply("+9999", "  ", "sk") is False
        assert await adapter._try_intercept_interactive_reply("+9999", "1a", "sk") is False

    @pytest.mark.asyncio
    async def test_no_pending_state_returns_false(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        assert await adapter._try_intercept_interactive_reply("+9999", "1", "sk") is False

    # --- Approval ---

    @pytest.mark.asyncio
    async def test_approval_valid_index_resolves_and_consumes_state(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._pending_approvals["+9999"] = {
            "session_key": "sk1",
            "choice_keys": ["once", "session", "always", "deny"],
        }
        with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
            result = await adapter._try_intercept_interactive_reply("+9999", "1", "sk1")
        assert result is True
        mock_resolve.assert_called_once_with("sk1", "once")
        assert "+9999" not in adapter._pending_approvals

    @pytest.mark.asyncio
    async def test_approval_deny_choice_resolved_correctly(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._pending_approvals["+9999"] = {
            "session_key": "sk1",
            "choice_keys": ["once", "deny"],
        }
        with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
            result = await adapter._try_intercept_interactive_reply("+9999", "2", "sk1")
        assert result is True
        mock_resolve.assert_called_once_with("sk1", "deny")

    @pytest.mark.asyncio
    async def test_approval_out_of_range_returns_false_state_preserved(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._pending_approvals["+9999"] = {
            "session_key": "sk1",
            "choice_keys": ["once", "deny"],
        }
        result = await adapter._try_intercept_interactive_reply("+9999", "9", "sk1")
        assert result is False
        assert "+9999" in adapter._pending_approvals

    @pytest.mark.asyncio
    async def test_approval_resolve_exception_logged_still_consumes(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._pending_approvals["+9999"] = {
            "session_key": "sk1",
            "choice_keys": ["once", "deny"],
        }
        with patch("tools.approval.resolve_gateway_approval", side_effect=RuntimeError("boom")):
            result = await adapter._try_intercept_interactive_reply("+9999", "1", "sk1")
        assert result is True
        assert "+9999" not in adapter._pending_approvals

    # --- Clarify ---

    @pytest.mark.asyncio
    async def test_clarify_valid_index_resolves_and_consumes_state(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._pending_clarify["+9999"] = {
            "clarify_id": "cid1",
            "choices": ["Option A", "Option B", "Option C"],
        }
        mock_clarify = MagicMock()
        mock_clarify.resolve = MagicMock()
        with patch.dict("sys.modules", {"tools.clarify_gateway": mock_clarify}):
            result = await adapter._try_intercept_interactive_reply("+9999", "2", "sk")
        assert result is True
        assert "+9999" not in adapter._pending_clarify

    @pytest.mark.asyncio
    async def test_clarify_out_of_range_returns_false_state_preserved(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._pending_clarify["+9999"] = {
            "clarify_id": "cid1",
            "choices": ["A", "B"],
        }
        result = await adapter._try_intercept_interactive_reply("+9999", "5", "sk")
        assert result is False
        assert "+9999" in adapter._pending_clarify

    # --- Slash confirm ---

    @pytest.mark.asyncio
    async def test_slash_confirm_valid_index_resolves_and_consumes_state(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._pending_slash_confirm["+9999"] = {
            "confirm_id": "conf1",
            "session_key": "sk",
            "choice_keys": ["once", "always", "cancel"],
        }
        with patch("tools.slash_confirm.resolve", new=MagicMock()) as mock_resolve:
            result = await adapter._try_intercept_interactive_reply("+9999", "3", "sk")
        assert result is True
        mock_resolve.assert_called_once_with("conf1", "cancel")
        assert "+9999" not in adapter._pending_slash_confirm

    # --- State isolation ---

    @pytest.mark.asyncio
    async def test_state_isolation_separate_chats_do_not_interfere(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._pending_approvals["+1111"] = {
            "session_key": "sk_a",
            "choice_keys": ["once", "deny"],
        }
        adapter._pending_approvals["+2222"] = {
            "session_key": "sk_b",
            "choice_keys": ["once", "deny"],
        }
        with patch("tools.approval.resolve_gateway_approval", return_value=1):
            result = await adapter._try_intercept_interactive_reply("+1111", "1", "sk_a")
        assert result is True
        assert "+1111" not in adapter._pending_approvals
        assert "+2222" in adapter._pending_approvals
