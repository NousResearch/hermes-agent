import time
import types

import pytest
from unittest.mock import AsyncMock, patch

from gateway.config import PlatformConfig


class TestMatrixExecApprovalReactions:


    @pytest.mark.asyncio
    async def test_reaction_resolves_pending_approval(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@liizfq:liizfq.top")
        from plugins.platforms.matrix.adapter import MatrixAdapter, _MatrixApprovalPrompt

        adapter = MatrixAdapter(PlatformConfig(enabled=True, token="tok", extra={"homeserver": "https://matrix.example.org"}))
        # Resolve user_id so _is_self_sender doesn't defensively drop all traffic (#15763).
        adapter._user_id = "@bot:example.org"
        adapter._approval_prompts_by_event["$target"] = _MatrixApprovalPrompt(
            session_key="sess-1", chat_id="!room:example.org", message_id="$target"
        )
        adapter._approval_prompt_by_session["sess-1"] = "$target"

        content = {"m.relates_to": {"event_id": "$target", "key": "✅"}}
        event = types.SimpleNamespace(
            sender="@liizfq:liizfq.top",
            event_id="$react1",
            room_id="!room:example.org",
            content=content,
        )

        with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
            await adapter._on_reaction(event)

        mock_resolve.assert_called_once_with("sess-1", "once")
        assert "$target" not in adapter._approval_prompts_by_event
        assert "sess-1" not in adapter._approval_prompt_by_session

    @pytest.mark.asyncio
    async def test_late_reaction_still_resolves_still_pending_approval(self, monkeypatch):
        """Swallowed-answer regression: the chat-side reaction window expiring BEFORE the blocked
        agent thread (Matrix 300s vs approvals.timeout 9000s) must NOT eat the user's answer.
        The adapter-side expiry is advisory; tools.approval's queue (still pending?) decides
        whether the choice lands — same contract as Telegram's late button tap."""
        monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@liizfq:liizfq.top")
        from plugins.platforms.matrix.adapter import MatrixAdapter, _MatrixApprovalPrompt

        adapter = MatrixAdapter(PlatformConfig(enabled=True, token="tok", extra={"homeserver": "https://matrix.example.org"}))
        adapter._user_id = "@bot:example.org"
        prompt = _MatrixApprovalPrompt(
            session_key="sess-1", chat_id="!room:example.org", message_id="$target",
            expires_at=time.monotonic() - 1,  # adapter-side window already closed
        )
        adapter._approval_prompts_by_event["$target"] = prompt
        adapter._approval_prompt_by_session["sess-1"] = "$target"
        adapter.send = AsyncMock(return_value=None)

        content = {"m.relates_to": {"event_id": "$target", "key": "✅"}}
        event = types.SimpleNamespace(
            sender="@liizfq:liizfq.top", event_id="$react1",
            room_id="!room:example.org", content=content,
        )

        with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
            await adapter._on_reaction(event)

        mock_resolve.assert_called_once_with("sess-1", "once")
        assert "$target" not in adapter._approval_prompts_by_event
        assert "sess-1" not in adapter._approval_prompt_by_session
        # The user must NOT be told "expired ... run the command again" when the answer landed.
        sent = " ".join(str(call) for call in adapter.send.call_args_list).lower()
        assert "expired" not in sent
        assert "honored" in sent

    @pytest.mark.asyncio
    async def test_late_reaction_with_nothing_pending_reports_expired(self, monkeypatch):
        """Resolver returns 0 (agent wait already timed out fail-closed, or resolved elsewhere):
        a late tap must report ⌛ expired, never claim approval — Telegram parity."""
        monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@liizfq:liizfq.top")
        from plugins.platforms.matrix.adapter import MatrixAdapter, _MatrixApprovalPrompt

        adapter = MatrixAdapter(PlatformConfig(enabled=True, token="tok", extra={"homeserver": "https://matrix.example.org"}))
        adapter._user_id = "@bot:example.org"
        prompt = _MatrixApprovalPrompt(
            session_key="sess-1", chat_id="!room:example.org", message_id="$target",
            expires_at=time.monotonic() - 1,
        )
        adapter._approval_prompts_by_event["$target"] = prompt
        adapter._approval_prompt_by_session["sess-1"] = "$target"
        adapter.send = AsyncMock(return_value=None)

        content = {"m.relates_to": {"event_id": "$target", "key": "✅"}}
        event = types.SimpleNamespace(
            sender="@liizfq:liizfq.top", event_id="$react1",
            room_id="!room:example.org", content=content,
        )

        with patch("tools.approval.resolve_gateway_approval", return_value=0) as mock_resolve:
            await adapter._on_reaction(event)

        mock_resolve.assert_called_once_with("sess-1", "once")
        assert prompt.resolved is True
        assert "$target" not in adapter._approval_prompts_by_event
        assert "sess-1" not in adapter._approval_prompt_by_session
        sent = " ".join(str(call) for call in adapter.send.call_args_list)
        assert "no command was waiting" in sent

    def test_approval_window_inherits_agent_timeout(self, monkeypatch):
        """The chat window and the blocked-thread deadline must not silently diverge:
        with no explicit override the approval window follows config approvals.timeout;
        pickers keep the explicit (or legacy 300s) window; the env var overrides both."""
        monkeypatch.delenv("MATRIX_APPROVAL_TIMEOUT_SECONDS", raising=False)
        monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@op:example.org")
        import tools.approval_context as ac
        monkeypatch.setattr(ac, "_get_approval_timeout", lambda: 9000)
        from plugins.platforms.matrix.adapter import MatrixAdapter

        adapter = MatrixAdapter(PlatformConfig(enabled=True, token="tok", extra={"homeserver": "https://matrix.example.org"}))
        assert adapter._approval_timeout_seconds == 9000
        assert adapter._picker_timeout_seconds == 300

        monkeypatch.setenv("MATRIX_APPROVAL_TIMEOUT_SECONDS", "600")
        adapter2 = MatrixAdapter(PlatformConfig(enabled=True, token="tok", extra={"homeserver": "https://matrix.example.org"}))
        assert adapter2._approval_timeout_seconds == 600
        assert adapter2._picker_timeout_seconds == 600
