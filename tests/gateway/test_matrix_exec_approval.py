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

        mock_resolve.assert_called_once_with("sess-1", "once", request_id=None)
        assert "$target" not in adapter._approval_prompts_by_event
        assert "sess-1" not in adapter._approval_prompt_by_session


class TestMatrixApprovalRequestIdForwarding:
    """#124974: a reaction on a card carrying ``approval_request_id`` must
    resolve ITS queued entry (real queue, real resolve), not the FIFO-oldest
    one; absent id → FIFO. Also pins the requester-only gate end to end
    (reviewer finding 2: the gate reads ``prompt.requester_user_id``, which
    the send path populates from the forwarded ``requester_user_id`` key)."""

    @staticmethod
    def _adapter():
        monkeypatch = __import__("pytest").MonkeyPatch()
        monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@liizfq:liizfq.top")
        from plugins.platforms.matrix.adapter import MatrixAdapter

        adapter = MatrixAdapter(PlatformConfig(enabled=True, token="tok", extra={"homeserver": "https://matrix.example.org"}))
        adapter._user_id = "@bot:example.org"
        return adapter, monkeypatch

    @staticmethod
    def _react(adapter, prompt, sender="@liizfq:liizfq.top"):
        import types as _types
        content = {"m.relates_to": {"event_id": prompt.message_id, "key": "✅"}}
        event = _types.SimpleNamespace(
            sender=sender, event_id="$react-rid", room_id="!room:example.org", content=content,
        )
        return adapter._on_reaction(event)

    @pytest.mark.asyncio
    async def test_reaction_with_request_id_resolves_its_own_card_entry(self):
        import pytest as _pytest
        from tests.gateway._approval_queue_helpers import (
            assert_resolved, assert_still_pending, clear_approvals, enqueue_approvals)
        from plugins.platforms.matrix.adapter import _MatrixApprovalPrompt

        adapter, monkeypatch = self._adapter()
        session_key = "sess-rid-1"
        old, new = enqueue_approvals(session_key, {"command": "old"}, {"command": "new"})
        rid = new.data["request_id"]
        prompt = _MatrixApprovalPrompt(
            session_key=session_key, chat_id="!room:example.org", message_id="$target-rid",
            requester_user_id="@liizfq:liizfq.top", request_id=rid,
        )
        adapter._approval_prompts_by_event["$target-rid"] = prompt
        adapter._approval_prompt_by_session[session_key] = "$target-rid"
        try:
            await self._react(adapter, prompt)
        finally:
            clear_approvals(session_key)
            monkeypatch.undo()
        assert_resolved(new, "once")
        assert_still_pending(old)
        assert prompt.resolved is True

    @pytest.mark.asyncio
    async def test_reaction_without_request_id_keeps_fifo(self):
        from tests.gateway._approval_queue_helpers import (
            assert_resolved, assert_still_pending, clear_approvals, enqueue_approvals)
        from plugins.platforms.matrix.adapter import _MatrixApprovalPrompt

        adapter, monkeypatch = self._adapter()
        session_key = "sess-fifo-1"
        old, new = enqueue_approvals(session_key, {"command": "old"}, {"command": "new"})
        prompt = _MatrixApprovalPrompt(
            session_key=session_key, chat_id="!room:example.org", message_id="$target-fifo",
            requester_user_id="@liizfq:liizfq.top",
        )
        adapter._approval_prompts_by_event["$target-fifo"] = prompt
        adapter._approval_prompt_by_session[session_key] = "$target-fifo"
        try:
            await self._react(adapter, prompt)
        finally:
            clear_approvals(session_key)
            monkeypatch.undo()
        assert_resolved(old, "once")
        assert_still_pending(new)

    @pytest.mark.asyncio
    async def test_send_prompt_carries_forwarded_ids_onto_the_prompt_object(self):
        """The send path must stamp request_id AND requester_user_id onto the
        stored prompt — the requester-only gate (:2550) reads the ATTRIBUTE,
        which is populated from the metadata dict key forwarded by #124974."""
        from unittest.mock import AsyncMock, MagicMock
        from gateway.platforms.base import ExecApprovalPrompt
        from gateway.platforms.base import SendResult as _SR

        adapter, monkeypatch = self._adapter()
        adapter._client = MagicMock()
        adapter.send = AsyncMock(return_value=_SR(success=True, message_id="$sent-rid"))
        adapter._send_reaction = AsyncMock(return_value="$react-evt-rid")
        prompt = ExecApprovalPrompt(
            chat_id="!room:example.org", session_key="sess-rid-2", text="run it", actions=[("✅", "once", "primary")],
            command="rm -rf /x", description="dangerous", smart_denied=False,
            metadata={"approval_request_id": "d" * 32, "requester_user_id": "@liizfq:liizfq.top"},
        )
        try:
            result = await adapter._send_exec_approval_prompt(prompt)
            assert result.success is True
            stored = adapter._approval_prompts_by_event["$sent-rid"]
            assert stored.request_id == "d" * 32
            assert stored.requester_user_id == "@liizfq:liizfq.top"
        finally:
            monkeypatch.undo()


class TestMatrixRequesterGateForwardedId:
    """Reviewer finding 2 verification: the reaction gate at :2550 reads the
    prompt ATTRIBUTE ``requester_user_id``. The stored object is the adapter's
    own ``_MatrixApprovalPrompt`` (which HAS the field) and the send path
    stamps it from the forwarded metadata dict key — so the gate is NOT inert
    once the metadata carries the id. A non-requester reaction is rejected."""

    @pytest.mark.asyncio
    async def test_non_requester_reaction_is_rejected(self):
        import pytest as _pytest
        from tests.gateway._approval_queue_helpers import clear_approvals, enqueue_approvals
        from plugins.platforms.matrix.adapter import _MatrixApprovalPrompt

        monkeypatch = _pytest.MonkeyPatch()
        monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@liizfq:liizfq.top,@alice:example.org")
        from plugins.platforms.matrix.adapter import MatrixAdapter
        from gateway.config import PlatformConfig

        adapter = MatrixAdapter(PlatformConfig(enabled=True, token="tok", extra={"homeserver": "https://matrix.example.org"}))
        adapter._user_id = "@bot:example.org"
        session_key = "sess-req-1"
        entries = enqueue_approvals(session_key, {"command": "old"})
        prompt = _MatrixApprovalPrompt(
            session_key=session_key, chat_id="!room:example.org", message_id="$target-req",
            requester_user_id="@liizfq:liizfq.top",
        )
        adapter._approval_prompts_by_event["$target-req"] = prompt
        adapter._approval_prompt_by_session[session_key] = "$target-req"

        import types as _types
        content = {"m.relates_to": {"event_id": "$target-req", "key": "✅"}}
        event = _types.SimpleNamespace(
            sender="@alice:example.org", event_id="$react-alice",
            room_id="!room:example.org", content=content,
        )
        try:
            await adapter._on_reaction(event)
        finally:
            clear_approvals(session_key)
            monkeypatch.undo()

        # The gate fired: the non-requester reaction did NOT resolve the entry.
        assert prompt.resolved is False
        assert not entries[0].event.is_set(), "non-requester reaction must not resolve the approval"
