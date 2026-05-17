import types

import pytest
from unittest.mock import AsyncMock, patch

from gateway.config import PlatformConfig


class TestMatrixExecApprovalReactions:
    @pytest.mark.asyncio
    async def test_send_exec_approval_registers_prompt_and_seeds_reactions(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@liizfq:liizfq.top")
        from gateway.platforms.matrix import MatrixAdapter

        adapter = MatrixAdapter(PlatformConfig(enabled=True, token="tok", extra={"homeserver": "https://matrix.example.org"}))
        adapter._client = types.SimpleNamespace()
        adapter.send = AsyncMock(return_value=types.SimpleNamespace(success=True, message_id="$evt1"))
        adapter._send_reaction = AsyncMock(return_value="$r")

        result = await adapter.send_exec_approval(
            chat_id="!room:example.org",
            command="rm -rf /tmp/test",
            session_key="sess-1",
            description="dangerous",
        )

        assert result.success is True
        assert adapter._approval_prompt_by_session["sess-1"] == "$evt1"
        assert adapter._approval_prompts_by_event["$evt1"].session_key == "sess-1"
        assert adapter._send_reaction.await_count == 2
        emojis = [call.args[2] for call in adapter._send_reaction.await_args_list]
        assert emojis == ["✅", "❎"]

    @pytest.mark.asyncio
    async def test_reaction_resolves_pending_approval(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@liizfq:liizfq.top")
        from gateway.platforms.matrix import MatrixAdapter, _MatrixApprovalPrompt

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
    async def test_reaction_ignores_user_not_authorized_by_gateway_allowlist(self, monkeypatch):
        monkeypatch.delenv("MATRIX_ALLOWED_USERS", raising=False)
        monkeypatch.delenv("MATRIX_ALLOW_ALL_USERS", raising=False)
        monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
        monkeypatch.setenv("GATEWAY_ALLOWED_USERS", "@authorized:example.org")
        from gateway.platforms.matrix import MatrixAdapter, _MatrixApprovalPrompt

        adapter = MatrixAdapter(PlatformConfig(enabled=True, token="tok", extra={"homeserver": "https://matrix.example.org"}))
        adapter._user_id = "@bot:example.org"
        adapter._approval_prompts_by_event["$target"] = _MatrixApprovalPrompt(
            session_key="sess-1", chat_id="!room:example.org", message_id="$target"
        )
        adapter._approval_prompt_by_session["sess-1"] = "$target"

        unauthorized_event = types.SimpleNamespace(
            sender="@intruder:example.org",
            event_id="$react-intruder",
            room_id="!room:example.org",
            content={"m.relates_to": {"event_id": "$target", "key": "✅"}},
        )
        authorized_event = types.SimpleNamespace(
            sender="@authorized:example.org",
            event_id="$react-authorized",
            room_id="!room:example.org",
            content={"m.relates_to": {"event_id": "$target", "key": "✅"}},
        )

        with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
            await adapter._on_reaction(unauthorized_event)
            mock_resolve.assert_not_called()
            assert "$target" in adapter._approval_prompts_by_event

            await adapter._on_reaction(authorized_event)

        mock_resolve.assert_called_once_with("sess-1", "once")
        assert "$target" not in adapter._approval_prompts_by_event
        assert "sess-1" not in adapter._approval_prompt_by_session

    @pytest.mark.asyncio
    async def test_reaction_uses_gateway_authorization_callback(self, monkeypatch):
        monkeypatch.delenv("MATRIX_ALLOWED_USERS", raising=False)
        monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)
        monkeypatch.delenv("MATRIX_ALLOW_ALL_USERS", raising=False)
        monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
        from gateway.platforms.matrix import MatrixAdapter, _MatrixApprovalPrompt

        class Runner:
            def _is_user_authorized(self, source):
                return source.user_id == "@paired:example.org"

            async def handle(self, event):
                return None

        adapter = MatrixAdapter(PlatformConfig(enabled=True, token="tok", extra={"homeserver": "https://matrix.example.org"}))
        adapter._user_id = "@bot:example.org"
        adapter.set_message_handler(Runner().handle)
        adapter._is_dm_room = AsyncMock(return_value=False)
        adapter._approval_prompts_by_event["$target"] = _MatrixApprovalPrompt(
            session_key="sess-1", chat_id="!room:example.org", message_id="$target"
        )
        adapter._approval_prompt_by_session["sess-1"] = "$target"
        event = types.SimpleNamespace(
            sender="@paired:example.org",
            event_id="$react-paired",
            room_id="!room:example.org",
            content={"m.relates_to": {"event_id": "$target", "key": "✅"}},
        )

        with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
            await adapter._on_reaction(event)

        adapter._is_dm_room.assert_awaited_once_with("!room:example.org")
        mock_resolve.assert_called_once_with("sess-1", "once")
