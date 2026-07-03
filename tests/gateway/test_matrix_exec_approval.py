import time
import types

import pytest
from unittest.mock import AsyncMock, patch

from gateway.config import PlatformConfig


class TestMatrixExecApprovalReactions:
    @pytest.mark.asyncio
    async def test_send_exec_approval_registers_prompt_and_seeds_reactions(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@liizfq:liizfq.top")
        from plugins.platforms.matrix.adapter import MatrixAdapter

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
        assert adapter._approval_prompts_by_event["$evt1"].requester_user_id is None
        assert adapter._send_reaction.await_count == 3
        emojis = [call.args[2] for call in adapter._send_reaction.await_args_list]
        assert emojis == ["✅", "♾️", "❌"]
        prompt_text = adapter.send.await_args.args[1]
        assert "✅ = approve once" in prompt_text
        assert "♾️ = approve always" in prompt_text
        assert "❌ = deny" in prompt_text
        assert "❎ = deny" not in prompt_text

    @pytest.mark.asyncio
    async def test_send_exec_approval_warns_when_seed_reaction_fails(self, caplog):
        from plugins.platforms.matrix.adapter import MatrixAdapter

        adapter = MatrixAdapter(PlatformConfig(enabled=True, token="tok", extra={"homeserver": "https://matrix.example.org"}))
        adapter._client = types.SimpleNamespace()
        adapter.send = AsyncMock(return_value=types.SimpleNamespace(success=True, message_id="$evt1"))
        adapter._send_reaction = AsyncMock(return_value=None)

        await adapter.send_exec_approval(
            chat_id="!room:example.org",
            command="rm -rf /tmp/test",
            session_key="sess-1",
        )

        assert "failed to add approval reaction" in caplog.text

    @pytest.mark.asyncio
    async def test_send_exec_approval_stores_requester_user(self, monkeypatch):
        from plugins.platforms.matrix.adapter import MatrixAdapter

        adapter = MatrixAdapter(PlatformConfig(enabled=True, token="tok", extra={"homeserver": "https://matrix.example.org"}))
        adapter._client = types.SimpleNamespace()
        adapter.send = AsyncMock(return_value=types.SimpleNamespace(success=True, message_id="$evt1"))
        adapter._send_reaction = AsyncMock(return_value="$r")

        await adapter.send_exec_approval(
            chat_id="!room:example.org",
            command="rm -rf /tmp/test",
            session_key="sess-1",
            metadata={"requester_user_id": "@alice:example.org"},
        )

        assert adapter._approval_prompts_by_event["$evt1"].requester_user_id == "@alice:example.org"

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
    async def test_infinity_reaction_resolves_approval_always(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@liizfq:liizfq.top")
        from plugins.platforms.matrix.adapter import MatrixAdapter, _MatrixApprovalPrompt

        adapter = MatrixAdapter(PlatformConfig(enabled=True, token="tok", extra={"homeserver": "https://matrix.example.org"}))
        adapter._user_id = "@bot:example.org"
        adapter._approval_prompts_by_event["$target"] = _MatrixApprovalPrompt(
            session_key="sess-1", chat_id="!room:example.org", message_id="$target"
        )
        adapter._approval_prompt_by_session["sess-1"] = "$target"
        adapter._redact_bot_approval_reactions = AsyncMock()

        event = types.SimpleNamespace(
            sender="@liizfq:liizfq.top",
            event_id="$react1",
            room_id="!room:example.org",
            content={"m.relates_to": {"event_id": "$target", "key": "♾️"}},
        )

        with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
            await adapter._on_reaction(event)

        mock_resolve.assert_called_once_with("sess-1", "always")

    @pytest.mark.asyncio
    async def test_reaction_from_non_requester_gets_feedback(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@alice:example.org,@bob:example.org")
        from plugins.platforms.matrix.adapter import MatrixAdapter, _MatrixApprovalPrompt

        adapter = MatrixAdapter(PlatformConfig(enabled=True, token="tok", extra={"homeserver": "https://matrix.example.org"}))
        adapter._user_id = "@bot:example.org"
        adapter.send = AsyncMock(return_value=types.SimpleNamespace(success=True, message_id="$feedback"))
        adapter._approval_prompts_by_event["$target"] = _MatrixApprovalPrompt(
            session_key="sess-1",
            chat_id="!room:example.org",
            message_id="$target",
            requester_user_id="@alice:example.org",
        )

        event = types.SimpleNamespace(
            sender="@bob:example.org",
            event_id="$react1",
            room_id="!room:example.org",
            content={"m.relates_to": {"event_id": "$target", "key": "✅"}},
        )

        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            await adapter._on_reaction(event)

        mock_resolve.assert_not_called()
        adapter.send.assert_awaited_once()
        assert "Only the user" in adapter.send.await_args.args[1]

    @pytest.mark.asyncio
    async def test_expired_approval_reaction_does_not_resolve(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@alice:example.org")
        from plugins.platforms.matrix.adapter import MatrixAdapter, _MatrixApprovalPrompt

        adapter = MatrixAdapter(PlatformConfig(enabled=True, token="tok", extra={"homeserver": "https://matrix.example.org"}))
        adapter._user_id = "@bot:example.org"
        adapter.send = AsyncMock(return_value=types.SimpleNamespace(success=True, message_id="$feedback"))
        adapter._redact_bot_approval_reactions = AsyncMock()
        adapter._approval_prompts_by_event["$target"] = _MatrixApprovalPrompt(
            session_key="sess-1",
            chat_id="!room:example.org",
            message_id="$target",
            expires_at=time.monotonic() - 1,
        )
        adapter._approval_prompt_by_session["sess-1"] = "$target"

        event = types.SimpleNamespace(
            sender="@alice:example.org",
            event_id="$react1",
            room_id="!room:example.org",
            content={"m.relates_to": {"event_id": "$target", "key": "✅"}},
        )

        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            await adapter._on_reaction(event)

        mock_resolve.assert_not_called()
        assert "$target" not in adapter._approval_prompts_by_event
        assert "sess-1" not in adapter._approval_prompt_by_session

    @pytest.mark.asyncio
    async def test_model_picker_reaction_selects_model(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@alice:example.org")
        from plugins.platforms.matrix.adapter import MatrixAdapter

        adapter = MatrixAdapter(PlatformConfig(enabled=True, token="tok", extra={"homeserver": "https://matrix.example.org"}))
        adapter._client = types.SimpleNamespace()
        adapter._user_id = "@bot:example.org"
        adapter.send = AsyncMock(return_value=types.SimpleNamespace(success=True, message_id="$picker"))
        adapter._send_reaction = AsyncMock(side_effect=["$r1", "$r2"])

        selected = {}

        async def on_selected(room_id, model_id, provider_slug):
            selected.update(room_id=room_id, model_id=model_id, provider_slug=provider_slug)
            return "switched"

        await adapter.send_model_picker(
            chat_id="!room:example.org",
            providers=[{"slug": "openai", "name": "OpenAI", "models": ["gpt-5.4", "gpt-5.3"]}],
            current_model="gpt-5.3",
            current_provider="openai",
            session_key="sess-1",
            on_model_selected=on_selected,
            metadata={"requester_user_id": "@alice:example.org"},
        )
        adapter._redact_bot_model_picker_reactions = AsyncMock()
        adapter.send = AsyncMock(return_value=types.SimpleNamespace(success=True, message_id="$confirm"))

        event = types.SimpleNamespace(
            sender="@alice:example.org",
            event_id="$react1",
            room_id="!room:example.org",
            content={"m.relates_to": {"event_id": "$picker", "key": "1️⃣"}},
        )

        await adapter._on_reaction(event)

        assert selected == {
            "room_id": "!room:example.org",
            "model_id": "gpt-5.4",
            "provider_slug": "openai",
        }
        assert "$picker" not in adapter._model_picker_prompts_by_event
        adapter.send.assert_awaited_once_with("!room:example.org", "switched", reply_to="$picker")
