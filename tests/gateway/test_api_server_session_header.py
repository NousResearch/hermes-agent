"""X-Hermes-Session-Id continuation on /v1/chat/completions: history provenance from the
session store and the refusals that must answer before the stream starts (#143686, #144227)."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_api_server import _create_app, _make_adapter

_DUMMY_KEY = "sk-" + "secret"  # the placeholder credential test_api_server's auth_adapter uses


@pytest.fixture
def auth_adapter():
    return _make_adapter(api_key=_DUMMY_KEY)


class TestSessionIdHeader:

    @pytest.mark.asyncio
    async def test_provided_session_id_loads_history_from_db(self, auth_adapter):
        """When X-Hermes-Session-Id is provided, history comes from SessionDB not request body."""
        mock_result = {"final_response": "OK", "messages": [], "api_calls": 1}
        db_history = [
            {"role": "user", "content": "stored message 1"},
            {"role": "assistant", "content": "stored reply 1"},
        ]
        mock_db = MagicMock()
        mock_db.get_session.return_value = {"id": "existing-session", "source": "api_server"}
        mock_db.get_messages_as_conversation.return_value = db_history
        mock_db.resolve_resume_session_id.side_effect = lambda sid: sid
        auth_adapter._session_db = mock_db
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

                resp = await cli.post(
                    "/v1/chat/completions",
                    headers={"X-Hermes-Session-Id": "existing-session", "Authorization": "Bearer sk-secret"},
                    # Request body has different history — should be ignored
                    json={
                        "model": "hermes-agent",
                        "messages": [
                            {"role": "user", "content": "old msg from client"},
                            {"role": "assistant", "content": "old reply from client"},
                            {"role": "user", "content": "new question"},
                        ],
                    },
                )

            assert resp.status == 200
            call_kwargs = mock_run.call_args.kwargs
            # History must come from DB, not from the request body
            assert call_kwargs["conversation_history"] == db_history
            assert call_kwargs["user_message"] == "new question"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
    async def test_foreign_surface_session_refused_before_stream(self, auth_adapter, stream):
        """#143686/#144227: continuing a CLI-created (local-) session through the API is a
        deliberate admission refusal, but it must answer 403 before the stream starts —
        as HTTP 200 + a mid-stream error chunk it reads as an empty successful completion
        to OpenAI-compatible clients that only watch status / [DONE]."""
        mock_db = MagicMock()
        mock_db.get_session.return_value = {"id": "local-deadbeef", "source": "tool"}
        mock_db.resolve_resume_session_id.side_effect = lambda sid: sid
        auth_adapter._session_db = mock_db
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                resp = await cli.post(
                    "/v1/chat/completions",
                    headers={"X-Hermes-Session-Id": "local-deadbeef", "Authorization": "Bearer sk-secret"},
                    json={
                        "model": "hermes-agent",
                        "stream": stream,
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
            assert resp.status == 403, f"stream={stream}: expected 403 before any body, got {resp.status}"
            body = await resp.json()
            assert "another surface" in body["error"]["message"]
            assert "tool" in body["error"]["message"]
            # The turn never starts, and the refused target's history is never read.
            mock_run.assert_not_called()
            mock_db.get_messages_as_conversation.assert_not_called()

    @pytest.mark.asyncio
    async def test_bot_room_session_continues(self, auth_adapter):
        """bot_room targets share the api principal's admissible sources: the pre-stream
        gate must not refuse them (#143686)."""
        mock_result = {"final_response": "OK", "messages": [], "api_calls": 1}
        mock_db = MagicMock()
        mock_db.get_session.return_value = {"id": "room-42", "source": "bot_room"}
        mock_db.get_messages_as_conversation.return_value = []
        mock_db.resolve_resume_session_id.side_effect = lambda sid: sid
        auth_adapter._session_db = mock_db
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/chat/completions",
                    headers={"X-Hermes-Session-Id": "room-42", "Authorization": "Bearer sk-secret"},
                    json={"model": "hermes-agent", "messages": [{"role": "user", "content": "hi"}]},
                )
            assert resp.status == 200
            mock_run.assert_called_once()
