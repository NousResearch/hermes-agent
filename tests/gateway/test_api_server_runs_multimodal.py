"""POST /v1/runs keeps multimodal (image) content in the input and in the history it builds."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_api_server_runs import _create_runs_app, _make_adapter


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.mark.asyncio
async def test_start_preserves_multimodal_input_and_history(adapter):
    app = _create_runs_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_create_agent") as mock_create:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "done"}
            mock_agent.session_prompt_tokens = 0
            mock_agent.session_completion_tokens = 0
            mock_agent.session_total_tokens = 0
            mock_create.return_value = mock_agent

            resp = await cli.post(
                "/v1/runs",
                json={
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "first turn"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "https://example.com/first.png"},
                                },
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "describe this"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "https://example.com/final.png"},
                                },
                            ],
                        },
                    ]
                },
            )
            assert resp.status == 202
            data = await resp.json()

            for _ in range(20):
                status_resp = await cli.get(f"/v1/runs/{data['run_id']}")
                status = await status_resp.json()
                if status["status"] == "completed":
                    break
                await asyncio.sleep(0.05)

            mock_agent.run_conversation.assert_called_once()
            kwargs = mock_agent.run_conversation.call_args.kwargs
            assert kwargs["user_message"] == [
                {"type": "text", "text": "describe this"},
                {"type": "image_url", "image_url": {"url": "https://example.com/final.png"}},
            ]
            assert kwargs["conversation_history"] == [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "first turn"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/first.png"}},
                    ],
                }
            ]
