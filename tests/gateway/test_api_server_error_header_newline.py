"""#133848: the X-Hermes-Error header must not carry control characters.

A failed turn's error text can contain newlines — OpenRouter's guardrail 404 does
("...following reasons ...:\\nModel blocked by guardrail..."). aiohttp refuses header values
containing control characters, so writing that text raw raised ValueError while the headers
were serialized and dropped the connection instead of returning the 200 + error-extras
response the client was owed.
"""

import pytest
from aiohttp.test_utils import TestClient, TestServer
from unittest.mock import AsyncMock, patch

from gateway.platforms.api_server_openai_routes import _header_safe_error_text
from tests.gateway.test_api_server import _create_app, _make_adapter

OPENROUTER_GUARDRAIL_404 = (
    "NotFoundError: 0 endpoints out of 3 requested are available matching your guardrail "
    "restrictions and data policy. We removed them for the following reasons ...:\n"
    "Model blocked by guardrail: 3 endpoints excluded; ..."
)


def _soft_fail_result(error: str) -> dict:
    """Failed turn that still produced text — the 200 + hermes-extras + X-Hermes-Error path."""
    return {
        "final_response": "partial answer",
        "completed": False,
        "partial": False,
        "failed": True,
        "error": error,
        "messages": [],
        "api_calls": 1,
    }


class TestHeaderSafeErrorText:
    def test_folds_newline_runs_to_a_single_space(self):
        out = _header_safe_error_text("reasons):\nModel blocked by guardrail")
        assert out == "reasons): Model blocked by guardrail"

    def test_no_control_characters_survive(self):
        # exactly what aiohttp rejects ("Forbidden control character detected in headers")
        out = _header_safe_error_text("a\r\nb\rc\nd\te\x00f\x7fg\x1bh")
        assert not [c for c in out if ord(c) < 0x20 or ord(c) == 0x7F]
        assert " " in out


class TestChatCompletionsErrorHeader:
    @pytest.mark.asyncio
    async def test_newline_error_still_returns_the_response(self):
        adapter = _make_adapter()
        app = _create_app(adapter)
        result = _soft_fail_result(OPENROUTER_GUARDRAIL_404)
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (result, usage)
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={"model": "hermes-agent", "messages": [{"role": "user", "content": "hello"}]},
                )

            assert resp.status == 200
            header = resp.headers.get("X-Hermes-Error", "")
            assert header, "the error header must still be sent"
            assert not [c for c in header if ord(c) < 0x20 or ord(c) == 0x7F]
            assert "Model blocked by guardrail" in header
            data = await resp.json()
            assert data["choices"][0]["finish_reason"] == "error"
