"""/v1/runs streaming releases MEDIA-tag text the streaming resolver held back, on every terminal path."""

from unittest.mock import MagicMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_api_server_runs import _create_runs_app, _make_adapter


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.mark.asyncio
async def test_structured_failure_still_flushes_held_back_media_text(adapter):
    """A structured failed result must not swallow buffered MEDIA-tag text.

    The streaming MEDIA-tag resolver holds back everything from the last
    "MEDIA:" onward until it can confirm whether a real tag completed. That
    buffer is only safe if EVERY terminal path releases it: a structured
    ``{"failed": True}`` result emits run.failed without going through the
    success branch, so a flush placed only there loses text that this
    endpoint previously delivered (raw, but delivered).
    """
    app = _create_runs_app(adapter)
    held_back = "MEDIA:/nonexistent/never-completed-"
    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_create_agent") as mock_create:
            mock_agent = MagicMock()

            def _run_conversation(*_args, **_kwargs):
                cb = mock_create.call_args.kwargs["stream_delta_callback"]
                cb("Saved to ")
                # Held back: a complete "MEDIA:" prefix whose path never
                # resolves, so the resolver keeps it pending to stream end.
                cb(held_back)
                return {"failed": True, "error": "provider rejected the request"}

            mock_agent.run_conversation.side_effect = _run_conversation
            mock_agent.session_prompt_tokens = 0
            mock_agent.session_completion_tokens = 0
            mock_agent.session_total_tokens = 0
            mock_create.return_value = mock_agent

            resp = await cli.post("/v1/runs", json={"input": "hello"})
            assert resp.status == 202
            run_id = (await resp.json())["run_id"]

            events_resp = await cli.get(f"/v1/runs/{run_id}/events")
            assert events_resp.status == 200
            body = await events_resp.text()

    assert "run.failed" in body
    # The pre-MEDIA text was already safe to emit during streaming...
    assert "Saved to " in body
    # ...and the held-back remainder must still arrive despite the failure.
    assert held_back in body
