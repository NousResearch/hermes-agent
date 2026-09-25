"""Regression tests for the serving-provider field in stream diagnostics (#90216).

OpenRouter-style relays re-roll the downstream provider per request — three consecutive calls
for one model came back served by Alibaba, Novita and StreamLake — and report the winner only
inside the delta chunk bodies (``{"provider": "Novita", ...}``).  Those responses carry no
``x-openrouter-provider`` header (just ``cf-ray`` / ``server: cloudflare``), so the existing
header snapshot cannot attribute a mid-stream drop to a provider at all.

Contract under test:

- A mid-stream drop's retry WARNING names the downstream from the attempt's first chunk.
- The ``post_api_request`` hook's ``response`` payload carries it as ``upstream_provider``.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

from tests.agent.test_run_agent import (  # noqa: F401  (_make_tool_defs used by the agent fixture)
    _make_tool_defs,
    _mock_response,
)
from tests.agent.test_first_chunk_at_hook import (  # noqa: F401  (shared fixture + harness)
    _make_stream_chunk,
    _run_with_hooks,
    agent,
)


def _chunk_with_provider(content=None, finish_reason=None, model=None, provider=None):
    """Streaming chunk plus the relay's per-chunk ``provider`` field."""
    chunk = _make_stream_chunk(content=content, finish_reason=finish_reason, model=model)
    if provider is not None:
        chunk.provider = provider
    return chunk


@patch("run_agent.AIAgent._create_request_openai_client")
@patch("run_agent.AIAgent._close_request_openai_client")
def test_mid_stream_drop_retry_line_names_the_serving_provider(_mock_close, mock_create, agent, caplog):
    """A real stream: the first chunk says who served it, then the connection drops."""

    def _dropping_stream():
        yield _chunk_with_provider(provider="Novita")  # no content: nothing delivered, so it retries
        raise ConnectionError("peer closed connection mid-stream")

    ok = [_chunk_with_provider(content="ok", finish_reason="stop", model="test-model", provider="Alibaba")]
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [_dropping_stream(), iter(ok)]
    mock_create.return_value = mock_client
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False

    with caplog.at_level(logging.WARNING):
        agent._interruptible_streaming_api_call({})

    drops = [r.getMessage() for r in caplog.records if "Stream drop" in r.getMessage()]
    assert drops, [r.getMessage() for r in caplog.records]
    assert "serving_provider=Novita" in drops[0]


# ── Conversation loop / hook payload level ───────────────────────────────


class TestServingProviderReachesPostApiRequest:
    """``upstream_provider`` on the post_api_request payload (plugin route auditing)."""

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_streamed_attempt_reports_the_serving_provider(self, _mock_close, mock_create, agent):
        chunks = [
            _chunk_with_provider(content="Hello", provider="Novita"),
            _chunk_with_provider(content=" world", finish_reason="stop", model="test-model", provider="Novita"),
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)
        mock_create.return_value = mock_client
        # A registered stream consumer forces the streaming path even though agent.client is a Mock.
        agent.stream_delta_callback = lambda _text: None

        result, post = _run_with_hooks(agent)

        assert result["final_response"] == "Hello world"
        assert len(post) == 1
        assert post[0]["response"]["upstream_provider"] == "Novita"
