"""Regression tests for the serving-provider field in stream diagnostics (#90216).

OpenRouter-style relays re-roll the downstream provider per request — three consecutive calls
for one model came back served by Alibaba, Novita and StreamLake — and report the winner only
inside the delta chunk bodies (``{"provider": "Novita", ...}``).  Those responses carry no
``x-openrouter-provider`` header (just ``cf-ray`` / ``server: cloudflare``), so the existing
header snapshot cannot attribute a mid-stream drop to a provider at all.

Contract under test:

- The first non-empty top-level ``provider`` in a chunk body lands in the per-attempt diag dict
  as ``serving_provider``; later chunks must not rewrite that attribution.
- ``log_stream_retry`` names it on the drop line (``serving_provider=-`` when unknown).
- A successful streamed attempt stashes it on the agent and the ``post_api_request`` hook
  payload carries it as ``upstream_provider`` — refreshed per attempt, never leaked from a
  prior call.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent
from tests.agent.test_run_agent import (  # noqa: F401  (_make_tool_defs used by the agent fixture)
    _make_tool_defs,
    _mock_response,
)
from tests.agent.test_first_chunk_at_hook import (  # noqa: F401  (shared fixture + harness)
    _make_stream_chunk,
    _run_with_hooks,
    agent,
)


def _make_agent() -> AIAgent:
    """Standalone agent for the diag/log-level tests (no conversation loop involved)."""
    return AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


def _chunk_with_provider(content=None, finish_reason=None, model=None, provider=None):
    """Streaming chunk plus the relay's per-chunk ``provider`` field."""
    chunk = _make_stream_chunk(content=content, finish_reason=finish_reason, model=model)
    if provider is not None:
        chunk.provider = provider
    return chunk


# ── Diag dict level ──────────────────────────────────────────────────────


def test_diag_records_first_serving_provider_from_chunk_body():
    agent_ = _make_agent()
    diag = AIAgent._stream_diag_init()
    assert diag["serving_provider"] is None

    agent_._stream_diag_note_serving_provider(diag, _chunk_with_provider(content="Hi", provider="Novita"))
    assert diag["serving_provider"] == "Novita"

    # The attempt was served by whatever produced its first chunk; a per-chunk rewrite would
    # just track the last one and lose the drop attribution.
    agent_._stream_diag_note_serving_provider(diag, _chunk_with_provider(content="!", provider="StreamLake"))
    assert diag["serving_provider"] == "Novita"


def test_diag_reads_provider_from_model_extra_and_ignores_chunks_without_one():
    """The OpenAI SDK parks unknown top-level fields in pydantic ``model_extra``."""
    agent_ = _make_agent()
    diag = AIAgent._stream_diag_init()

    agent_._stream_diag_note_serving_provider(diag, _chunk_with_provider(content="plain"))
    assert diag["serving_provider"] is None

    agent_._stream_diag_note_serving_provider(diag, SimpleNamespace(model_extra={"provider": "Alibaba"}))
    assert diag["serving_provider"] == "Alibaba"

    # Empty / non-string values are not a serving provider either.
    other = AIAgent._stream_diag_init()
    agent_._stream_diag_note_serving_provider(other, SimpleNamespace(provider="   "))
    assert other["serving_provider"] is None


def test_log_stream_retry_names_the_serving_provider(caplog):
    agent_ = _make_agent()
    agent_.provider = "openrouter"

    diag = AIAgent._stream_diag_init()
    diag["serving_provider"] = "Novita"

    with caplog.at_level(logging.WARNING):
        agent_._log_stream_retry(
            kind="drop", error=ConnectionError("peer closed"), attempt=2, max_attempts=3,
            mid_tool_call=False, diag=diag,
        )

    msg = next(r.getMessage() for r in caplog.records if "Stream drop" in r.getMessage())
    assert "serving_provider=Novita" in msg

    caplog.clear()
    unknown = AIAgent._stream_diag_init()
    with caplog.at_level(logging.WARNING):
        agent_._log_stream_retry(
            kind="drop", error=ConnectionError("peer closed"), attempt=2, max_attempts=3,
            mid_tool_call=False, diag=unknown,
        )
    msg = next(r.getMessage() for r in caplog.records if "Stream drop" in r.getMessage())
    assert "serving_provider=-" in msg


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
        assert post[0]["upstream_provider"] == "Novita"
        assert agent._last_serving_provider == "Novita"

    def test_non_streamed_attempt_reports_no_upstream_provider(self, agent):
        """A stale value from an earlier call must not leak into the next payload."""
        agent._last_serving_provider = "Novita"  # as a prior streamed attempt left it
        agent.client.chat.completions.create.return_value = _mock_response(
            content="Done", finish_reason="stop"
        )

        result, post = _run_with_hooks(agent)

        assert result["final_response"] == "Done"
        assert len(post) == 1
        assert post[0]["upstream_provider"] is None
