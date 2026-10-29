"""An empty compression summary must log why it failed before aborting.

A provider refusal (finish_reason=content_filter) and a reply carrying only
omitted/empty thinking both arrive here with no text, and both used to surface
as the same "returned empty content" error with no record of the stop reason,
so the logs cannot distinguish a content-filter stop from a provider-side
empty reply. The abort behavior is unchanged; the warning line records the
finish reason, output token count, and whether a reasoning field came back.
"""

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor


def _compressor():
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(
            model="main-model", provider="main-provider", quiet_mode=True,
            base_url="https://main.example/v1", api_key="sk-MAIN-SECRET",
            summary_model_override="summary-model",
        )


def _response(content="", *, finish_reason="stop", completion_tokens=0, reasoning=None):
    message = SimpleNamespace(content=content, reasoning=reasoning)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=SimpleNamespace(completion_tokens=completion_tokens),
    )


def test_empty_summary_logs_finish_reason_and_output_tokens(caplog):
    compressor = _compressor()
    refusal_stop = _response(finish_reason="content_filter", completion_tokens=17)

    with patch("agent.context_compressor.call_llm", return_value=refusal_stop):
        with caplog.at_level(logging.WARNING, logger="agent.context_compressor"):
            with pytest.raises(RuntimeError, match="returned empty content"):
                compressor._call_summary_llm("Summarize this conversation.", 0.0)

    warnings = [r for r in caplog.records if "carried no text" in r.getMessage()]
    assert len(warnings) == 1, f"expected one diagnostic warning, got: {caplog.records}"
    message = warnings[0].getMessage()
    assert "finish_reason=content_filter" in message
    assert "completion_tokens=17" in message
    assert "reasoning_field=absent" in message


def test_empty_summary_with_reasoning_field_reports_it_present(caplog):
    compressor = _compressor()
    # Thinking-only / omitted-thinking shape: a reasoning field came back but no text.
    thinking_only = _response(finish_reason="stop", completion_tokens=42, reasoning="")

    with patch("agent.context_compressor.call_llm", return_value=thinking_only):
        with caplog.at_level(logging.WARNING, logger="agent.context_compressor"):
            with pytest.raises(RuntimeError, match="returned empty content"):
                compressor._call_summary_llm("Summarize this conversation.", 0.0)

    warnings = [r for r in caplog.records if "carried no text" in r.getMessage()]
    assert len(warnings) == 1, f"expected one diagnostic warning, got: {caplog.records}"
    message = warnings[0].getMessage()
    assert "finish_reason=stop" in message
    assert "completion_tokens=42" in message
    assert "reasoning_field=present" in message
