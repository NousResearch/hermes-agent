"""The compression dispatch log must not name the main model for a route call_llm picks (#143576)."""

import logging
from types import SimpleNamespace
from unittest.mock import patch

from agent.context_compressor import ContextCompressor


def _compressor(**kwargs):
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(model="main-model", provider="main-provider", quiet_mode=True, **kwargs)


def _served_by(provider, model):
    def _call(**kwargs):
        kwargs["route_info"].update(provider=provider, model=model)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="summary " * 20), finish_reason="stop")])
    return _call


def test_auto_route_logs_the_model_that_served_the_summary(caplog):
    compressor = _compressor()
    with caplog.at_level(logging.INFO, logger="agent.context_compressor"), patch(
        "agent.context_compressor.call_llm", side_effect=_served_by("aux-provider", "aux-model")
    ):
        compressor._call_summary_llm("Summarize this conversation.", 0.0)

    messages = [r.getMessage() for r in caplog.records]
    dispatched = next(m for m in messages if m.startswith("Compression summary call dispatched"))
    finished = next(m for m in messages if m.startswith("Compression summary call finished"))
    assert "model=auto " in dispatched and "main-model" not in dispatched
    assert "provider=aux-provider model=aux-model " in finished


def test_explicit_summary_model_is_logged_as_requested(caplog):
    compressor = _compressor(summary_model_override="pinned-aux")
    with caplog.at_level(logging.INFO, logger="agent.context_compressor"), patch(
        "agent.context_compressor.call_llm", side_effect=_served_by("aux-provider", "pinned-aux")
    ):
        compressor._call_summary_llm("Summarize this conversation.", 0.0)

    dispatched = next(r.getMessage() for r in caplog.records
                      if r.getMessage().startswith("Compression summary call dispatched"))
    assert "model=pinned-aux " in dispatched
