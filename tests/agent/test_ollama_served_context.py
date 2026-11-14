"""Ollama's /v1 route drops the per-request num_ctx; /api/ps tells the truth after the first response.

Covers agent/ollama_served_context.py (#43900, #132607): a served window smaller than the detected
one must warn once and clamp the compressor; a matching window must stay silent.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.ollama_served_context import reconcile_served_context_after_response


def _agent(believed=131072, window=131072, base_url="http://localhost:11434/v1"):
    compressor = MagicMock()
    compressor.context_length = window
    return SimpleNamespace(
        model="llama3.2:1b", base_url=base_url, api_key="ollama", provider="custom", api_mode="chat_completions",
        _ollama_num_ctx=believed, context_compressor=compressor, _emit_warning=MagicMock(),
    )


def test_served_window_below_detected_warns_once_and_clamps_compressor():
    agent = _agent()
    with patch("agent.ollama_served_context.query_ollama_served_context", return_value=4096) as probe:
        reconcile_served_context_after_response(agent, api_call_count=1)
        reconcile_served_context_after_response(agent, api_call_count=1)
    probe.assert_called_once()
    agent._emit_warning.assert_called_once()
    message = agent._emit_warning.call_args.args[0]
    assert "4,096" in message and "131,072" in message and "OLLAMA_CONTEXT_LENGTH=131072" in message
    assert agent.context_compressor.update_model.call_args.kwargs["context_length"] == 4096


def test_served_window_matching_detected_is_silent():
    agent = _agent(believed=65536, window=65536)
    with patch("agent.ollama_served_context.query_ollama_served_context", return_value=65536):
        reconcile_served_context_after_response(agent, api_call_count=1)
    agent._emit_warning.assert_not_called()
    agent.context_compressor.update_model.assert_not_called()
