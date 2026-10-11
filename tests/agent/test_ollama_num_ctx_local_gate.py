"""Regression tests for the local-endpoint gate on ``model.ollama_num_ctx`` (#134140).

The explicit override used to apply on any base_url, so a pin intended for a sibling
local alias in the same ``model:`` block clamped the context compressor of the hosted
default model down to the alias's num_ctx (e.g. a 500K-window model compressed at
``floor(65536 * 0.85)``).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

# Imported at collection time on purpose: _ra() lazily imports run_agent, whose module
# body probes the checkout's git dir — the per-test home-io guard refuses that probe in
# a worktree whose git dir lives under the real HERMES_HOME.
import run_agent  # noqa: F401
from agent.agent_init import (
    _clamp_compressor_to_ollama_num_ctx,
    _configure_ollama_num_ctx,
)


def _make_agent(base_url, compressor_window=500000):
    compressor = SimpleNamespace(context_length=compressor_window)
    compressor.update_model = MagicMock()
    return SimpleNamespace(
        model="grok-4.7",
        base_url=base_url,
        api_key="k",
        provider="xai-oauth",
        api_mode="chat",
        quiet_mode=True,
        _ollama_num_ctx=None,
        context_compressor=compressor,
    )


class TestOllamaNumCtxLocalGate:
    def test_override_ignored_on_hosted_endpoint(self):
        agent = _make_agent("https://api.x.ai/v1/")
        _configure_ollama_num_ctx(agent, {"ollama_num_ctx": 65536}, None)
        assert agent._ollama_num_ctx is None

    def test_override_ignored_without_base_url(self):
        agent = _make_agent(None)
        _configure_ollama_num_ctx(agent, {"ollama_num_ctx": 65536}, None)
        assert agent._ollama_num_ctx is None

    def test_override_applied_on_local_endpoint(self):
        agent = _make_agent("http://127.0.0.1:8080/v1")
        _configure_ollama_num_ctx(agent, {"ollama_num_ctx": 65536}, None)
        assert agent._ollama_num_ctx == 65536

    def test_hosted_compressor_window_not_clamped(self):
        agent = _make_agent("https://api.x.ai/v1/")
        _configure_ollama_num_ctx(agent, {"ollama_num_ctx": 65536}, None)
        _clamp_compressor_to_ollama_num_ctx(agent)
        assert agent.context_compressor.context_length == 500000
        agent.context_compressor.update_model.assert_not_called()

    def test_local_compressor_window_clamped(self):
        agent = _make_agent("http://127.0.0.1:8080/v1")
        _configure_ollama_num_ctx(agent, {"ollama_num_ctx": 65536}, None)
        _clamp_compressor_to_ollama_num_ctx(agent)
        agent.context_compressor.update_model.assert_called_once()
        assert (
            agent.context_compressor.update_model.call_args.kwargs["context_length"]
            == 65536
        )
