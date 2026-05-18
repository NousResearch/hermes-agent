"""Regression tests for exact vs projected compression token tracking."""

import sys
import types
from pathlib import Path

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

from unittest.mock import patch

import run_agent
from agent.context_compressor import ContextCompressor


def _make_agent_and_compressor():
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        compressor = ContextCompressor(model="test/model", quiet_mode=True)

    compressor.compress = lambda messages, current_tokens=None, focus_topic=None: [
        {"role": "assistant", "content": "compressed"}
    ]

    agent = run_agent.AIAgent.__new__(run_agent.AIAgent)
    agent.session_id = "session-1"
    agent.model = "test/model"
    agent.platform = ""
    agent.context_compressor = compressor
    agent._memory_manager = None
    agent._todo_store = types.SimpleNamespace(format_for_injection=lambda: "")
    agent._invalidate_system_prompt = lambda: None
    agent._build_system_prompt = lambda system_message: f"sys::{system_message}"
    agent._cached_system_prompt = None
    agent._session_db = None
    agent._session_db_created = False
    agent._last_flushed_db_idx = 0
    agent._last_compression_summary_warning = None
    agent._last_aux_fallback_warning_key = None
    agent._emit_warning = lambda *a, **k: None
    agent._vprint = lambda *a, **k: None
    agent.status_callback = None
    agent.commit_memory_session = lambda messages: None
    agent.logs_dir = Path("/tmp")
    agent.session_log_file = agent.logs_dir / "session.json"
    agent.tools = [{"type": "function", "function": {"name": "tool", "description": "schema padding" * 40}}]
    agent.log_prefix = ""
    return agent, compressor


def test_compress_context_preserves_last_provider_prompt_tokens():
    agent, compressor = _make_agent_and_compressor()
    compressor.last_provider_prompt_tokens = 800
    compressor.projected_prompt_tokens = 800
    compressor.projected_prompt_tokens_source = "provider_exact"
    compressor._transcript_mutated_since_api = False

    messages, new_system = agent._compress_context(
        [{"role": "user", "content": "hello"}],
        "base-system",
        approx_tokens=900,
    )

    assert new_system == "sys::base-system"
    assert messages == [{"role": "assistant", "content": "compressed"}]
    assert compressor.last_provider_prompt_tokens == 800
    assert compressor.projected_prompt_tokens_source == "estimated_post_compression"
    assert compressor._transcript_mutated_since_api is True


def test_post_compression_pressure_does_not_reuse_stale_provider_exact():
    agent, compressor = _make_agent_and_compressor()
    compressor.last_provider_prompt_tokens = 800
    compressor.projected_prompt_tokens = 800
    compressor.projected_prompt_tokens_source = "provider_exact"
    compressor._transcript_mutated_since_api = False

    messages, new_system = agent._compress_context(
        [{"role": "user", "content": "hello"}],
        "base-system",
        approx_tokens=900,
    )

    tokens, source = compressor.get_current_request_pressure(
        messages=messages,
        system_prompt=new_system,
        tools=agent.tools,
    )

    assert source == "estimated_post_compression"
    assert tokens == compressor.projected_prompt_tokens
    assert tokens != 800
