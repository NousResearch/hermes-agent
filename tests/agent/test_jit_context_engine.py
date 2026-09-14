"""Test JIT Context Engine loading, compilation, scoping, and bounded selection."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from agent.agent_init import _enforce_minimum_context, _select_context_engine
from agent.conversation_loop import _ollama_context_limit_error
from plugins.context_engine import load_context_engine


def test_jit_context_engine_loaded_by_name():
    engine = load_context_engine("jit")
    assert engine is not None
    assert engine.name == "jit"
    assert engine.threshold_tokens <= 35000


def test_agent_init_selects_jit_engine():
    cfg = {"context": {"engine": "jit"}}
    engine = _select_context_engine(cfg)
    assert engine is not None
    assert engine.name == "jit"


def test_jit_compiles_non_empty_capsule_without_external_plugins(tmp_path):
    """Prove self-contained L0/L1 compiler compiles real <ONA_CONTEXT> capsule without missing modules."""
    engine = load_context_engine("jit")
    assert engine is not None
    engine.db_path = tmp_path / "test_jit.db"
    engine.session_id = "test_session_isolated"

    capsule = engine._resolve_capsule(
        incoming_message={"role": "user", "content": "Implement project boocco feature with strict compliance."},
        history_messages=[],
    )

    assert isinstance(capsule, str)
    assert len(capsule) > 0
    assert "<ONA_CONTEXT" in capsule
    assert "</ONA_CONTEXT>" in capsule
    assert 'scope="boocco"' in capsule or 'scope="general"' in capsule


def test_jit_bounds_current_turn_tool_output_and_preserves_tool_call_id():
    """Verify tool output bounding for current turn while preserving tool_call_id and schema pairing."""
    engine = load_context_engine("jit")
    assert engine is not None
    engine.max_current_tool_chars = 1500

    huge_payload = "A" * 20000
    messages = [
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": "Please read the large configuration file."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_read_large_cfg_987",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "huge.txt"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_read_large_cfg_987",
            "name": "read_file",
            "content": huge_payload,
        },
    ]

    selected = engine.select_context(messages)
    assert len(selected) == 4

    tool_msg = selected[3]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call_read_large_cfg_987"
    assert tool_msg["name"] == "read_file"
    # Content must be clamped to predictable size
    assert len(tool_msg["content"]) < 3000
    assert "truncated by JIT context engine" in tool_msg["content"]
    assert "current turn" in tool_msg["content"]

    # Assistant message and its tool call id must remain completely intact
    assistant_msg = selected[2]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["tool_calls"][0]["id"] == "call_read_large_cfg_987"


def test_non_jit_ollama_16k_still_enforces_safety_floor():
    """Verify non-JIT sessions retain the safety floor for small Ollama contexts."""
    # 1. Non-JIT mock agent (default compressor)
    non_jit_agent = MagicMock()
    non_jit_agent.provider = "ollama"
    non_jit_agent.model = "qwen3.8:9b"
    non_jit_agent._ollama_num_ctx = 16384
    non_jit_agent.tools = [{"type": "function", "function": {"name": "terminal"}}]
    non_jit_compressor = MagicMock()
    non_jit_compressor.name = "compressor"
    non_jit_agent.context_compressor = non_jit_compressor

    err = _ollama_context_limit_error(non_jit_agent, request_tokens=5000)
    assert err is not None
    assert "Ollama loaded `qwen3.8:9b` with only 16,384 tokens of runtime context" in err
    assert "context.engine: jit" in err

    # 2. JIT mock agent
    jit_engine = load_context_engine("jit")
    non_jit_agent.context_compressor = jit_engine

    err_jit = _ollama_context_limit_error(non_jit_agent, request_tokens=5000)
    assert err_jit is None


def test_enforce_minimum_context_scoping():
    """Verify _enforce_minimum_context raises for non-JIT below 64K, but passes for JIT."""
    agent = MagicMock()
    agent.provider = "ollama"
    agent.model = "qwen3.8:9b"
    agent._config_context_length = 16384

    # Non-JIT compressor below floor
    compressor = MagicMock()
    compressor.name = "compressor"
    compressor.context_length = 16384
    agent.context_compressor = compressor

    with pytest.raises(ValueError) as excinfo:
        _enforce_minimum_context(agent)
    assert "below the minimum 64,000 required" in str(excinfo.value)
    assert "context.engine: jit" in str(excinfo.value)

    # With JIT engine, below floor is safely permitted
    jit_engine = load_context_engine("jit")
    assert jit_engine is not None
    jit_engine.context_length = 16384
    agent.context_compressor = jit_engine

    # Should not raise
    _enforce_minimum_context(agent)
