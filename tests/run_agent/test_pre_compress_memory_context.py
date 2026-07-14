"""Behavior contracts for the pre-compression memory-context handoff."""

import os
from unittest.mock import MagicMock, patch

import pytest


def _make_agent(memory_manager, compressor):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "x"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=None,
            session_id="test-session",
            skip_context_files=True,
            skip_memory=True,
        )

    agent._memory_manager = memory_manager
    agent.context_compressor = compressor
    agent._compression_feasibility_checked = True
    agent._invalidate_system_prompt = lambda: None
    agent._build_system_prompt = lambda _message: "new-system-prompt"
    return agent


def _messages():
    return [{"role": "user", "content": f"message {i}"} for i in range(6)]


def _configure_engine_state(engine):
    engine.compression_count = 1
    engine.last_prompt_tokens = 0
    engine.last_completion_tokens = 0
    engine._last_summary_error = None
    engine._last_compress_aborted = False
    engine._last_aux_model_failure_model = None
    engine._last_aux_model_failure_error = None


def test_on_pre_compress_result_reaches_compressor_with_existing_options():
    manager = MagicMock()
    manager.on_pre_compress.return_value = "Checkpoint id: ctx-orchestrator"
    received = {}
    compressor = MagicMock()

    def capture_compress(
        incoming,
        current_tokens=None,
        focus_topic=None,
        force=False,
        memory_context="",
    ):
        received.update(
            current_tokens=current_tokens,
            focus_topic=focus_topic,
            force=force,
            memory_context=memory_context,
        )
        return [incoming[0], incoming[-1]]

    compressor.compress.side_effect = capture_compress
    _configure_engine_state(compressor)
    agent = _make_agent(manager, compressor)
    messages = _messages()

    agent._compress_context(
        messages,
        "sys",
        approx_tokens=100_000,
        focus_topic="checkpoint continuity",
        force=True,
    )

    manager.on_pre_compress.assert_called_once_with(messages)
    assert received == {
        "current_tokens": 100_000,
        "focus_topic": "checkpoint continuity",
        "force": True,
        "memory_context": "Checkpoint id: ctx-orchestrator",
    }


def test_legacy_engine_receives_only_supported_compression_arguments():
    manager = MagicMock()
    manager.on_pre_compress.return_value = "Checkpoint id: unsupported-by-legacy"
    calls = []

    class StrictLegacyEngine:
        def compress(self, messages, current_tokens=None):
            calls.append(current_tokens)
            return [messages[0], messages[-1]]

    engine = StrictLegacyEngine()
    _configure_engine_state(engine)
    agent = _make_agent(manager, engine)

    compressed, _prompt = agent._compress_context(
        _messages(),
        "sys",
        approx_tokens=100_000,
        focus_topic="unsupported focus",
        force=True,
    )

    assert len(compressed) == 2
    assert calls == [100_000]


def test_provider_context_is_strictly_sanitized_before_plugin_engine(monkeypatch):
    prefix_secret = "sk-" + "a" * 30
    query_secret = "opaque-query-secret"
    userinfo_value = "opaque-userinfo-value"
    fragment_secret = "FRAG_SECRET"
    relative_secret = "REL_SECRET"
    encoded_key_secret = "ENC_SECRET"
    network_userinfo_secret = "NET_SECRET"
    manager = MagicMock()
    manager.on_pre_compress.return_value = (
        f"api key: {prefix_secret}\n"
        f"callback: https://example.test/cb?access_token={query_secret}&state=ok\n"
        f"endpoint: https://user:{userinfo_value}@example.test/private\n"
        f"fragment: https://x.test/#access_token={fragment_secret}&view=public\n"
        f"relative: /resume?token={relative_secret}&view=public\n"
        f"encoded: https://x.test/cb?client%5Fsecret={encoded_key_secret}&view=public\n"
        f"network: //user:{network_userinfo_secret}@x.test/path"
    )
    received = []
    compressor = MagicMock()

    def capture_compress(messages, current_tokens=None, memory_context="", **_kwargs):
        received.append(memory_context)
        return [messages[0], messages[-1]]

    compressor.compress.side_effect = capture_compress
    _configure_engine_state(compressor)
    agent = _make_agent(manager, compressor)

    # Provider-to-engine handoff is an external-LLM egress boundary, so it
    # remains strict even when display/log redaction was explicitly disabled.
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
    agent._compress_context(_messages(), "sys", approx_tokens=100_000)

    assert len(received) == 1
    context = received[0]
    assert prefix_secret not in context
    assert query_secret not in context
    assert userinfo_value not in context
    assert fragment_secret not in context
    assert relative_secret not in context
    assert encoded_key_secret not in context
    assert network_userinfo_secret not in context
    assert "access_token=***" in context
    assert "https://user:***@example.test/private" in context
    assert "https://x.test/#access_token=***&view=public" in context
    assert "/resume?token=***&view=public" in context
    assert "client%5Fsecret=***&view=public" in context
    assert "//user:***@x.test/path" in context


def test_provider_context_is_bounded_before_plugin_engine():
    manager = MagicMock()
    manager.on_pre_compress.return_value = "HEAD-SENTINEL" + "x" * 8_000 + "TAIL-SENTINEL"
    received = []
    compressor = MagicMock()

    def capture_compress(messages, current_tokens=None, memory_context="", **_kwargs):
        received.append(memory_context)
        return [messages[0], messages[-1]]

    compressor.compress.side_effect = capture_compress
    _configure_engine_state(compressor)
    agent = _make_agent(manager, compressor)

    agent._compress_context(_messages(), "sys", approx_tokens=100_000)

    assert len(received) == 1
    context = received[0]
    assert len(context) <= 6_000
    assert context.startswith("HEAD-SENTINEL")
    assert context.endswith("TAIL-SENTINEL")
    assert "[memory provider context truncated]" in context


def test_internal_engine_type_error_propagates_after_one_call():
    manager = MagicMock()
    manager.on_pre_compress.return_value = "Checkpoint id: ctx-typeerror"
    calls = []

    class BrokenEngine:
        def compress(
            self,
            messages,
            current_tokens=None,
            focus_topic=None,
            force=False,
            memory_context="",
        ):
            calls.append(memory_context)
            raise TypeError("engine implementation bug")

    engine = BrokenEngine()
    _configure_engine_state(engine)
    agent = _make_agent(manager, engine)

    with pytest.raises(TypeError, match="engine implementation bug"):
        agent._compress_context(_messages(), "sys", approx_tokens=100_000)

    assert calls == ["Checkpoint id: ctx-typeerror"]
