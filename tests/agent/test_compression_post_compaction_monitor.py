from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.conversation_compression import compress_context


class _TodoStore:
    def format_for_injection(self):
        return ""


def _make_agent(compressed_messages, *, threshold_tokens=50_000):
    compressor = MagicMock()
    compressor.compress.return_value = compressed_messages
    compressor.compression_count = 1
    compressor.threshold_tokens = threshold_tokens
    compressor._last_compress_aborted = False
    compressor._last_summary_error = None
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None

    agent = SimpleNamespace(
        session_id="session-1",
        model="test/model",
        tools=None,
        context_compressor=compressor,
        _compression_feasibility_checked=True,
        _memory_manager=None,
        _todo_store=_TodoStore(),
        _session_db=None,
        _cached_system_prompt=None,
        _emit_status=MagicMock(),
        _emit_warning=MagicMock(),
        _vprint=MagicMock(),
        _invalidate_system_prompt=MagicMock(),
        _build_system_prompt=MagicMock(return_value="rebuilt system prompt"),
    )
    return agent


def test_post_compaction_monitor_warns_inline_when_compressed_context_still_exceeds_threshold(monkeypatch):
    """The monitor is a post-compress hook: it runs during compress_context, not on a scheduler."""
    messages = [{"role": "user", "content": f"before {i}"} for i in range(12)]
    compressed = [{"role": "user", "content": "summary"}, {"role": "user", "content": "tail"}]
    agent = _make_agent(compressed, threshold_tokens=50_000)
    monkeypatch.setattr(
        "agent.conversation_compression.estimate_request_tokens_rough",
        lambda *_args, **_kwargs: 55_000,
    )

    returned, _prompt = compress_context(agent, messages, "sys", approx_tokens=80_000)

    assert returned == compressed
    agent._emit_warning.assert_called_once()
    warning = agent._emit_warning.call_args.args[0]
    assert "Compression pressure remains high" in warning
    assert "80,000→55,000" in warning
    assert "12→2 messages" in warning
    assert "post-compression hook" in warning


def test_post_compaction_monitor_does_not_warn_when_post_pressure_is_low(monkeypatch):
    messages = [{"role": "user", "content": f"before {i}"} for i in range(12)]
    compressed = [{"role": "user", "content": "summary"}, {"role": "user", "content": "tail"}]
    agent = _make_agent(compressed, threshold_tokens=50_000)
    monkeypatch.setattr(
        "agent.conversation_compression.estimate_request_tokens_rough",
        lambda *_args, **_kwargs: 30_000,
    )

    compress_context(agent, messages, "sys", approx_tokens=80_000)

    agent._emit_warning.assert_not_called()


def test_post_compaction_monitor_is_not_a_periodic_scheduler(monkeypatch):
    """The hook evaluates the just-executed compression result with no cron/scheduler dependency."""
    messages = [{"role": "user", "content": f"before {i}"} for i in range(8)]
    compressed = [{"role": "user", "content": "summary"}]
    agent = _make_agent(compressed, threshold_tokens=10_000)
    monkeypatch.setattr(
        "agent.conversation_compression.estimate_request_tokens_rough",
        lambda *_args, **_kwargs: 12_000,
    )

    compress_context(agent, messages, "sys", approx_tokens=20_000)

    agent.context_compressor.compress.assert_called_once_with(
        messages,
        current_tokens=20_000,
        focus_topic=None,
        force=False,
    )
    agent._emit_warning.assert_called_once()
