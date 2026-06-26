"""Regression tests for compression-triggered context handoff writes."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from run_agent import AIAgent


def _make_agent(engine) -> AIAgent:
    """Create a minimal AIAgent shell for direct _compress_context tests."""
    agent = object.__new__(AIAgent)
    agent.context_compressor = engine
    agent.session_id = "sess-handoff"
    agent.model = "test/model"
    agent.provider = "test-provider"
    agent.platform = "qqbot"
    agent.logs_dir = MagicMock()
    agent.quiet_mode = True
    agent._todo_store = SimpleNamespace(format_for_injection=lambda: "继续实现 hook")
    agent._memory_manager = None
    agent._session_db = None
    agent._cached_system_prompt = None
    agent.log_prefix = ""
    agent.tools = []
    agent._gateway_session_key = "qq:chat"
    agent._current_turn_id = "turn-1"
    agent._compression_feasibility_checked = True
    agent._last_compaction_in_place = False
    agent.status_callback = None
    agent.event_callback = None
    agent._custom_providers = {}
    agent._vprint = lambda *a, **kw: None
    agent._emit_status = lambda *a, **kw: None
    agent._emit_warning = lambda *a, **kw: None
    agent._invalidate_system_prompt = lambda *a, **kw: None
    agent._build_system_prompt = lambda *a, **kw: "new-system-prompt"
    agent.commit_memory_session = lambda *a, **kw: None
    return agent


def _messages() -> list[dict[str, str]]:
    """Return messages used by compression handoff tests."""
    return [
        {"role": "user", "content": "继续 Hermes Auto Handoff"},
        {"role": "assistant", "content": "先写测试"},
    ]


def _latest_payload(home: Path) -> dict:
    """Load the latest handoff payload from a temp Hermes home."""
    return json.loads((home / "handoffs" / "latest.json").read_text())


def test_compress_context_writes_pre_compression_handoff(tmp_path: Path, monkeypatch):
    """Successful compression writes a pre-compression handoff snapshot."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    engine = MagicMock()
    engine.compress.return_value = [{"role": "user", "content": "summary"}]
    engine.compression_count = 1
    engine._last_compress_aborted = False
    engine._last_summary_error = None
    engine._last_aux_model_failure_model = None
    engine._last_aux_model_failure_error = None
    engine.last_prompt_tokens = 0
    engine.last_completion_tokens = 0
    agent = _make_agent(engine)

    compressed, system_prompt = agent._compress_context(
        _messages(),
        "system",
        approx_tokens=99,
        focus_topic="handoff",
    )

    assert compressed[0] == {"role": "user", "content": "summary"}
    assert compressed[-1] == {"role": "user", "content": "继续实现 hook"}
    assert system_prompt == "new-system-prompt"
    payload = _latest_payload(tmp_path)
    assert payload["reason"] == "pre_compression"
    assert payload["approx_tokens"] == 99
    assert payload["focus_topic"] == "handoff"


def test_compress_context_overwrites_handoff_when_summary_aborts(
    tmp_path: Path, monkeypatch
):
    """Summary abort writes a failure-specific handoff while preserving messages."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    engine = MagicMock()
    engine.compress.return_value = _messages()
    engine.compression_count = 0
    engine._last_compress_aborted = True
    engine._last_summary_error = "aux 502"
    engine._last_aux_model_failure_model = None
    engine._last_aux_model_failure_error = None
    agent = _make_agent(engine)
    messages = _messages()

    returned, _ = agent._compress_context(messages, "system", approx_tokens=1000)

    assert returned is messages
    payload = _latest_payload(tmp_path)
    assert payload["reason"] == "compression_aborted"
    assert payload["error"] == "aux 502"


def test_compress_context_writes_handoff_before_reraising_exception(
    tmp_path: Path, monkeypatch
):
    """Unexpected compressor exceptions still leave a handoff before re-raise."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    engine = MagicMock()
    engine.compress.side_effect = RuntimeError("boom")
    engine.compression_count = 0
    agent = _make_agent(engine)

    try:
        agent._compress_context(_messages(), "system", approx_tokens=123)
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("expected RuntimeError")

    payload = _latest_payload(tmp_path)
    assert payload["reason"] == "compression_exception"
    assert "RuntimeError: boom" == payload["error"]
