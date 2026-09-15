"""The concurrent completion log must report the serialized size of multimodal results.

The native vision fast path returns an envelope dict; the concurrent worker logged
``len(result)`` directly, so a ~100 KB image payload showed up as ``completed
(0.14s, 4 chars)`` — the dict key count. The sequential path already measures the
serialized form; the concurrent log must agree, or parallel multimodal calls look
truncated exactly while debugging a repeat-call loop.
"""

import json
import logging
import threading
import time
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _isolate_hermes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir(exist_ok=True)


def _make_agent(monkeypatch):
    """Minimal AIAgent-like stub, mirroring test_start_order_gate.py."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "")
    import run_agent as _ra

    class _Stub:
        _interrupt_requested = False
        _interrupt_message = None
        _execution_thread_id = threading.current_thread().ident
        _interrupt_thread_signal_pending = False
        log_prefix = ""
        quiet_mode = True
        verbose_logging = False
        log_prefix_chars = 200
        _checkpoint_mgr = MagicMock(enabled=False)
        tool_progress_callback = None
        tool_start_callback = None
        tool_complete_callback = None
        tool_progress_mode = "off"
        _todo_store = MagicMock()
        _session_db = None
        valid_tool_names = set()
        _turns_since_memory = 0
        _iters_since_skill = 0
        _current_tool = None
        _last_activity = 0
        _print_fn = print
        session_id = ""
        _current_turn_id = ""
        _current_api_request_id = ""
        _active_children: list = []

        def __init__(self):
            self._tool_worker_threads: set = set()
            self._tool_worker_threads_lock = threading.Lock()
            self._active_children_lock = threading.Lock()

        def _touch_activity(self, desc):
            self._last_activity = time.time()

        def _vprint(self, msg, force=False):
            pass

        def _safe_print(self, msg):
            pass

        def _should_emit_quiet_tool_messages(self):
            return False

        def _should_start_quiet_spinner(self):
            return False

        def _has_stream_consumers(self):
            return False

        def _tool_result_content_for_active_model(self, name, result):
            return result

        def _record_file_mutation_result(self, *a, **kw):
            pass

    stub = _Stub()
    stub._subdirectory_hints = MagicMock()
    stub._subdirectory_hints.check_tool_call = lambda *a, **kw: None
    stub._flush_messages_to_session_db = lambda *a, **kw: None
    stub._append_guardrail_observation = lambda name, result, *a, **kw: result
    stub._execute_tool_calls_concurrent = (
        _ra.AIAgent._execute_tool_calls_concurrent.__get__(stub)
    )
    stub.interrupt = _ra.AIAgent.interrupt.__get__(stub)
    stub.clear_interrupt = _ra.AIAgent.clear_interrupt.__get__(stub)
    stub._apply_pending_steer_to_tool_results = lambda *a, **kw: None
    stub._guardrail_block_result = lambda d: json.dumps({"error": "blocked"})
    return stub


class _FakeToolCall:
    def __init__(self, name, call_id):
        self.function = MagicMock(name=name, arguments="{}")
        self.function.name = name
        self.id = call_id


class _FakeAssistantMsg:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls


def _multimodal_envelope() -> dict:
    """Shaped like the native vision tool-result envelope (tools/vision_tools.py)."""
    return {
        "_multimodal": True,
        "content": [
            {"type": "text", "text": "Image loaded into your context — " + "x" * 800},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64," + "A" * 4000},
            },
        ],
        "text_summary": "Image attached natively for the main model. Answer using built-in vision.",
        "meta": {"image_url": "photo.jpg", "size_bytes": 204800, "native_vision": True},
    }


def test_concurrent_completion_log_reports_serialized_multimodal_size(
    monkeypatch, caplog
):
    agent = _make_agent(monkeypatch)
    import agent.tool_executor as te

    monkeypatch.setattr(te, "_resolve_concurrent_tool_timeout", lambda: 6.0)

    envelope = _multimodal_envelope()
    agent._tool_guardrails = MagicMock()
    agent._tool_guardrails.before_call = lambda *a, **kw: MagicMock(
        allows_execution=True
    )
    agent._invoke_tool = MagicMock(return_value=envelope)

    msg = _FakeAssistantMsg([_FakeToolCall("vision_analyze", "tc_1")])
    messages: list = []
    with caplog.at_level(logging.INFO, logger="agent.tool_executor"):
        agent._execute_tool_calls_concurrent(msg, messages, "task")

    completed = [
        r
        for r in caplog.records
        if "vision_analyze" in r.getMessage() and "completed" in r.getMessage()
    ]
    assert completed, (
        "no completion log line captured for the concurrent vision_analyze call"
    )

    logged = completed[0].getMessage()
    expected = len(str(envelope))
    assert f", {expected} chars)" in logged, (
        f"completion log did not report the serialized multimodal size: {logged!r}"
    )


def test_concurrent_completion_log_still_reports_plain_string_size(monkeypatch, caplog):
    agent = _make_agent(monkeypatch)
    import agent.tool_executor as te

    monkeypatch.setattr(te, "_resolve_concurrent_tool_timeout", lambda: 6.0)

    text_result = "A" * 123
    agent._tool_guardrails = MagicMock()
    agent._tool_guardrails.before_call = lambda *a, **kw: MagicMock(
        allows_execution=True
    )
    agent._invoke_tool = MagicMock(return_value=text_result)

    msg = _FakeAssistantMsg([_FakeToolCall("terminal", "tc_1")])
    messages: list = []
    with caplog.at_level(logging.INFO, logger="agent.tool_executor"):
        agent._execute_tool_calls_concurrent(msg, messages, "task")

    completed = [
        r
        for r in caplog.records
        if "terminal" in r.getMessage() and "completed" in r.getMessage()
    ]
    assert completed, "no completion log line captured for the concurrent terminal call"

    logged = completed[0].getMessage()
    assert f", 123 chars)" in logged, (
        f"plain-string results must keep logging their exact length: {logged!r}"
    )
