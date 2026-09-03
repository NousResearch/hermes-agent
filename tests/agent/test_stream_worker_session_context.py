"""Regression tests for session attribution inside the streaming worker threads.

``interruptible_streaming_api_call`` runs the provider call on a worker thread
— one for the OpenAI/Anthropic path, another for Bedrock Converse. The
``[session]`` tag that every log line carries comes from a ``threading.local``
in :mod:`hermes_logging`, and ``_context_thread_target`` carries the caller's
ContextVars across the thread boundary but cannot carry a thread-local. Both
workers therefore start UNBOUND and everything they log is formatted WITHOUT a
session tag.

In a process serving concurrent sessions this is the difference between a
diagnosable failure and an anonymous one: each worker is where its own stream
failure is logged, so the one line carrying the cause is the one line that
names no session.

The two production-path tests below drive the real
``interruptible_streaming_api_call`` and assert on a record emitted from INSIDE
each worker — the placement is the point, so exercising the helper alone would
not catch a worker that was never bound. The caller's thread is deliberately
left unbound in both: the tag must come from ``agent.session_id``, not from
whatever the calling thread happened to inherit.

The remaining tests pin the helper's best-effort contract.
"""

from __future__ import annotations

import logging
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import hermes_logging
from agent import chat_completion_helpers as cch
from agent.chat_completion_helpers import _bind_worker_session_context

SESSION_ID = "20260101_000000_abcdef01"
SESSION_TAG = f" [{SESSION_ID}]"


class _WorkerRecords(logging.Handler):
    """Collect records emitted from threads other than the test's own.

    Handlers run synchronously on the emitting thread, so ``record.thread``
    identifies the worker that created the record — which is exactly the
    placement these tests are about.
    """

    def __init__(self, caller_thread_id: int):
        super().__init__()
        self._caller_thread_id = caller_thread_id
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread != self._caller_thread_id:
            self.records.append(record)

    def starting_with(self, prefix: str) -> list[logging.LogRecord]:
        return [r for r in self.records if str(r.msg).startswith(prefix)]


@pytest.fixture
def worker_records():
    """Capture worker-thread records with the caller's thread left UNBOUND."""
    # Idempotent; other suites reload modules, so re-assert the factory that
    # puts ``session_tag`` on every record.
    hermes_logging._install_session_record_factory()
    # The caller MUST stay unbound: a tag that leaked in from this thread
    # would make the assertions pass without the worker ever being bound.
    hermes_logging.clear_session_context()

    handler = _WorkerRecords(threading.get_ident())
    previous_level = cch.logger.level
    cch.logger.addHandler(handler)
    cch.logger.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        cch.logger.removeHandler(handler)
        cch.logger.setLevel(previous_level)


# ── Production path: OpenAI/Anthropic streaming worker ─────────────────────

def _make_agent():
    """Real agent on the chat_completions streaming path (mirrors
    tests/run_agent/test_partial_stream_finish_reason.py)."""
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    agent.session_id = SESSION_ID
    return agent


@patch("run_agent.AIAgent._close_request_openai_client")
@patch("run_agent.AIAgent._create_request_openai_client")
def test_standard_streaming_worker_record_carries_the_agent_session_tag(
    mock_create, _mock_close, worker_records
):
    """The pre-delivery failure log is emitted from the worker thread; it used
    to name no session, so in a concurrent process the one line carrying the
    cause could not be attributed to the session that suffered it."""
    mock_client = MagicMock()
    # Not a timeout/connection/parse error, so the worker takes the
    # "failed before delivery" branch instead of the retry path.
    mock_client.chat.completions.create.side_effect = RuntimeError(
        "provider returned garbage"
    )
    mock_create.return_value = mock_client

    agent = _make_agent()

    with pytest.raises(RuntimeError):
        agent._interruptible_streaming_api_call({})

    failures = worker_records.starting_with("Streaming failed before delivery")
    assert failures, (
        "expected the worker's pre-delivery failure log; got "
        f"{[str(r.msg) for r in worker_records.records]}"
    )
    assert all(r.session_tag == SESSION_TAG for r in failures)


# ── Production path: Bedrock Converse streaming worker ─────────────────────

class _BedrockAgent:
    """Minimal Bedrock-mode agent (mirrors
    tests/agent/test_bedrock_interrupt_post_worker.py)."""

    api_mode = "bedrock_converse"
    _interrupt_requested = False
    _disable_streaming = False
    reasoning_callback = None
    stream_delta_callback = None
    provider = "bedrock"
    model = "anthropic.claude-3-sonnet-20240229-v1:0"
    _consecutive_stale_streams = 0
    session_id = SESSION_ID

    def _has_stream_consumers(self):
        return False

    def _buffer_status(self, *a, **k):
        pass

    def _claim_stream_writer(self):
        return 1

    def _fire_stream_delta(self, text):
        pass

    def _fire_tool_gen_started(self, name):
        pass

    def _fire_reasoning_delta(self, text):
        pass

    def _safe_print(self, *a, **k):
        pass


def test_bedrock_streaming_worker_record_carries_the_agent_session_tag(
    worker_records,
):
    """The IAM stream-denial log lives in the Bedrock worker, which is a
    separate thread from the standard path's and needs its own binding."""
    agent = _BedrockAgent()

    def _denied(**kwargs):
        raise RuntimeError(
            "AccessDeniedException: bedrock:InvokeModelWithResponseStream"
        )

    # The denial branch falls back to non-streaming converse() and returns a
    # finished response in place of a stream; a non-empty ``choices`` is what
    # relay's completed-response predicate keys on.
    resp = SimpleNamespace(
        choices=[SimpleNamespace(index=0, message=None, finish_reason="stop")],
        usage=None,
        stop_reason="end_turn",
    )
    fake_client = SimpleNamespace(
        converse_stream=_denied,
        converse=lambda **kw: {"output": {}},
    )

    with patch("agent.bedrock_adapter._get_bedrock_runtime_client", return_value=fake_client), \
         patch("agent.bedrock_adapter.is_streaming_access_denied_error", return_value=True), \
         patch("agent.bedrock_adapter.is_stale_connection_error", return_value=False), \
         patch("agent.bedrock_adapter.invalidate_runtime_client", lambda *a, **k: None), \
         patch("agent.bedrock_adapter.normalize_converse_response", side_effect=lambda r: resp), \
         patch("agent.bedrock_adapter.stream_converse_with_callbacks", return_value=resp):
        api_kwargs = {"__bedrock_region__": "us-east-1", "__bedrock_converse__": True}
        cch.interruptible_streaming_api_call(agent, api_kwargs)

    denials = worker_records.starting_with("bedrock: converse_stream denied by IAM")
    assert denials, (
        "expected the Bedrock worker's IAM-denial log; got "
        f"{[str(r.msg) for r in worker_records.records]}"
    )
    assert all(r.session_tag == SESSION_TAG for r in denials)


# ── Helper contract ────────────────────────────────────────────────────────

class _Agent:
    """Minimal agent-like object — only ``session_id`` matters here."""

    def __init__(self, session_id=None):
        if session_id is not None:
            self.session_id = session_id


class _SessionLessAgent:
    """An agent object that doesn't expose ``session_id`` at all."""


def _new_record_session_tag():
    """``session_tag`` of a record created on the CURRENT thread."""
    record = logging.getLogger("stream-worker-test").makeRecord(
        "stream-worker-test", logging.INFO, __file__, 1, "msg", None, None
    )
    return getattr(record, "session_tag", None)


def _session_tag_in_worker(agent):
    """Bind + create one LogRecord on a fresh thread; return its tag."""
    captured = {}

    def _worker():
        _bind_worker_session_context(agent)
        captured["tag"] = _new_record_session_tag()

    hermes_logging._install_session_record_factory()
    thread = threading.Thread(target=_worker)
    thread.start()
    thread.join()
    return captured["tag"]


def test_binding_does_not_leak_into_the_calling_thread():
    hermes_logging.clear_session_context()
    _session_tag_in_worker(_Agent(SESSION_ID))
    assert _new_record_session_tag() == ""


def test_session_less_agent_leaves_the_worker_unbound():
    assert _session_tag_in_worker(_SessionLessAgent()) == ""
    assert _session_tag_in_worker(_Agent(None)) == ""


def test_binding_swallows_logging_failures(monkeypatch):
    def _boom(_session_id):
        raise RuntimeError("boom")

    monkeypatch.setattr(hermes_logging, "set_session_context", _boom)
    # Best-effort by contract: the worker must run the API call even when the
    # logging module is unusable.
    assert _session_tag_in_worker(_Agent(SESSION_ID)) == ""
