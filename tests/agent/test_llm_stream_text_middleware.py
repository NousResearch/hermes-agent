"""Tests for synchronous llm_stream_text transformation before live delivery."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent import chat_completion_helpers as helpers
from agent.codex_runtime import _consume_codex_event_stream, make_codex_app_server_event_bridge
from agent.transports.codex_app_server_session import CodexAppServerSession
from agent.stream_delivery import StreamDeliveryMixin
from hermes_cli.middleware import LLMStreamMiddlewareRefusal


class _Agent(StreamDeliveryMixin):
    session_id = "session-1"
    model = "model-1"
    provider = "openrouter"
    platform = "cli"
    show_commentary = True
    _current_turn_id = "turn-1"
    _current_api_request_id = "request-1"
    _api_call_count = 1
    _stream_callback = None
    _stream_needs_break = False
    _stream_think_scrubber = None
    _stream_context_scrubber = None
    _stream_reasoning_hooks_enabled = False

    def __init__(self):
        self.stream_delta_callback = None
        self.reasoning_callback = None
        self.interim_assistant_callback = None
        self._streamed_assistant_text_parts = []
        self._delivered_interim_texts = set()

    @staticmethod
    def _strip_think_blocks(text):
        return text

    def _stream_writer_superseded(self):
        return False


def test_text_is_transformed_before_display_and_observer(monkeypatch):
    seen_context = []
    observed = []
    delivered = []

    def transform(text, *, kind, **context):
        seen_context.append((kind, context))
        return f"restored:{text}"

    monkeypatch.setattr(
        "hermes_cli.middleware.run_llm_stream_text_middleware",
        transform,
    )
    monkeypatch.setattr(
        _Agent,
        "_enqueue_stream_hook",
        lambda self, event, **fields: observed.append((event, fields)),
    )

    agent = _Agent()
    agent.stream_delta_callback = delivered.append
    agent._fire_stream_delta("token")

    assert delivered == ["restored:token"]
    assert observed[-1] == (
        "on_stream_delta",
        {"delta": "restored:token", "kind": "text"},
    )
    kind, context = seen_context[-1]
    assert kind == "text"
    assert context["session_id"] == "session-1"
    assert context["turn_id"] == "turn-1"
    assert context["api_request_id"] == "request-1"
    assert context["provider"] == "openrouter"
    assert context["model"] == "model-1"


def test_reasoning_is_transformed_before_reasoning_callback(monkeypatch):
    delivered = []

    monkeypatch.setattr(
        "hermes_cli.middleware.run_llm_stream_text_middleware",
        lambda text, *, kind, **context: f"{kind}:{text}",
    )

    agent = _Agent()
    agent.reasoning_callback = delivered.append
    agent._fire_reasoning_delta("token")

    assert delivered == ["reasoning:token"]


def test_interim_is_transformed_before_interim_callback(monkeypatch):
    delivered = []

    monkeypatch.setattr(
        "hermes_cli.middleware.run_llm_stream_text_middleware",
        lambda text, *, kind, **context: f"{kind}:{text}",
    )

    agent = _Agent()
    agent.interim_assistant_callback = (
        lambda text, *, already_streamed=False: delivered.append(
            (text, already_streamed)
        )
    )
    agent._emit_interim_assistant_message({"content": "token"}, live=True)

    assert delivered == [("interim:token", False)]


def test_live_interim_suppression_does_not_resurrect_structured_commentary(monkeypatch):
    transformed = []
    delivered = []

    def transform(text, *, kind, **context):
        transformed.append((kind, text))
        if text in {"secret-a", "secret-b"}:
            return ""
        return text

    monkeypatch.setattr(
        "hermes_cli.middleware.run_llm_stream_text_middleware",
        transform,
    )

    agent = _Agent()
    agent.interim_assistant_callback = (
        lambda text, *, already_streamed=False: delivered.append(
            (text, already_streamed)
        )
    )
    agent._emit_interim_assistant_message(
        {
            "content": "top-level fallback must not be used",
            "codex_message_items": [
                {
                    "type": "message",
                    "phase": "commentary",
                    "content": [{"type": "output_text", "text": "secret-a"}],
                },
                {
                    "type": "message",
                    "phase": "commentary",
                    "content": [{"type": "output_text", "text": "secret-b"}],
                },
            ],
        },
        live=True,
    )

    assert transformed == [
        ("interim", "secret-a"),
        ("interim", "secret-b"),
    ]
    assert delivered == []


def test_post_response_interim_skips_live_transform(monkeypatch):
    delivered = []

    def fail_if_called(*args, **kwargs):
        raise AssertionError("live transform must not run after provider completion")

    monkeypatch.setattr(
        "hermes_cli.middleware.run_llm_stream_text_middleware",
        fail_if_called,
    )

    agent = _Agent()
    agent.interim_assistant_callback = (
        lambda text, *, already_streamed=False: delivered.append(
            (text, already_streamed)
        )
    )
    agent._emit_interim_assistant_message({"content": "already restored"})

    assert delivered == [("already restored", False)]


def test_fail_closed_transform_error_prevents_text_delivery(monkeypatch):
    delivered = []

    def fail(*args, **kwargs):
        raise RuntimeError("privacy transform failed")

    monkeypatch.setattr(
        "hermes_cli.middleware.run_llm_stream_text_middleware",
        fail,
    )

    agent = _Agent()
    agent.stream_delta_callback = delivered.append

    with pytest.raises(RuntimeError, match="privacy transform failed"):
        agent._fire_stream_delta("must-not-display")

    assert delivered == []

class _ManagedStream:
    response = None
    final_response = None

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __iter__(self):
        return iter(self._chunks)

    def close(self):
        return None


def _chat_chunk(*, content=None, tool_calls=None):
    delta = SimpleNamespace(
        content=content,
        reasoning_content=None,
        reasoning=None,
        reasoning_details=None,
        refusal=None,
        tool_calls=tool_calls,
        model_extra={},
    )
    choice = SimpleNamespace(delta=delta, finish_reason=None)
    return SimpleNamespace(
        choices=[choice], model=None, id=None, provider=None, usage=None
    )


def _tool_delta():
    return SimpleNamespace(
        index=0,
        id="call-1",
        function=SimpleNamespace(name="read_file", arguments='{"path":"x"}'),
        extra_content=None,
        model_extra={},
    )


def _suppressed_stream_call(monkeypatch, agent, chunks):
    call = helpers._StreamingCall.__new__(helpers._StreamingCall)
    call.agent = agent
    call.api_kwargs = {}
    call.result = {"response": None, "error": None, "partial_tool_names": []}
    call.clients = SimpleNamespace(diag=None, set_stream_handle=lambda stream: None)
    call._stream_stale_timeout = 1.0
    call.deltas_were_sent = {"yes": False}
    call.first_delta_fired = {"done": False}
    call.provider_tool_in_flight = {"yes": False}
    call.last_chunk_time = {"t": 0.0}
    call._stream_timeouts = lambda: (1.0, 1.0, 1.0)
    call._new_diag = lambda: {}
    call._set_managed_stream = lambda stream: stream
    call._count_chunk = lambda diag, chunk: None
    call._stream_attempt_is_active = lambda stream_attempt_id: True
    call._stream_attempt_was_cancelled = lambda stream_attempt_id: False
    call._close_managed_stream = lambda: None
    call._emit_tool_started = lambda name: None
    call._emit_reasoning = lambda text: None
    call._finish_chat_stream = lambda *args, **kwargs: "done"
    monkeypatch.setattr(helpers, "_relay_stream_identity", lambda *args, **kwargs: {})
    monkeypatch.setattr(helpers, "_relay_stream_metadata", lambda *args, **kwargs: {})
    from agent import relay_llm
    monkeypatch.setattr(relay_llm, "stream", lambda *args, **kwargs: _ManagedStream(chunks))
    return call


def test_tool_delta_then_content_is_transformed_at_suppressed_sink(monkeypatch):
    delivered = []
    transformed = []

    def transform(text, *, kind, **context):
        transformed.append((kind, text))
        return text.replace("secret", "SAFE")

    monkeypatch.setattr(
        "hermes_cli.middleware.run_llm_stream_text_middleware",
        transform,
    )

    agent = _Agent()
    agent.api_mode = "chat_completions"
    agent.base_url = ""
    agent._interrupt_requested = False
    agent.stream_delta_callback = delivered.append
    call = _suppressed_stream_call(
        monkeypatch,
        agent,
        [
            _chat_chunk(tool_calls=[_tool_delta()]),
            _chat_chunk(content="<thi"),
            _chat_chunk(content="nk>secret</think>"),
        ],
    )

    assert call._call_chat_completions(1) == "done"
    assert transformed == [("text", "<thi"), ("text", "nk>secret</think>")]
    assert delivered == ["<thi", "nk>SAFE</think>"]
    assert agent._current_streamed_assistant_text == "<think>SAFE</think>"


def test_tool_delta_then_content_closed_refusal_never_reaches_sink(monkeypatch):
    delivered = []

    def refuse(*args, **kwargs):
        raise LLMStreamMiddlewareRefusal(ConnectionError("policy refused"))

    monkeypatch.setattr(
        "hermes_cli.middleware.run_llm_stream_text_middleware",
        refuse,
    )

    agent = _Agent()
    agent.api_mode = "chat_completions"
    agent.base_url = ""
    agent._interrupt_requested = False
    agent.stream_delta_callback = delivered.append
    call = _suppressed_stream_call(
        monkeypatch,
        agent,
        [_chat_chunk(tool_calls=[_tool_delta()]), _chat_chunk(content="secret")],
    )

    with pytest.raises(LLMStreamMiddlewareRefusal, match="policy refused"):
        call._call_chat_completions(1)

    assert delivered == []
    assert agent._current_streamed_assistant_text == ""


@pytest.mark.parametrize("already_delivered", [False, True])
def test_stream_owner_never_converts_closed_refusal_to_partial(monkeypatch, already_delivered):
    refusal = LLMStreamMiddlewareRefusal(ConnectionError("closed boundary"))
    call = helpers._StreamingCall.__new__(helpers._StreamingCall)
    call.agent = SimpleNamespace(_interrupt_requested=False)
    call.result = {"response": None, "error": refusal}
    call.deltas_were_sent = {"yes": already_delivered}
    call.clients = SimpleNamespace(diag={})
    call._resolve_stale_timeout = lambda: None
    call._run_call = lambda: None
    call._monitor_loop = lambda: None
    monkeypatch.setattr(helpers, "should_use_direct_api_call", lambda agent: True)

    with pytest.raises(LLMStreamMiddlewareRefusal, match="closed boundary"):
        call.run()


def test_stream_error_handler_never_retries_transport_shaped_refusal():
    refusal = LLMStreamMiddlewareRefusal(ConnectionError("looks transient but is policy"))
    call = helpers._StreamingCall.__new__(helpers._StreamingCall)
    call.agent = SimpleNamespace()
    call.result = {"response": None, "error": None}
    call._request_cancelled = {"value": False}
    call.deltas_were_sent = {"yes": True}

    assert call._handle_stream_error(refusal, attempt=0, max_retries=2) is False
    assert call.result["error"] is refusal


def test_codex_live_interim_closed_refusal_escapes_guard(monkeypatch):
    def refuse(*args, **kwargs):
        raise LLMStreamMiddlewareRefusal(RuntimeError("codex privacy refusal"))

    monkeypatch.setattr(
        "hermes_cli.middleware.run_llm_stream_text_middleware",
        refuse,
    )

    agent = _Agent()
    agent.interim_assistant_callback = lambda text, *, already_streamed=False: None
    bridge = make_codex_app_server_event_bridge(agent)

    with pytest.raises(LLMStreamMiddlewareRefusal, match="codex privacy refusal"):
        bridge({
            "method": "item/completed",
            "params": {"item": {"type": "agentMessage", "id": "m1", "text": "secret"}},
        })

def test_codex_app_server_transport_does_not_swallow_closed_refusal():
    refusal = LLMStreamMiddlewareRefusal(RuntimeError("session privacy refusal"))
    session = CodexAppServerSession.__new__(CodexAppServerSession)

    def on_event(note):
        raise refusal

    session._on_event = on_event

    with pytest.raises(LLMStreamMiddlewareRefusal, match="session privacy refusal"):
        session._absorb_notification(SimpleNamespace(), None, {"method": "item/completed"})


def test_codex_responses_event_owner_does_not_swallow_closed_refusal():
    refusal = LLMStreamMiddlewareRefusal(RuntimeError("responses privacy refusal"))

    def on_event(event):
        raise refusal

    with pytest.raises(LLMStreamMiddlewareRefusal, match="responses privacy refusal"):
        _consume_codex_event_stream([{}], model="test-model", on_event=on_event)