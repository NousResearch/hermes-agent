"""``agent.status_output.notify_turn_status`` — a plugin's line on the status rail of the live turn.

A provider-supplied client (``ProviderProfile.create_client``) or a plugin that does work
inside a provider call — waiting out a rate limit, switching credentials — has no agent to
speak through. These tests pin the seam: outside a call it is a no-op, inside
``perform_api_call`` it reaches the turn's ``_emit_status_kind`` (CLI line + status_callback),
it follows the call into its worker threads, and it is unbound again when the call ends.
"""

from __future__ import annotations

import threading
import types

import pytest

from agent import status_output
from agent.status_output import notify_turn_status


def test_outside_a_call_it_is_a_no_op():
    assert notify_turn_status("waiting") is False


def test_empty_or_non_text_messages_are_ignored():
    token = status_output._TURN_STATUS_SINK.set(lambda kind, message: pytest.fail("must not be called"))
    try:
        assert notify_turn_status("   ") is False
        assert notify_turn_status(None) is False  # type: ignore[arg-type]
    finally:
        status_output._TURN_STATUS_SINK.reset(token)


def test_message_is_one_bounded_line_and_kind_is_restricted():
    seen = []
    token = status_output._TURN_STATUS_SINK.set(lambda kind, message: seen.append((kind, message)))
    try:
        assert notify_turn_status("a\nb   c") is True
        assert notify_turn_status("x" * 1000, kind="warn") is True
        assert notify_turn_status("y", kind="error") is True
    finally:
        status_output._TURN_STATUS_SINK.reset(token)
    assert seen[0] == ("lifecycle", "a b c")
    assert seen[1][0] == "warn" and len(seen[1][1]) == status_output.TURN_STATUS_MAX_CHARS
    assert seen[2][0] == "lifecycle"


def test_a_sink_that_declines_is_reported():
    token = status_output._TURN_STATUS_SINK.set(lambda kind, message: False)
    try:
        assert notify_turn_status("x", kind="activity") is False
    finally:
        status_output._TURN_STATUS_SINK.reset(token)


def test_a_failing_sink_never_raises():
    def boom(kind, message):
        raise RuntimeError("display gone")

    token = status_output._TURN_STATUS_SINK.set(boom)
    try:
        assert notify_turn_status("waiting") is False
    finally:
        status_output._TURN_STATUS_SINK.reset(token)


def test_worker_threads_started_for_the_call_inherit_the_rail():
    from agent.chat_completion_helpers import _context_thread_target

    seen = []
    token = status_output._TURN_STATUS_SINK.set(lambda kind, message: seen.append(message))
    try:
        results = []
        worker = threading.Thread(target=_context_thread_target(lambda: results.append(notify_turn_status("from worker"))))
        worker.start()
        worker.join()
    finally:
        status_output._TURN_STATUS_SINK.reset(token)
    assert results == [True] and seen == ["from worker"]


def test_perform_api_call_binds_the_rail_for_the_call_only(monkeypatch):
    import hermes_cli.middleware as middleware
    from agent import turn_api_call

    emitted = []
    inside = []

    def fake_middleware(api_kwargs, next_call, **context):
        inside.append(notify_turn_status("KAME: waiting for a key"))
        return types.SimpleNamespace(choices=[])

    monkeypatch.setattr(middleware, "run_llm_execution_middleware", fake_middleware)
    monkeypatch.setattr(turn_api_call, "_should_stream", lambda agent: False)
    agent = types.SimpleNamespace(
        api_mode="chat_completions", session_id="s", platform="cli", model="m", provider="p",
        base_url="https://p.invalid", _pending_redirect=False, _model_request_active=None,
        _pending_redirect_lock=None, _has_pending_redirect=lambda: False,
        _emit_status_kind=lambda kind, message, *, origin: emitted.append((kind, message, origin)),
    )
    verdict = turn_api_call.perform_api_call(
        agent, api_kwargs={}, _original_api_kwargs={}, _llm_middleware_trace=[],
        _moa_prepared_request=None, _retry=None, thinking_spinner=None, retry_count=0,
        api_call_count=1, api_request_id="r", effective_task_id="t", turn_id="u", interrupted=False,
    )
    assert verdict.action == "fallthrough"
    assert inside == [True]
    assert emitted == [("lifecycle", "KAME: waiting for a key", "notify_turn_status")]
    assert notify_turn_status("after the call") is False


def test_activity_goes_to_the_thinking_line(monkeypatch):
    import hermes_cli.middleware as middleware
    from agent import turn_api_call

    thinking = []
    results = []

    def fake_middleware(api_kwargs, next_call, **context):
        results.append(notify_turn_status("⏳ waiting on gemini - KAME 13/15 keys healthy", kind="activity"))
        return types.SimpleNamespace(choices=[])

    monkeypatch.setattr(middleware, "run_llm_execution_middleware", fake_middleware)
    monkeypatch.setattr(turn_api_call, "_should_stream", lambda agent: False)
    agent = types.SimpleNamespace(
        api_mode="chat_completions", session_id="s", platform="cli", model="m", provider="p",
        base_url="https://p.invalid", _pending_redirect=False, _model_request_active=None,
        _pending_redirect_lock=None, _has_pending_redirect=lambda: False, thinking_callback=thinking.append,
        _emit_status_kind=lambda *a, **k: pytest.fail("activity must not become a status line"),
    )
    turn_api_call.perform_api_call(
        agent, api_kwargs={}, _original_api_kwargs={}, _llm_middleware_trace=[],
        _moa_prepared_request=None, _retry=None, thinking_spinner=None, retry_count=0,
        api_call_count=1, api_request_id="r", effective_task_id="t", turn_id="u", interrupted=False,
    )
    assert results == [True]
    assert thinking == ["⏳ waiting on gemini - KAME 13/15 keys healthy"]


def test_activity_uses_the_cores_wait_notice_when_the_agent_has_one(monkeypatch):
    import hermes_cli.middleware as middleware
    from agent import turn_api_call

    notices = []

    def fake_middleware(api_kwargs, next_call, **context):
        notify_turn_status("⏳ waiting on gemini - KAME 0/15 keys healthy", kind="activity")
        return types.SimpleNamespace(choices=[])

    monkeypatch.setattr(middleware, "run_llm_execution_middleware", fake_middleware)
    monkeypatch.setattr(turn_api_call, "_should_stream", lambda agent: False)
    agent = types.SimpleNamespace(
        api_mode="chat_completions", session_id="s", platform="cli", model="m", provider="p",
        base_url="https://p.invalid", _pending_redirect=False, _model_request_active=None,
        _pending_redirect_lock=None, _has_pending_redirect=lambda: False,
        _emit_wait_notice=notices.append,
        thinking_callback=lambda text: pytest.fail("the wait notice already drew the line"),
        _emit_status_kind=lambda *a, **k: pytest.fail("activity must not become a status line"),
    )
    turn_api_call.perform_api_call(
        agent, api_kwargs={}, _original_api_kwargs={}, _llm_middleware_trace=[],
        _moa_prepared_request=None, _retry=None, thinking_spinner=None, retry_count=0,
        api_call_count=1, api_request_id="r", effective_task_id="t", turn_id="u", interrupted=False,
    )
    assert notices == ["⏳ waiting on gemini - KAME 0/15 keys healthy"]
