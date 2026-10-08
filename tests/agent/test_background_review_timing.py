"""Lifecycle regressions for auxiliary.background_review.timing (#83193)."""

from __future__ import annotations

from agent.app_server_turn_finalization import finish_app_server_turn

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent import background_review
from agent.turn_api_call import _should_stream
from agent.turn_finalizer import finalize_turn
from run_agent import AIAgent


def _make_agent() -> AIAgent:
    return AIAgent(
        model="openai/gpt-4o-mini",
        provider="openrouter",
        api_key="sk-dummy",
        base_url="https://openrouter.ai/api/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )


def _stub_for_finalize(agent: AIAgent, events: list[str]) -> None:
    agent._spawn_background_review = MagicMock()
    agent._run_background_review_before_final = MagicMock(
        side_effect=lambda **_kwargs: events.append("review")
    )
    agent._save_trajectory = MagicMock()
    agent._cleanup_task_resources = MagicMock()
    agent._persist_session = MagicMock()
    agent._session_messages = []
    agent._file_mutation_verifier_enabled = lambda: False
    agent.clear_interrupt = MagicMock()
    agent._stream_callback = lambda _text: None
    agent._sync_external_memory_for_turn = MagicMock()
    agent._skill_nudge_interval = 0
    agent._iters_since_skill = 0
    agent.valid_tool_names = {"memory"}
    agent.iteration_budget = MagicMock(remaining=100, used=5, max_total=100)
    agent.max_iterations = 50
    agent._emit_status = MagicMock()
    agent._safe_print = lambda text: events.append(f"print:{text}")
    agent._apply_persist_user_message_override = MagicMock()
    agent.context_compressor = None
    agent._turn_preflight_display_snapshot = None
    agent._turn_received_provider_response = False
    agent._turn_failed_file_mutations = {}
    agent._db_flush_scan_prefix = None
    agent._background_review_turn_settings = {
        "enabled": True,
        "timing": "before_final",
        "task_cfg": {"timing": "before_final"},
    }
    agent._deferred_completion_banner = "completed"


def _finalize(agent: AIAgent):
    return finalize_turn(
        agent,
        final_response="ok",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=[{"role": "assistant", "content": "ok"}],
        conversation_history=[],
        effective_task_id="test",
        turn_id="test-turn",
        user_message="test",
        original_user_message="test",
        _should_review_memory=True,
        _turn_exit_reason="text_response(finish_reason=stop)",
    )


def test_timing_defaults_and_accepts_strict_mode(caplog):
    assert background_review.background_review_timing({}) == "background"
    assert background_review.background_review_timing({"timing": "before_final"}) == "before_final"

    with caplog.at_level("WARNING", logger="agent.background_review"):
        assert background_review.background_review_timing({"timing": "after_delivery"}) == "background"
    assert "Invalid auxiliary.background_review.timing" in caplog.text


def test_before_final_forces_non_streaming_even_with_consumers():
    agent = _make_agent()
    agent.stream_delta_callback = lambda _text: None
    agent._background_review_turn_settings = {"timing": "before_final"}

    assert _should_stream(agent) is False


def test_background_mode_preserves_streaming_default():
    agent = _make_agent()
    agent.stream_delta_callback = lambda _text: None
    agent._background_review_turn_settings = {"timing": "background"}

    assert _should_stream(agent) is True


def test_disabled_review_does_not_suppress_streaming():
    agent = _make_agent()
    agent.stream_delta_callback = lambda _text: None
    agent._background_review_turn_settings = {
        "enabled": False,
        "timing": "before_final",
    }

    assert _should_stream(agent) is True


def test_inline_review_restores_foreground_approval_callback(monkeypatch):
    from tools.terminal_tool import _get_approval_callback, set_approval_callback

    def sentinel(*_args, **_kwargs):
        return "once"

    callbacks_seen_during_review = []

    def fake_run_review_fork(*args, **_kwargs):
        callbacks_seen_during_review.append(_get_approval_callback())
        state = args[5]
        state.review_messages = []

    monkeypatch.setattr(background_review, "_parent_can_emit_tool_calls", lambda _agent: True)
    monkeypatch.setattr(background_review, "_run_review_fork", fake_run_review_fork)
    monkeypatch.setattr(background_review, "summarize_background_review_actions", lambda *_a, **_k: [])

    set_approval_callback(sentinel)
    try:
        background_review._run_review_in_thread(SimpleNamespace(), [], "review")
        assert callbacks_seen_during_review == [background_review._bg_review_auto_deny]
        assert _get_approval_callback() is sentinel
    finally:
        set_approval_callback(None)


def test_before_final_runs_inline_without_creating_thread(monkeypatch):
    agent = _make_agent()
    calls = []
    run = background_review._BackgroundReviewRun()

    monkeypatch.setattr(background_review, "prepare_background_review_run", lambda parent: run)

    def fake_spawn(parent, snapshot, **kwargs):
        calls.append(("spawn", threading.get_ident(), snapshot, kwargs))
        return lambda: calls.append(("target", threading.get_ident())), "prompt"

    monkeypatch.setattr(background_review, "spawn_background_review_thread", fake_spawn)
    monkeypatch.setattr(
        "run_agent.threading.Thread",
        lambda **_kwargs: pytest.fail("before_final must not create a daemon thread"),
    )

    caller_thread = threading.get_ident()
    agent._run_background_review_before_final(
        [{"role": "user", "content": {"nested": ["value"]}}],
        review_memory=True,
        task_cfg={"timing": "before_final"},
    )

    assert calls[0][0:2] == ("spawn", caller_thread)
    assert calls[1] == ("target", caller_thread)
    assert calls[0][3]["review_run"] is run


def test_before_final_review_and_summary_boundary_precede_terminal_banner():
    agent = _make_agent()
    events: list[str] = []
    _stub_for_finalize(agent, events)

    result = _finalize(agent)

    assert result["final_response"] == "ok"
    assert events == ["review", "print:completed"]
    agent._spawn_background_review.assert_not_called()
    agent._run_background_review_before_final.assert_called_once()
    assert agent._stream_callback is None
    assert agent._background_review_turn_settings is None


def test_before_final_review_failure_is_fail_open_and_still_reaches_terminal_boundary():
    agent = _make_agent()
    events: list[str] = []
    _stub_for_finalize(agent, events)
    agent._run_background_review_before_final.side_effect = RuntimeError("review failed")

    result = _finalize(agent)

    assert result["final_response"] == "ok"
    assert events == ["print:completed"]
    assert agent._stream_callback is None
    assert agent._background_review_turn_settings is None


def test_strict_app_server_bridge_withholds_assistant_text_but_keeps_reasoning_progress():
    from agent.codex_runtime import make_codex_app_server_event_bridge

    agent = SimpleNamespace(
        _background_review_turn_settings={"enabled": True, "timing": "before_final"},
        _fire_stream_delta=MagicMock(),
        _fire_reasoning_delta=MagicMock(),
        _emit_interim_assistant_message=MagicMock(),
        _touch_activity=MagicMock(),
        _current_streamed_assistant_text="partial",
        show_commentary=True,
    )
    bridge = make_codex_app_server_event_bridge(agent)

    bridge({"method": "item/agentMessage/delta", "params": {"delta": "final answer"}})
    bridge({"method": "item/reasoning/delta", "params": {"delta": "working"}})
    bridge({
        "method": "item/completed",
        "params": {"item": {"type": "agentMessage", "text": "final answer"}},
    })

    agent._fire_stream_delta.assert_not_called()
    agent._emit_interim_assistant_message.assert_not_called()
    agent._fire_reasoning_delta.assert_called_once_with("working")
    assert agent._current_streamed_assistant_text == ""


def test_app_server_strict_mode_runs_review_inline(monkeypatch):
    from agent import codex_runtime

    events: list[str] = []
    agent = SimpleNamespace(
        _iters_since_skill=0,
        _skill_nudge_interval=0,
        valid_tool_names=set(),
        skip_background_review=False,
        _background_review_turn_settings={
            "enabled": True,
            "timing": "before_final",
            "task_cfg": {"timing": "before_final"},
        },
        _sync_external_memory_for_turn=MagicMock(),
        _run_background_review_before_final=MagicMock(
            side_effect=lambda **_kwargs: events.append("review")
        ),
        _spawn_background_review=MagicMock(),
    )
    turn = SimpleNamespace(
        tool_iterations=0,
        interrupted=False,
        error=None,
        final_text="done",
    )
    monkeypatch.setattr(codex_runtime, "_record_codex_app_server_compaction", lambda *_args: None)
    monkeypatch.setattr(codex_runtime, "_record_codex_app_server_usage", lambda *_args, **_kwargs: {})

    result = finish_app_server_turn(
        agent,
        turn,
        [{"role": "assistant", "content": "done"}],
        original_user_message="question",
        should_review_memory=True,
    )

    assert result == {}
    assert events == ["review"]
    agent._run_background_review_before_final.assert_called_once()
    call = agent._run_background_review_before_final.call_args.kwargs
    assert call["task_cfg"] == {"timing": "before_final"}
    agent._spawn_background_review.assert_not_called()


def test_full_turn_snapshots_strict_timing_and_returns_only_after_review(monkeypatch):
    from tests.agent.test_run_agent import _mock_response

    agent = _make_agent()
    agent.stream_delta_callback = lambda _text: None
    agent._memory_nudge_interval = 1
    agent._turns_since_memory = 0
    agent._memory_store = object()
    agent.valid_tool_names.add("memory")
    agent._interruptible_api_call = MagicMock(
        return_value=_mock_response(content="answer", finish_reason="stop")
    )
    agent._interruptible_streaming_api_call = MagicMock(
        side_effect=AssertionError("strict timing must not stream terminal text")
    )
    events: list[str] = []
    agent._run_background_review_before_final = MagicMock(
        side_effect=lambda **_kwargs: events.append("review")
    )
    agent._spawn_background_review = MagicMock()

    monkeypatch.setattr(
        background_review,
        "load_background_review_settings",
        lambda: (True, {"timing": "before_final"}),
    )

    result = agent.run_conversation("question")
    events.append("returned")

    assert result["final_response"] == "answer"
    assert events == ["review", "returned"]
    agent._interruptible_api_call.assert_called_once()
    agent._interruptible_streaming_api_call.assert_not_called()
    agent._spawn_background_review.assert_not_called()
    assert agent._background_review_turn_settings is None


@pytest.mark.parametrize("prior", ["idle", "pending", "finishes-on-cancel"])
def test_inline_review_reports_contention_and_retries_after_ack(monkeypatch, caplog, prior):
    agent = _make_agent()
    previous = None if prior == "idle" else background_review.prepare_background_review_run(agent)
    reviewed = []
    if prior == "finishes-on-cancel":
        class PreviousFork:
            def hard_interrupt(self, message=None, *, tool_reason=None):
                background_review.finish_background_review_run(agent, previous)
        assert previous.begin_request(PreviousFork())
    monkeypatch.setattr(background_review, "_BACKGROUND_REVIEW_CANCEL_TIMEOUT_SECONDS", 0 if prior == "pending" else 10)

    def spawn(parent, snapshot, **kwargs):
        def target():
            reviewed.append(kwargs["review_run"])
            background_review.finish_background_review_run(parent, kwargs["review_run"])
        return target, "review"
    monkeypatch.setattr(background_review, "spawn_background_review_thread", spawn)
    try:
        with caplog.at_level("WARNING"):
            result = agent._run_background_review_before_final(
                [{"role": "assistant", "content": "answer"}], review_memory=True)
        expected = "skipped" if prior == "pending" else "ran"
        assert isinstance(result, dict) and result["status"] == expected
        if prior == "pending":
            assert result["reason"] == "previous_review_still_running"
            assert "before-final review skipped" in caplog.text.lower()
            assert reviewed == []
            assert agent._background_review_run is previous
            assert previous.cancel_requested.is_set()
        else:
            assert len(reviewed) == 1 and reviewed[0] is not previous
            assert agent._background_review_run is None
    finally:
        background_review.finish_background_review_run(agent, previous)


@pytest.mark.parametrize("runtime", ["generic", "app-server"])
def test_busy_inline_review_disposition_reaches_the_turn_result(monkeypatch, runtime):
    agent = _make_agent()
    events = []
    _stub_for_finalize(agent, events)
    agent._run_background_review_before_final = AIAgent._run_background_review_before_final.__get__(agent, AIAgent)
    previous = background_review.prepare_background_review_run(agent)
    monkeypatch.setattr(background_review, "_BACKGROUND_REVIEW_CANCEL_TIMEOUT_SECONDS", 0)
    try:
        if runtime == "generic":
            result = _finalize(agent)
            assert events == ["print:completed"]
        else:
            from agent import codex_runtime
            turn = SimpleNamespace(final_text="ok", interrupted=False, tool_iterations=0, input_tokens=0,
                output_tokens=0, cached_tokens=0, cache_creation_tokens=0, api_calls=1,
                error=None, usage=None)
            monkeypatch.setattr(codex_runtime, "_record_codex_app_server_usage", lambda *a, **kw: {})
            monkeypatch.setattr(codex_runtime, "_record_codex_app_server_compaction", lambda *a: None)
            monkeypatch.setattr(codex_runtime, "_store_codex_thread_id", lambda *a: None)
            result = finish_app_server_turn(agent, turn, [],
                original_user_message="test", should_review_memory=True)
        assert result["background_review"]["status"] == "skipped"
        assert result["background_review"]["reason"] == "previous_review_still_running"
        agent._spawn_background_review.assert_not_called()
    finally:
        background_review.finish_background_review_run(agent, previous)
