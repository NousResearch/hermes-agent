"""Prepared review requests must fit the detached review model's threshold."""
from unittest.mock import patch
from contextlib import contextmanager
from functools import wraps

import pytest

from tests.run_agent.test_background_review_input_budget import (
    _make_loop_agent, _run_with_responses, _final_response, _tool_response,
)


@contextmanager
def _pressure(*values):
    from agent.turn_request_assembly import assemble_api_request
    values = iter(values)

    @wraps(assemble_api_request)
    def assemble(*args, **kwargs):
        request = assemble_api_request(*args, **kwargs)
        request.request_pressure_tokens = next(values)
        return request

    with patch("agent.conversation_loop.assemble_api_request", new=assemble):
        yield


@pytest.fixture(autouse=True)
def _restore_review_origin():
    # These loops run inline instead of on the production disposable thread.
    from tools.skill_provenance import set_current_write_origin, reset_current_write_origin
    token = set_current_write_origin("foreground")
    try:
        yield
    finally:
        reset_current_write_origin(token)


@pytest.mark.parametrize("pressure", [1000, 1001, None, -1, float("nan"), True, "100"])
def test_initial_review_rejects_oversized_or_unavailable_pressure(pressure):
    agent = _make_loop_agent()
    agent._memory_write_origin = "background_review"
    agent._review_input_token_budget = None
    agent._review_defer_compaction_before_first_response = True
    agent.context_compressor.threshold_tokens = 1000
    agent.compression_enabled = True
    with _pressure(pressure), patch.object(agent, "_compress_context") as compress:
        result = _run_with_responses(agent, [_final_response()])
    agent.client.chat.completions.create.assert_not_called()
    compress.assert_not_called()
    assert result["completed"] is False
    assert result["api_calls"] == 0
    assert result["turn_exit_reason"] in {"review_request_oversized", "review_request_size_unavailable"}


@pytest.mark.parametrize(("origin", "threshold"), [
    (None, 1000), (None, None), ("side_question", 1000), ("side_question", None),
    ("background_review", 1000),
])
def test_under_limit_and_nonreview_requests_proceed(origin, threshold):
    agent = _make_loop_agent()
    agent._memory_write_origin = origin
    agent._review_input_token_budget = 600000
    agent.context_compressor.threshold_tokens = threshold
    with _pressure(999 if origin == "background_review" else 1001):
        result = _run_with_responses(agent, [_final_response()])
    assert result["completed"] is True
    assert agent.client.chat.completions.create.call_count == 1


@pytest.mark.parametrize("threshold", [None, 0, -1, float("inf"), "bad", True])
def test_review_missing_threshold_skips(threshold):
    agent = _make_loop_agent()
    agent._memory_write_origin = "background_review"
    agent._review_input_token_budget = None
    agent.context_compressor.threshold_tokens = threshold
    with _pressure(100):
        result = _run_with_responses(agent, [_final_response()])
    agent.client.chat.completions.create.assert_not_called()
    assert result["turn_exit_reason"] == "review_request_size_unavailable"


def test_later_oversized_prepared_request_never_reaches_provider():
    agent = _make_loop_agent()
    agent._memory_write_origin = "background_review"
    agent._review_input_token_budget = None
    agent.context_compressor.threshold_tokens = 1000
    with _pressure(100, 1001):
        result = _run_with_responses(agent, [_tool_response(100), _final_response()])
    assert agent.client.chat.completions.create.call_count == 1
    assert result["completed"] is False
    assert result["turn_exit_reason"] == "review_request_oversized"


@pytest.mark.parametrize("reason", ["review_request_oversized", "review_request_size_unavailable"])
def test_skipped_review_reports_skip_and_releases_ownership(reason):
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from agent import background_review as review
    parent = SimpleNamespace(
        client=None, _background_review_agent=None, _active_children=[],
        memory_notifications="on", background_review_callback=None,
    )
    fork = MagicMock()
    fork._session_messages = []
    fork.run_conversation.return_value = {"completed": False, "turn_exit_reason": reason}
    run = MagicMock()
    run.cancel_requested.is_set.return_value = False
    run.begin_request.return_value = True
    with patch.object(review, "build_cache_parity_fork", return_value=(fork, {}, False)), \
         patch.object(review, "_review_tool_whitelist", return_value=(set(), set())), \
         patch.object(review, "_snapshot_review_usage", return_value={}), \
         patch.object(review, "_record_review_usage_to_parent"), \
         patch.object(review, "finish_background_review_run") as finish, \
         patch.object(review, "_log_review_completion") as completion, \
         patch.object(review, "_set_thread_approval_callback") as approval:
        review._run_review_in_thread(parent, [], "review", {}, run)
    completion.assert_called_once_with({}, "skipped")
    assert parent._background_review_agent is None
    assert parent._active_children == []
    finish.assert_called_with(parent, run)
    fork.release_clients.assert_called_once()
    approval.assert_called_with(None)


@pytest.mark.parametrize("rebuilt_pressure", [900, 1001])
def test_later_detached_compaction_rebuild_is_checked(rebuilt_pressure):
    agent = _make_loop_agent()
    agent._memory_write_origin = "background_review"
    agent._review_input_token_budget = None
    agent._review_defer_compaction_before_first_response = True
    agent.compression_enabled = True
    agent.context_compressor.threshold_tokens = 1000
    agent.context_compressor.should_compress.side_effect = lambda tokens: tokens >= 1000
    agent.compression_in_place = True

    def compress(messages, system, **kwargs):
        return list(messages), system

    with _pressure(100, 1500, rebuilt_pressure), \
         patch.object(agent, "_compress_context", side_effect=compress) as compression:
        result = _run_with_responses(agent, [_tool_response(100), _final_response()])
    compression.assert_called_once()
    assert agent.client.chat.completions.create.call_count == (2 if rebuilt_pressure < 1000 else 1)
    assert result["completed"] is (rebuilt_pressure < 1000)


def test_routed_fork_uses_own_window_not_parent_capacity():
    from agent.background_review import build_cache_parity_fork
    parent = _make_loop_agent()
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
        "provider": "openai", "model": "review-small", "api_key": "test-key",
        "api_mode": "chat_completions", "base_url": "https://example.com/v1",
    }), patch("agent.process_bootstrap.OpenAI"), \
         patch("model_tools.get_tool_definitions", return_value=[]), \
         patch("model_tools.check_toolset_requirements", return_value={}), \
         patch("agent.model_metadata.get_model_context_length", return_value=128000), \
         patch("agent.context_compressor.get_model_context_length", return_value=128000):
        fork, _, routed = build_cache_parity_fork(parent, {"provider": "openai", "model": "review-small"}, max_iterations=3)
    assert routed
    assert fork.context_compressor is not parent.context_compressor
    assert fork.context_compressor.context_length == 128000
    assert fork.context_compressor.threshold_tokens < parent.context_compressor.threshold_tokens
    fork.client = parent.client
    fork._cached_system_prompt = "system"
    fork._disable_streaming = True
    with _pressure(fork.context_compressor.threshold_tokens):
        result = _run_with_responses(fork, [_final_response()])
    fork.client.chat.completions.create.assert_not_called()
    assert result["turn_exit_reason"] == "review_request_oversized"
    assert parent.context_compressor.threshold_tokens == 999_999_999
    fork.release_clients()


def test_standalone_curator_is_not_a_snapshot_review_fork():
    agent = _make_loop_agent()
    agent._memory_write_origin = "background_review"  # curator's skill-write protection
    assert not hasattr(agent, "_review_input_token_budget")
    agent.context_compressor.threshold_tokens = 1000
    with _pressure(1001):
        result = _run_with_responses(agent, [_final_response()])
    assert agent.client.chat.completions.create.call_count == 1
    assert result["completed"] is True
