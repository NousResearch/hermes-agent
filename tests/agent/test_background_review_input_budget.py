"""Regression tests for the background-review aggregate input budget (#93057).

The review fork replays its snapshot on every provider request in its tool
loop. Detached in-memory compaction bounds any SINGLE request; the aggregate
input budget (``_review_input_token_budget``, set by
``_run_review_in_thread`` from ``auxiliary.background_review.max_input_tokens``)
bounds the WHOLE review: the tool loop stops before the provider call that
would cross it, mirroring the iteration-budget exit.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _tool_call() -> SimpleNamespace:
    return SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(name="web_search", arguments='{"query": "x"}'),
    )


def _tool_response(prompt_tokens: int) -> SimpleNamespace:
    message = SimpleNamespace(
        content=None,
        reasoning_content=None,
        reasoning=None,
        tool_calls=[_tool_call()],
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
        model="test/model",
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=1,
            total_tokens=prompt_tokens + 1,
        ),
    )


def _final_response() -> SimpleNamespace:
    message = SimpleNamespace(
        content="done",
        reasoning_content=None,
        reasoning=None,
        tool_calls=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        model="test/model",
        usage=None,
    )


def _tool_definition() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }


@contextmanager
def _agent_init_seams(context_window: int = 256_000):
    """The provider, tool-registry and model-catalog seams ``AIAgent.__init__`` reaches."""
    with (
        patch("model_tools.get_tool_definitions", return_value=[_tool_definition()]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.model_metadata.get_model_context_length", return_value=context_window),
        patch("agent.context_compressor.get_model_context_length", return_value=context_window),
    ):
        yield


def _make_loop_agent():
    with _agent_init_seams():
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            max_iterations=10,
        )
    _wire_loop_doubles(agent)
    return agent


def _wire_loop_doubles(agent):
    """A scripted client, a compressor that never fires and an inline tool executor."""
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent._disable_streaming = True
    agent.tool_delay = 0
    agent.save_trajectories = False
    agent.max_compression_attempts = 1

    compressor = MagicMock()
    compressor.protect_first_n = 3
    compressor.protect_last_n = 20
    compressor.threshold_tokens = 999_999_999  # never fire compaction here
    compressor.context_length = 1_000_000_000
    compressor.last_prompt_tokens = -1
    compressor._verify_compaction_cleared_threshold = False
    compressor.awaiting_real_usage_after_compression = False
    compressor.should_compress.return_value = False
    compressor.should_compress_info.return_value = (False, None)
    compressor.should_compress_preflight.return_value = False
    compressor.should_defer_preflight_to_real_usage.return_value = False
    compressor.get_active_compression_failure_cooldown.return_value = None
    compressor.select_context.return_value = None
    compressor.get_automatic_compaction_status_message.return_value = ""
    agent.compression_enabled = False  # isolate the budget behavior under test
    agent.context_compressor = compressor

    def _fake_execute_tool_calls(assistant_message, messages, *_args):
        tool_call = assistant_message.tool_calls[0]
        messages.append(
            {
                "role": "tool",
                "name": tool_call.function.name,
                "tool_call_id": tool_call.id,
                "content": "ok",
            }
        )

    agent._execute_tool_calls = _fake_execute_tool_calls


def _run_with_responses(agent, responses, *, user_message="do some tool work"):
    agent.client.chat.completions.create.side_effect = responses
    with (
        patch.object(agent, "_flush_messages_to_session_db", return_value=True),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(user_message)
    return result


def test_review_input_budget_stops_tool_loop_before_next_provider_call():
    """The projected next request stops the loop before aggregate input can cross."""
    agent = _make_loop_agent()
    agent._review_input_token_budget = 100_000

    responses = [
        _tool_response(50_000),
        _tool_response(50_000),
        _tool_response(50_000),  # projected aggregate would cross the ceiling
        _tool_response(50_000),
        _final_response(),
    ]
    result = _run_with_responses(agent, responses)

    create = agent.client.chat.completions.create
    assert create.call_count == 1, (
        f"expected the loop to stop before crossing the input budget, "
        f"but {create.call_count} provider calls were made (budget "
        f"{agent._review_input_token_budget}, "
        f"used {agent.session_input_tokens})"
    )
    assert agent.session_input_tokens == 50_000
    assert result["completed"] is False


def test_review_input_budget_preflight_blocks_request_that_would_cross_ceiling():
    agent = _make_loop_agent()
    agent._review_input_token_budget = 1

    result = _run_with_responses(agent, [_final_response()])

    assert agent.client.chat.completions.create.call_count == 0
    assert result["completed"] is False


def test_refused_first_request_logs_the_review_owner_and_a_stable_reason(caplog):
    """A detached review whose FIRST request is refused by the aggregate budget makes zero
    provider calls and writes nothing. That exit must be as greppable as every other skip:
    one body-free line carrying the hashed owner tag and a stable reason, never the
    session id."""
    from agent import review_admission

    agent = _make_loop_agent()
    agent.session_id = "loop-session-id"
    agent._review_input_token_budget = 1
    agent._review_owner_tag = review_admission.owner_tag("profile", agent.session_id)

    with caplog.at_level("INFO"):
        result = _run_with_responses(agent, [_final_response()])

    assert agent.client.chat.completions.create.call_count == 0
    assert result["completed"] is False
    skips = [
        record.getMessage()
        for record in caplog.records
        if "Background review skipped" in record.getMessage()
    ]
    assert len(skips) == 1, skips
    assert f"owner={agent._review_owner_tag}" in skips[0]
    assert review_admission.REASON_INPUT_BUDGET_REFUSED in skips[0]
    assert agent.session_id not in skips[0]


def test_exhaustion_after_a_completed_request_is_not_a_refused_review(caplog):
    """Upstream's contract: the budget-crossing request completes and the loop stops at the
    top of the next iteration. That exhaustion is not a skipped review and must not be
    logged as one."""
    agent = _make_loop_agent()
    agent._review_input_token_budget = 100_000
    agent._review_owner_tag = "0123456789ab"

    with caplog.at_level("INFO"):
        _run_with_responses(
            agent, [_tool_response(50_000), _tool_response(50_000), _final_response()]
        )

    assert agent.client.chat.completions.create.call_count >= 1
    assert "Background review skipped" not in caplog.text


def test_retry_of_a_failed_attempt_reserves_the_request_once():
    """The aggregate budget is reserved once per REQUEST, not per provider attempt. A request
    projected above half the remaining budget (the normal long-session case once the replay
    fills it) whose first attempt dies of a retryable provider error must still be retried:
    the failed attempt's reservation is released, the retry is admitted and completes, and the
    counter carries one projection, never two."""
    agent = _make_loop_agent()
    agent._review_input_token_budget = 100_000

    class RateLimited(Exception):
        status_code = 429

    reserved_per_attempt = []

    def _create(**_kwargs):
        reserved_per_attempt.append(agent._review_input_tokens_reserved)
        if len(reserved_per_attempt) == 1:
            raise RateLimited("rate limit exceeded")
        return _final_response()

    with patch("agent.turn_api_error.interruptible_backoff_sleep", return_value=None):
        result = _run_with_responses(agent, _create, user_message="x" * 300_000)

    create = agent.client.chat.completions.create
    assert create.call_count == 2, (
        f"expected the retry of the failed attempt to be admitted, but {create.call_count} "
        f"provider call(s) were made (reserved={agent._review_input_tokens_reserved}, "
        f"budget={agent._review_input_token_budget})"
    )
    assert result["completed"] is True
    projection = reserved_per_attempt[0]
    # The case under test: one request alone projects above half the budget but fits in it.
    assert (
        agent._review_input_token_budget // 2
        < projection
        <= agent._review_input_token_budget
    )
    assert reserved_per_attempt == [projection, projection]
    assert agent._review_input_tokens_reserved == projection


def test_retry_of_an_invalid_response_reserves_the_request_once():
    """A malformed HTTP-200 body (``validate_response_shape`` rejects it; no usage is folded)
    re-attempts the SAME request without raising, so the retry loop never passes its error
    handlers. The reservation must still be released before that re-attempt, else a request
    projected above half the remaining budget is refused on its own retry."""
    agent = _make_loop_agent()
    agent._review_input_token_budget = 100_000

    reserved_per_attempt = []

    def _create(**_kwargs):
        reserved_per_attempt.append(agent._review_input_tokens_reserved)
        if len(reserved_per_attempt) == 1:
            return None  # validate_response_shape -> "response is None"
        return _final_response()

    # retry_invalid_response imports the backoff sleep function-locally from agent.turn_recovery.
    with patch("agent.turn_recovery.interruptible_backoff_sleep", return_value=None):
        result = _run_with_responses(agent, _create, user_message="x" * 300_000)

    create = agent.client.chat.completions.create
    assert create.call_count == 2, (
        f"expected the retry of the invalid response to be admitted, but {create.call_count} "
        f"provider call(s) were made (reserved={agent._review_input_tokens_reserved}, "
        f"budget={agent._review_input_token_budget})"
    )
    assert result["completed"] is True
    projection = reserved_per_attempt[0]
    assert (
        agent._review_input_token_budget // 2
        < projection
        <= agent._review_input_token_budget
    )
    assert reserved_per_attempt == [projection, projection]
    assert agent._review_input_tokens_reserved == projection


def test_no_budget_attribute_leaves_tool_loop_unbounded():
    """Agents without ``_review_input_token_budget`` (every normal agent)
    are unaffected by the gate and consume all scripted responses."""
    agent = _make_loop_agent()

    responses = [
        _tool_response(50_000),
        _tool_response(50_000),
        _tool_response(50_000),
        _final_response(),
    ]
    result = _run_with_responses(agent, responses)

    assert agent.client.chat.completions.create.call_count == 4
    assert result["completed"] is True
    assert result["final_response"] == "done"


@pytest.mark.parametrize(
    ("write_origin", "expected"),
    [("background_review", 49_152), ("side_question", None)],
)
def test_aggregate_input_budget_is_scoped_to_the_review_fork(write_origin, expected):
    """75% of the fork's window bounds the automatic review; a /btw fork carries no aggregate
    budget at all (its single request is answered for by the provider's window)."""
    from agent.background_review import build_cache_parity_fork

    parent = _make_loop_agent()
    with _agent_init_seams(context_window=65_536):
        fork, _rt, _routed = build_cache_parity_fork(
            parent, {}, max_iterations=3, write_origin=write_origin
        )

    assert fork._review_input_token_budget == expected


def test_side_question_fork_first_request_is_never_refused_by_the_review_budget():
    """The aggregate budget bounds the WHOLE automatic review and is reserved before every
    provider request, the first included. /btw shares the fork builder but is one
    prefix-extension call over the live transcript: upstream always issued that request and
    let the provider answer for the window, so a transcript past the review's 75% share must
    still reach the provider instead of silently falling back to the digest."""
    from agent.background_review import _review_input_token_budget, build_cache_parity_fork
    from agent.model_metadata import estimate_messages_tokens_rough
    from agent.side_question import answer_side_question

    parent = _make_loop_agent()
    built = []

    def build_wired_fork(agent, task_cfg, **kwargs):
        with _agent_init_seams(context_window=65_536):
            fork, rt, routed = build_cache_parity_fork(agent, task_cfg, **kwargs)
        # What the review budget would allow a fork of this window, before the loop doubles
        # replace the compressor the resolver reads.
        review_share = _review_input_token_budget({}, fork)
        _wire_loop_doubles(fork)
        fork.client.chat.completions.create.side_effect = [_final_response()]
        fork._cleanup_task_resources = lambda *_a, **_k: None
        built.append((fork.client.chat.completions.create, review_share))  # close() drops client
        return fork, rt, routed

    history = [
        {"role": "user", "content": "x " * 110_000},
        {"role": "assistant", "content": "noted"},
    ]
    with (
        patch("agent.background_review.build_cache_parity_fork", build_wired_fork),
        patch(
            "agent.side_question._answer_via_oneshot",
            side_effect=AssertionError("/btw fell back to the digest"),
        ),
    ):
        answer = answer_side_question("what did I ask for?", history, parent_agent=parent)

    ((create, review_share),) = built
    assert estimate_messages_tokens_rough(history) > review_share  # past the review's budget
    assert create.call_count == 1
    assert answer == "done"


def test_review_input_budget_exhausted_predicate_edge_cases():
    """The gate only arms for a positive int budget and a real token count."""
    from agent.conversation_loop import _review_input_budget_exhausted

    class _Agent:
        pass

    agent = _Agent()
    assert _review_input_budget_exhausted(agent) is False

    agent._review_input_token_budget = None
    agent.session_input_tokens = 999_999
    assert _review_input_budget_exhausted(agent) is False

    agent._review_input_token_budget = 0
    assert _review_input_budget_exhausted(agent) is False

    agent._review_input_token_budget = -1
    assert _review_input_budget_exhausted(agent) is False

    agent._review_input_token_budget = "100"
    assert _review_input_budget_exhausted(agent) is False

    agent._review_input_token_budget = True
    assert _review_input_budget_exhausted(agent) is False

    agent._review_input_token_budget = 100_000
    agent.session_input_tokens = 99_999
    assert _review_input_budget_exhausted(agent) is False

    agent.session_input_tokens = 100_000
    assert _review_input_budget_exhausted(agent) is True


def test_review_input_budget_counts_cached_provider_input():
    from agent.conversation_loop import _review_input_budget_exhausted

    agent = SimpleNamespace(
        _review_input_token_budget=100,
        session_input_tokens=0,
        session_prompt_tokens=100,
    )

    assert _review_input_budget_exhausted(agent) is True


@pytest.mark.parametrize("raw", [0, -5, 600_001])
def test_operator_cannot_disable_or_raise_automatic_review_input_bound(raw):
    """A disable (<= 0) or an over-ceiling value never makes an automatic review unbounded: with a
    window large enough for the derived default to reach the ceiling, every such value is 600k."""
    from agent.background_review import _review_input_token_budget

    fork = SimpleNamespace(context_compressor=SimpleNamespace(context_length=2_000_000))
    budget = _review_input_token_budget({"max_input_tokens": raw}, fork)
    assert budget is not None
    assert budget == 600_000


@pytest.mark.parametrize(
    ("config_value", "expected"),
    [
        ({"max_input_tokens": 1_000_000}, 600_000),
        ({"max_input_tokens": 600_001}, 600_000),
        ({"max_input_tokens": 0}, 120_000),
        ({"max_input_tokens": -5}, 120_000),
        ({"max_input_tokens": "not-a-number"}, 120_000),
        ({"max_input_tokens": True}, 120_000),
        ({}, 120_000),
        ({"max_input_tokens": "300000"}, 300_000),
    ],
)
def test_review_input_token_budget_resolution(config_value, expected):
    """Explicit values keep their override up to the 600k ceiling; unset, disable, boolean and
    garbage fall back to the derived default (120k with no resolvable window) — never unlimited."""
    from agent.background_review import _review_input_token_budget

    assert _review_input_token_budget(config_value) == expected


def test_review_input_token_budget_default_tracks_forks_context_window():
    """Unset or malformed ``max_input_tokens`` → 75% of the fork's RESOLVED window (a 65k local
    model gets ~49k, not the cloud-scale 600k), capped at 600k; unknown window → 120k fallback."""
    from agent.background_review import _review_input_token_budget

    def fork(window):
        return SimpleNamespace(context_compressor=SimpleNamespace(context_length=window))

    assert _review_input_token_budget({}, fork(65_536)) == 49_152
    assert _review_input_token_budget({"max_input_tokens": "not-a-number"}, fork(65_536)) == 49_152
    assert _review_input_token_budget({}, fork(2_000_000)) == 600_000
    assert _review_input_token_budget({}, fork(None)) == 120_000
    assert _review_input_token_budget({}, None) == 120_000


def test_background_review_config_does_not_freeze_a_fixed_input_budget():
    """The config default must leave the budget resolver access to the active runtime."""
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert "max_input_tokens" not in DEFAULT_CONFIG["auxiliary"]["background_review"]
