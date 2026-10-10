"""Automatic review on small context windows: the derived budget funds a bounded review.

Measured before this fix: the bare default toolset (``get_tool_definitions(quiet_mode=True)``,
~10k estimated tokens) plus the combined review prompt (~2.1k) and the request margin (2k) with
a 4k-token system prompt is ~18k of fixed request parts — more than one third of the derived
75% aggregate on a 65,536-token window (16,384). ``replay_token_budget`` collapsed to 1,
``bounded_replay_history`` returned nothing, and EVERY automatic review was skipped as
``oversized_snapshot`` on exactly the managed-local deployments this PR defers for idle, where
upstream replayed the whole snapshot. The derived aggregate now funds three requests of the
fixed parts plus a replay floor (never past what one request on the window carries, nor the
600k ceiling); an explicit ``max_input_tokens`` is never raised, and fixed parts that exceed a
whole request skip the review loudly under their own slug instead of ``oversized_snapshot``.
"""

from __future__ import annotations

import logging
import types

import pytest

import run_agent as run_agent_module
from agent import background_review as background_review_module
from agent import review_admission
from agent.model_metadata import (
    _estimate_tools_tokens_rough,
    estimate_messages_tokens_rough,
)
from model_tools import get_tool_definitions
from run_agent import AIAgent
from tests.agent.test_background_review_compute_overlap import (  # fixtures
    _bare_agent,
    _config,
    _patch_config,
    _snapshot,
    clear_review_admission_state,
    review_forks,
)

_FOUR_K_TOKEN_PROMPT = "system prompt " + "s" * 16_000


def _gateway_agent(window):
    """A parent on the DEFAULT toolset with a realistic system prompt and the real review prompt."""
    agent = _bare_agent()
    if window is not None:
        agent.context_compressor = types.SimpleNamespace(context_length=window)
    agent._cached_system_prompt = _FOUR_K_TOKEN_PROMPT
    agent.tools = get_tool_definitions(quiet_mode=True)
    agent._COMBINED_REVIEW_PROMPT = background_review_module._COMBINED_REVIEW_PROMPT
    return agent


def _windowed_fork(monkeypatch, window):
    """The review_forks recorder, resolving ``window`` like a same-model fork would."""
    fake_fork = run_agent_module.AIAgent

    class WindowedFork(fake_fork):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.context_compressor = types.SimpleNamespace(context_length=window)

    monkeypatch.setattr(run_agent_module, "AIAgent", WindowedFork)


def _walk_three_requests(record) -> list[str]:
    """Drive the loop's own reservation accounting through a read, a write and a close."""
    from agent.conversation_loop import _reserve_review_input_request

    fork = types.SimpleNamespace(
        _review_input_token_budget=record["attrs"]["_review_input_token_budget"],
        session_prompt_tokens=0,
        _review_input_tokens_reserved=0,
    )
    tools_tokens = _estimate_tools_tokens_rough(record["attrs"]["tools"])
    messages = (
        [{"role": "system", "content": record["attrs"]["_cached_system_prompt"]}]
        + list(record["history"])
        + [{"role": "user", "content": record["user_message"]}]
    )
    admitted = []
    for step, tool_result in (
        ("read", "r" * 4_000),
        ("write", "w" * 4_000),
        ("close", None),
    ):
        projected = estimate_messages_tokens_rough(messages) + tools_tokens
        if not _reserve_review_input_request(fork, projected):
            break
        admitted.append(step)
        fork.session_prompt_tokens += projected
        if tool_result is not None:
            messages += [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": step,
                            "type": "function",
                            "function": {"name": "memory", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": step, "content": tool_result},
            ]
    return admitted


def test_default_toolset_on_a_64k_window_still_spawns_an_automatic_review(
    review_forks,  # health: allow F811 -- pytest injects the imported fixture by parameter name
    monkeypatch,
):
    """The production regime the budget-formula tests never covered: a 65,536-token window, the
    default toolset, a 4k-token system prompt. The fixed parts exceed a third of the derived
    aggregate, yet the review spawns, replays the (small) snapshot verbatim, and its fork's
    aggregate funds a read, a write and a closing request."""
    _patch_config(monkeypatch, _config())
    window = 65_536
    _windowed_fork(monkeypatch, window)
    agent = _gateway_agent(window)
    prompt = agent._COMBINED_REVIEW_PROMPT
    derived = background_review_module._review_input_token_budget({}, agent)
    overhead = review_admission.request_overhead_tokens(agent, prompt)
    assert (
        overhead > derived // review_admission.REVIEW_REQUEST_SHARES
    )  # the old rule: replay 1
    snapshot = _snapshot(pairs=6, filler_chars=2_000)
    assert (
        estimate_messages_tokens_rough(snapshot) < review_admission.REPLAY_FLOOR_TOKENS
    )

    AIAgent._spawn_background_review(
        agent, messages_snapshot=snapshot, review_memory=True, review_skills=True
    )

    assert len(review_forks) == 1
    record = review_forks[0]
    assert record["history"] == snapshot
    budgets = review_admission.review_budgets({}, agent, prompt)
    assert record["attrs"]["_review_input_token_budget"] == budgets.aggregate
    assert _walk_three_requests(record) == ["read", "write", "close"]


def test_review_budgets_floor_the_derived_aggregate_on_small_windows():
    """The rule, window by window, on the default toolset with a 4k system prompt."""
    prompt = background_review_module._COMBINED_REVIEW_PROMPT
    shares = review_admission.REVIEW_REQUEST_SHARES
    floor = review_admission.REPLAY_FLOOR_TOKENS

    # 200k: the share already funds more than the floor — today's arithmetic, untouched.
    agent = _gateway_agent(200_000)
    budgets = review_admission.review_budgets({}, agent, prompt)
    derived = background_review_module._review_input_token_budget({}, agent)
    assert budgets.aggregate == derived == 150_000
    assert budgets.replay == derived // shares - budgets.overhead > floor
    assert budgets.unfunded is False

    # 64k: the share cannot carry the floor — the aggregate is raised to fund three requests of
    # the fixed parts plus the floor, and the replay is the floor.
    agent = _gateway_agent(65_536)
    budgets = review_admission.review_budgets({}, agent, prompt)
    derived = background_review_module._review_input_token_budget({}, agent)
    assert derived == 49_152 and derived // shares < budgets.overhead
    assert budgets.replay == floor
    assert budgets.aggregate == shares * (budgets.overhead + floor) > derived
    assert budgets.unfunded is False

    # 32k: one request (75% of the window) carries less than the floor — the replay is what
    # fits, the aggregate funds three such requests, nothing wider.
    agent = _gateway_agent(32_768)
    budgets = review_admission.review_budgets({}, agent, prompt)
    request_fit = (
        int(32_768 * background_review_module._REVIEW_INPUT_CONTEXT_FRACTION)
        - budgets.overhead
    )
    assert 0 < request_fit < floor
    assert budgets.replay == request_fit
    assert budgets.aggregate == shares * (budgets.overhead + request_fit)
    assert budgets.unfunded is False

    # 16k: the fixed parts alone exceed a request — nothing can be replayed.
    agent = _gateway_agent(16_384)
    budgets = review_admission.review_budgets({}, agent, prompt)
    assert budgets.unfunded is True and budgets.replay == 1

    # The ceiling still binds on every window.
    assert review_admission.replay_token_budget(
        {}, _gateway_agent(2_000_000), prompt
    ) == (review_admission.MAX_REPLAY_TOKENS_DEFAULT)


def test_explicit_max_input_tokens_is_never_raised():
    """An operator's cap can narrow the review, never be lifted by the floor: a cap that funds a
    narrower-than-floor share replays that share; one that cannot fund the fixed parts leaves
    the review unfunded."""
    prompt = background_review_module._COMBINED_REVIEW_PROMPT
    agent = _gateway_agent(65_536)
    shares = review_admission.REVIEW_REQUEST_SHARES

    budgets = review_admission.review_budgets(
        {"max_input_tokens": 80_000}, agent, prompt
    )
    assert budgets.aggregate == 80_000
    assert budgets.replay == 80_000 // shares - budgets.overhead
    assert 0 < budgets.replay < review_admission.REPLAY_FLOOR_TOKENS
    assert budgets.unfunded is False

    budgets = review_admission.review_budgets(
        {"max_input_tokens": 48_000}, agent, prompt
    )
    assert budgets.aggregate == 48_000
    assert budgets.unfunded is True


@pytest.mark.parametrize(
    "task_cfg, window",
    [({}, 16_384), ({"max_input_tokens": 48_000}, 65_536)],
    ids=["fixed_parts_exceed_a_request", "explicit_cap_below_the_fixed_parts"],
)
def test_unfunded_fixed_parts_skip_the_review_loudly(
    review_forks,  # health: allow F811 -- pytest injects the imported fixture by parameter name
    monkeypatch,
    caplog,
    task_cfg,
    window,
):
    """Fixed parts no replay can fit are a configuration problem, not a big turn: skipped at
    WARNING under their own owner-tagged slug, never as ``oversized_snapshot`` at INFO."""
    _patch_config(monkeypatch, _config(**task_cfg))
    _windowed_fork(monkeypatch, window)
    agent = _gateway_agent(window)

    with caplog.at_level(logging.INFO):
        AIAgent._spawn_background_review(
            agent,
            messages_snapshot=_snapshot(pairs=1, filler_chars=10),
            review_memory=True,
            review_skills=True,
        )

    assert review_forks == []
    lines = [
        r
        for r in caplog.records
        if review_admission.REASON_OVERHEAD_EXCEEDS_BUDGET in r.getMessage()
    ]
    assert len(lines) == 1, caplog.text
    assert lines[0].levelno == logging.WARNING
    assert (
        review_admission.owner_tag(
            review_admission.current_profile_key(), agent.session_id
        )
        in lines[0].getMessage()
    )
    assert agent.session_id not in lines[0].getMessage()
    assert review_admission.REASON_OVERSIZED not in caplog.text
