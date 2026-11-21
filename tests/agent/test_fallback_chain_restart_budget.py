"""Regression for the rebuilt-restart cap × fallback-chain interaction.

Each provider switch spends one unit of the ``restart_with_rebuilt_messages`` budget
(``agent/turn_iteration_prep.py``), previously capped at flat ``max_retries`` (default 3).
A chain longer than ``max_retries + 1`` rungs therefore could never reach its tail: the
turn was killed with ``rebuilt_restart_limit_exceeded`` one provider switch before
testing the next rung — the 2026-11-10 incident where rungs 6-8 of an 8-rung chain were
unreachable by construction.

The cap must scale with the chain: every rebuilt restart follows a successful fallback
activation (``_arm_fallback_restart`` / content-filter fallback), the chain index is
monotone once a rung binds (primary-transport recovery is refused while
``_fallback_activated``), so ``max_retries + len(_fallback_chain)`` bounds the walk
without cutting it short.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent

MAX_RETRIES = 3
CHAIN_LEN = 9  # > MAX_RETRIES + 1: unreachable under the old flat cap


def _make_tool_defs():
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "search",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def _make_agent(chain):
    """Minimal AIAgent whose fallback chain is ``chain`` (list of provider/model rungs)."""
    with (
        patch("model_tools.get_tool_definitions", return_value=_make_tool_defs()),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()),
    ):
        agent = AIAgent(
            api_key="primary-key-abcdef12",
            base_url="https://primary.invalid/v1",
            provider="primary-prov",
            model="primary-model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=chain,
        )
    agent.client = MagicMock()
    return agent


def _chain(n: int) -> list[dict]:
    return [
        {"provider": f"rung-prov-{i}", "model": f"rung-model-{i}",
         "base_url": f"https://rung-{i}.invalid/v1"}
        for i in range(n)
    ]


def _mock_response(content: str):
    msg = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="rung-model-x", usage=None)


class RateLimitError(Exception):
    status_code = 429

    def __init__(self):
        super().__init__("Error code: 429 - rate limit exceeded")
        self.response = SimpleNamespace(headers={})
        self.body = {"error": {"message": "rate limit exceeded"}}


def _drive_walk(agent, responses_by_attempt):
    """Drive one ``run_conversation`` over a chain where every rung except the last 429s.

    ``responses_by_attempt`` maps attempt number -> response/exception; attempts beyond the
    mapping raise AssertionError so an unexpected extra call fails loudly.
    """
    calls: list[tuple[str, str]] = []

    def fake_api_call(api_kwargs):
        provider, model = agent.provider, agent.model
        calls.append((provider, model))
        outcome = responses_by_attempt.get(len(calls))
        if outcome is None:
            raise AssertionError(f"unexpected API call #{len(calls)} on {provider}/{model}")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    mock_client = MagicMock()
    mock_client.api_key = "primary-key-abcdef12"
    mock_client.base_url = "https://primary.invalid/v1"
    mock_client._custom_headers = None
    mock_client.default_headers = None

    with (
        patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()),
        patch("agent.auxiliary_client.resolve_provider_client", return_value=(mock_client, None)),
        patch(
            "hermes_cli.model_normalize.normalize_model_for_provider",
            side_effect=lambda m, p: m,
        ),
        patch("agent.model_metadata.get_model_context_length", return_value=200000),
    ):
        result = agent.run_conversation("hello")

    return result, calls


class TestLongChainWalk:

    def test_chain_longer_than_cap_reaches_healthy_tail(self):
        """9 rungs, final rung healthy: the turn must complete on the last rung.

        Under the flat cap the walk died on the 4th switch with
        ``rebuilt_restart_limit_exceeded`` and never issued the rung-5 request.
        """
        agent = _make_agent(_chain(CHAIN_LEN))
        agent._api_max_retries = MAX_RETRIES
        # Attempt 1: primary 429s. Each walk attempt then 429s until the final rung answers.
        script: dict[int, object] = {1: RateLimitError()}
        walk_attempts = 1 + CHAIN_LEN  # primary + each rung gets one attempt before switching
        for i in range(2, walk_attempts):
            script[i] = RateLimitError()
        script[walk_attempts] = _mock_response("recovered on the final rung")

        result, calls = _drive_walk(agent, script)

        assert result["completed"] is True, result
        assert result["final_response"] == "recovered on the final rung"
        assert result["turn_exit_reason"].startswith("text_response")
        # The walk reached and completed on the last rung.
        assert calls[-1] == ("rung-prov-8", "rung-model-8"), calls
        assert len(calls) == 1 + CHAIN_LEN, calls

    def test_walk_terminates_when_every_rung_fails(self):
        """All 9 rungs + primary 429: every rung is attempted once, then the terminal rung
        burns its same-provider retries and the walk ENDS — no infinite loop, no mid-chain death."""
        agent = _make_agent(_chain(CHAIN_LEN))
        agent._api_max_retries = MAX_RETRIES
        script: dict[int, object] = {}
        for i in range(1, 2 + CHAIN_LEN + MAX_RETRIES + 2):  # enough 429s for any walk length
            script[i] = RateLimitError()

        result, calls = _drive_walk(agent, script)

        # Not completed (the explainer text is the final response); every rung attempted.
        assert result["completed"] is False, result
        rung_providers = [c[0] for c in calls if c[0].startswith("rung-prov-")]
        # Rungs 0..7 one attempt each (a next rung is available after each 429)…
        assert rung_providers[:-MAX_RETRIES] == [f"rung-prov-{i}" for i in range(CHAIN_LEN - 1)], calls
        # …and the final rung, with the chain spent, exhausts its same-provider retries.
        assert rung_providers[-MAX_RETRIES:] == ["rung-prov-8"] * MAX_RETRIES, calls
        assert calls[0] == ("primary-prov", "primary-model"), calls

    def test_chain_spent_walk_ends_via_same_provider_retry_bound(self):
        """After the chain is spent the same-provider retry budget bounds the turn.

        This pins the monotone bound: the walk never re-enters the chain (primary once,
        the single rung for ``max_retries`` attempts, then the turn ends).
        """
        agent = _make_agent(_chain(1))
        agent._api_max_retries = MAX_RETRIES
        # Primary 429 -> rung switch; rung 429 with the chain spent -> retries to exhaustion.
        script = {i: RateLimitError() for i in range(1, 2 + MAX_RETRIES)}
        result, calls = _drive_walk(agent, script)
        assert result["completed"] is False, result
        assert [c[0] for c in calls] == ["primary-prov"] + ["rung-prov-0"] * MAX_RETRIES, calls


class TestTripSiteHonesty:

    def test_cap_trip_stashes_attempted_and_untried(self):
        """A genuine budget-exhausted turn (ping-pong arming past the scaled cap) stashes
        the walk summary, and the explainer names attempted vs never-attempted rungs."""
        from agent.turn_iteration_prep import (
            FALLBACK_WALK_SUMMARY_ATTR, apply_retry_restarts,
        )
        from agent.turn_retry_state import TurnRetryState
        from types import SimpleNamespace as _NS

        budget = _NS(refunds=0)
        budget.refund = lambda: setattr(budget, "refunds", budget.refunds + 1)
        agent = _NS(
            model="primary-model", provider="primary-prov",
            iteration_budget=budget,
            _fallback_chain=_chain(CHAIN_LEN), _fallback_index=3,
        )
        _retry = TurnRetryState(restart_with_rebuilt_messages=True)
        restart_count = 0
        verdicts = []
        # Re-arm every iteration: the walk must refund until the scaled cap
        # (max_retries + chain) trips, then break with the honest exit reason.
        while len(verdicts) < MAX_RETRIES + CHAIN_LEN + 5 and (
            not verdicts or verdicts[-1].action != "break"
        ):
            _retry.restart_with_rebuilt_messages = True
            v = apply_retry_restarts(
                agent, _retry=_retry, response=None, interrupted=False, messages=[],
                conversation_history=[], user_message="hi", api_kwargs={},
                current_turn_user_idx=0, final_response=None, retry_count=0,
                max_retries=MAX_RETRIES, api_call_count=1, restart_count=restart_count,
                length_continue_retries=0,
                _preflight_compression_blocked=True, _turn_exit_reason="unknown",
            )
            verdicts.append(v)
            restart_count = v.restart_count
        assert verdicts[-1].action == "break"
        assert verdicts[-1]._turn_exit_reason == "rebuilt_restart_limit_exceeded"
        summary = getattr(agent, FALLBACK_WALK_SUMMARY_ATTR)
        assert {e["provider"] for e in summary["attempted"]} == {f"rung-prov-{i}" for i in range(3)}
        assert {e["provider"] for e in summary["untried"]} == {f"rung-prov-{i}" for i in range(3, 9)}

    def test_explainer_names_attempted_and_untried_on_cap_trip(self):
        """The delivered explanation on a cap-tripped turn lists both sets by name."""
        from agent.turn_explainers import explain_fallback_walk_exhaustion

        agent = SimpleNamespace(
            model="primary-model", provider="primary-prov",
            _fallback_chain=_chain(CHAIN_LEN), _fallback_index=3,
            _fallback_walk_summary={
                "attempted": [{"label": "rung-model-0 (rung-prov-0)"}, {"label": "rung-model-1 (rung-prov-1)"}],
                "untried": [{"label": "rung-model-8 (rung-prov-8)"}],
                "primary": {"provider": "primary-prov", "model": "primary-model"},
            },
        )
        out = explain_fallback_walk_exhaustion(agent)
        assert "rung-prov-0" in out and "rung-prov-1" in out
        assert "rung-prov-8" in out
        assert "never sent" in out
        # The dishonest sentence is gone.
        assert "every provider in the fallback chain kept failing over" not in out

    def test_result_carries_explainer_text_naming_providers(self):
        """The delivered explanation must name attempted and (if any) untried providers."""
        from agent.turn_explainers import explain_fallback_walk_exhaustion

        agent = SimpleNamespace(
            model="primary-model", provider="primary-prov",
            _fallback_chain=_chain(CHAIN_LEN), _fallback_index=3,
        )
        out = explain_fallback_walk_exhaustion(agent)
        assert "rung-prov-0" in out
        assert "rung-prov-8" in out
        assert "never sent" in out or "untried" in out
