"""A slow turn (many tool rounds, or a failed tool that then worked) queues the background review."""

import json
from unittest.mock import MagicMock

from agent.turn_finalizer import _recovered_tool_names, _turn_tool_rounds, finalize_turn
from tests.agent.test_turn_finalizer_iteration_limit_exit import _LimitAgent


def _tool(name, payload):
    return {"role": "tool", "name": name, "tool_name": name, "content": json.dumps(payload)}


_RECOVERED_TURN = [
    {"role": "user", "content": "blinds to 30%"},
    _tool("ha_list_entities", {"result": {"count": 2}}),
    _tool("ha_call_service", {"error": "Failed to call cover.set_cover_position: 400"}),
    _tool("ha_get_state", {"result": {"state": "closed"}}),
    _tool("ha_call_service", {"result": {"success": True}}),
]


class TestRecoveredToolNames:
    def test_failure_then_success_of_same_tool(self):
        assert _recovered_tool_names(_RECOVERED_TURN) == {"ha_call_service"}

    def test_failure_without_later_success_is_not_a_recovery(self):
        assert _recovered_tool_names(_RECOVERED_TURN[:3]) == set()

    def test_only_the_latest_user_turn_counts(self):
        messages = _RECOVERED_TURN[:3] + [{"role": "user", "content": "next"}] + _RECOVERED_TURN[4:]
        assert _recovered_tool_names(messages) == set()

    def test_non_json_and_null_error_rows_are_not_failures(self):
        messages = [
            {"role": "user", "content": "x"},
            {"role": "tool", "tool_name": "web", "content": "<untrusted_tool_result>{\"error\": 1}"},
            _tool("web", {"result": 1, "error": None}),
            _tool("web", {"result": 2}),
        ]
        assert _recovered_tool_names(messages) == set()


def _recovery_agent(enabled=True, rounds=0):
    agent = _LimitAgent()
    agent._review_slow_turns = enabled
    agent._slow_turn_tool_rounds = rounds
    agent._skill_nudge_interval = 15
    agent._memory_nudge_interval = 10
    agent._turns_since_memory = 3
    agent._memory_store = object()
    agent.valid_tool_names = ["skill_manage", "memory", "ha_call_service"]
    agent._spawn_background_review = MagicMock()
    return agent


def _finalize(agent, messages=_RECOVERED_TURN):
    return finalize_turn(
        agent,
        final_response="done",
        api_call_count=4,
        interrupted=False,
        failed=False,
        messages=list(messages),
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="blinds to 30%",
        original_user_message="blinds to 30%",
        _should_review_memory=False,
        _turn_exit_reason="text_response",
    )


def test_recovery_queues_memory_and_skill_review_once_per_tool():
    agent = _recovery_agent()
    _finalize(agent)
    agent._spawn_background_review.assert_called_once()
    kwargs = agent._spawn_background_review.call_args.kwargs
    assert kwargs["review_memory"] is True and kwargs["review_skills"] is True
    assert agent._turns_since_memory == 0

    _finalize(agent)
    agent._spawn_background_review.assert_called_once()


def test_recovery_review_can_be_disabled():
    agent = _recovery_agent(enabled=False)
    _finalize(agent)
    agent._spawn_background_review.assert_not_called()


def _assistant_round(name):
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name}}]}


def _slow_turn(rounds):
    turn = [{"role": "user", "content": "living room red"}]
    for _ in range(rounds):
        turn += [_assistant_round("ha_get_state"), _tool("ha_get_state", {"result": {"state": "on"}})]
    return turn


def test_turn_tool_rounds_counts_only_the_latest_turn():
    assert _turn_tool_rounds(_slow_turn(2) + _slow_turn(3)) == 3


def test_slow_turn_queues_review_once_per_session():
    agent = _recovery_agent(rounds=4)
    _finalize(agent, _slow_turn(3))
    agent._spawn_background_review.assert_not_called()

    _finalize(agent, _slow_turn(4))
    agent._spawn_background_review.assert_called_once()

    _finalize(agent, _slow_turn(6))
    agent._spawn_background_review.assert_called_once()


def test_slow_turn_keys_are_read_from_config_yaml(tmp_path):
    from types import SimpleNamespace

    from agent.agent_init_skills import _apply_skill_review_config
    from hermes_cli.config import load_config_readonly
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    (tmp_path / "config.yaml").write_text(
        "skills:\n  review_slow_turns: false\n  slow_turn_tool_rounds: 6\n  creation_nudge_interval: 3\n",
        encoding="utf-8")
    token = set_hermes_home_override(tmp_path)
    try:
        config = load_config_readonly()
    finally:
        reset_hermes_home_override(token)
    agent = SimpleNamespace()
    _apply_skill_review_config(agent, config)
    assert (agent._review_slow_turns, agent._slow_turn_tool_rounds, agent._skill_nudge_interval) == (False, 6, 3)
    _apply_skill_review_config(agent, {"skills": {"slow_turn_tool_rounds": "many"}})
    assert (agent._review_slow_turns, agent._slow_turn_tool_rounds, agent._skill_nudge_interval) == (True, 4, 10)
