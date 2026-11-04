"""run_conversation ends the turn when a model keeps sending non-object tool arguments (#125368).

Split out of ``tests/agent/test_run_agent.py`` (code-health size ratchet); shares its fixtures.
"""

from unittest.mock import patch

import pytest

from tests.agent import test_run_agent as _base
from tests.agent.test_run_agent import _mock_response, _mock_tool_call

# Reuse the facade's fixtures by name (an import would shadow-redefine ``agent`` — F811).
agent = _base.agent
_mock_plugin_discovery = _base._mock_plugin_discovery


class TestMalformedToolArguments:
    def _setup_agent(self, agent):
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.compression_enabled = False
        agent.save_trajectories = False

    def _run_with_responses(self, agent, responses):
        self._setup_agent(agent)
        agent.client.chat.completions.create.side_effect = responses
        with (
            patch("model_tools.handle_function_call", return_value="search result") as dispatch,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            return agent.run_conversation("search something"), dispatch

    @staticmethod
    def _tool_response(arguments):
        return _mock_response(content="", finish_reason="tool_calls",
                              tool_calls=[_mock_tool_call(name="web_search", arguments=arguments)])

    @pytest.mark.parametrize("arguments", ["{'query': 'x'}", '{"query": "x",}', "[1, 2]", "42", '"just a string"'])
    def test_malformed_tool_arguments_end_the_turn_instead_of_looping(self, agent, arguments):
        """#125368: a model that never sends a JSON object gets two recovery rounds (9 responses),
        then a typed failure, instead of re-requesting until the iteration budget is gone."""
        result, dispatch = self._run_with_responses(agent, [self._tool_response(arguments)] * 50)
        assert result["api_calls"] == 9
        assert result["failure_reason"] == "malformed_tool_call"
        assert result["failure_retryable"] is False
        assert agent.model in result["final_response"] and "/model" in result["final_response"]
        dispatch.assert_not_called()

    def test_malformed_tool_arguments_recover_after_error_results(self, agent):
        bad, good = self._tool_response("[1]"), self._tool_response('{"query": "x"}')
        done = _mock_response(content="Done", finish_reason="stop")
        result, dispatch = self._run_with_responses(agent, [bad] * 4 + [good, done])
        assert result["final_response"] == "Done"
        assert result["completed"] is True
        dispatch.assert_called_once()

    def test_valid_tool_round_resets_the_malformed_argument_budget(self, agent):
        # Two recovery rounds, a valid call, then two more: 4 rounds in one turn, never 3 in a row.
        bad, good = self._tool_response("[1]"), self._tool_response('{"query": "x"}')
        done = _mock_response(content="Done", finish_reason="stop")
        result, dispatch = self._run_with_responses(agent, [bad] * 6 + [good] + [bad] * 6 + [good, done])
        assert result["final_response"] == "Done"
        assert dispatch.call_count == 2
