"""``post_api_request`` cost payload matches the turn's ``api_call_records``.

Split out of ``tests/agent/test_run_agent.py`` (code-health size ratchet); shares its fixtures.
"""

from unittest.mock import patch

import pytest

from tests.agent import test_run_agent as _base
from tests.agent.test_run_agent import _mock_response, _mock_tool_call

# Reuse the facade's fixtures by name (an import would shadow-redefine ``agent`` — F811).
agent = _base.agent
_mock_plugin_discovery = _base._mock_plugin_discovery


class TestPostApiRequestCost:
    def _setup_agent(self, agent):
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.compression_enabled = False
        agent.save_trajectories = False

    def test_post_api_request_cost_matches_the_turns_api_call_records(self, agent):
        from decimal import Decimal

        from agent.usage_pricing import CostResult

        self._setup_agent(agent)
        usage = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        tc = _mock_tool_call(name="web_search", arguments="{}", call_id="c1")
        agent.client.chat.completions.create.side_effect = [
            _mock_response(content="", finish_reason="tool_calls", tool_calls=[tc], usage=usage),
            _mock_response(content="Done searching", finish_reason="stop", usage=usage),
        ]
        amounts = iter([Decimal("0.25"), Decimal("0.5")])
        hook_calls = []

        def _record_hook(name, **kwargs):
            hook_calls.append((name, kwargs))
            return []

        with (
            patch("model_tools.handle_function_call", return_value="search result"),
            patch("agent.turn_usage.estimate_usage_cost", side_effect=lambda *a, **k: CostResult(
                amount_usd=next(amounts), status="estimated", source="official_docs_snapshot", label="$")),
            patch("hermes_cli.lifecycle.has_hook", side_effect=lambda name: name == "post_api_request"),
            patch("hermes_cli.lifecycle.invoke_hook", side_effect=_record_hook),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("search something")

        records = result["api_call_records"]
        hook_costs = [kw["cost"] for name, kw in hook_calls if name == "post_api_request"]
        assert [r["estimated_cost_usd"] for r in records] == [0.25, 0.5]
        assert hook_costs == [
            {k: r[k] for k in ("estimated_cost_usd", "cost_status", "cost_source")} for r in records]
        assert sum(r["estimated_cost_usd"] for r in records) == pytest.approx(result["estimated_cost_usd"])
        assert [r["total_tokens"] for r in records] == [120, 120]
