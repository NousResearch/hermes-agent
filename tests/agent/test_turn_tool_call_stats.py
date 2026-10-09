"""Turn-level tool-call blocked tally (#135544).

A cron run whose tools were ALL blocked by a fail-closed plugin raised nothing and
delivered a normal-looking report, so the ledger booked it as a success unless the
model happened to open with an exact ``[CRON_FAILURE]`` line. The runtime signal —
how many tool calls were attempted vs blocked before dispatch — is now tallied per
turn in ``_commit_tool_result`` and exported on the ``run_conversation`` result, so
schedulers can tell a dead turn from a healthy one without trusting a self-report.

Producer contract pinned here:

  * every call blocked (plugin hook) -> ``tool_calls_attempted == tool_calls_blocked``
  * a text-only turn leaves the result dict without either key

The fire-path consumption (all-blocked => failed bookkeeping) is pinned separately
in tests/cron/test_run_one_job.py.
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent

# Non-secret placeholder (assembled, not a credential literal): this agent never
# reaches a real provider — the client is a MagicMock below.
_TEST_API_KEY = "test-" + "key"


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _make_agent():
    hermes_home = Path(tempfile.mkdtemp(prefix="hermes-stats-home-"))
    (hermes_home / "logs").mkdir(parents=True, exist_ok=True)
    with (
        patch(
            "model_tools.get_tool_definitions",
            return_value=_make_tool_defs("web_search"),
        ),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("run_agent._hermes_home", hermes_home),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(
            api_key=_TEST_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _mock_tool_call(name="web_search", arguments="{}", call_id="call_1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def test_all_blocked_turn_exports_block_stats():
    """Every tool call blocked by a plugin hook: the result carries the dead-turn signal
    (attempted == blocked) even though the turn itself completed normally."""
    agent = _make_agent()
    agent.client.chat.completions.create.side_effect = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call(call_id="c1")],
        ),
        _mock_response(content="done without data", finish_reason="stop"),
    ]

    with (
        patch(
            "hermes_cli.plugins._dispatch_pre_tool_call_hooks",
            return_value=(
                "Native handoff unavailable; tool blocked by escalation policy",
                None,
            ),
        ) as hooks,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("collect the daily numbers")

    assert hooks.call_count == 1
    assert result["tool_calls_attempted"] == 1
    assert result["tool_calls_blocked"] == 1
    assert result["final_response"] == "done without data"


def test_text_only_turn_omits_block_stats():
    """No tool call attempted: neither key is stamped, so text-only turns keep the
    result dict unchanged for callers keyed on absence."""
    agent = _make_agent()
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="nothing to do", finish_reason="stop"),
    ]

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("just say hi")

    assert "tool_calls_attempted" not in result
    assert "tool_calls_blocked" not in result
