import json
import logging

from agent.request_budget import RequestBudget, estimate_tool_schema_tokens


def test_tool_schema_token_estimate_counts_serialized_schema():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            },
        }
    ]

    count, tokens, byte_count = estimate_tool_schema_tokens(tools)

    assert count == 1
    assert tokens > 0
    assert byte_count > 0


def test_request_budget_keeps_multi_tool_batch_names_machine_readable():
    budget = RequestBudget(
        session_id="session-1",
        turn_id="turn-1",
        model="gpt-5.5",
        provider="openai-codex",
        platform="slack",
    )

    budget.add_tool_execution(["read_file", "web_search"], 0.123)
    payload = budget.snapshot(reason="tool_response", api_calls=1)

    assert payload["tool_call_count"] == 2
    assert payload["tool_names"] == ["read_file", "web_search"]
    assert payload["tool_execution_ms"] == 123


def test_request_budget_logs_core_turn_phases(caplog):
    budget = RequestBudget(
        session_id="session-1",
        turn_id="turn-1",
        model="gpt-5.5",
        provider="openai-codex",
        platform="slack",
    )
    budget.record_tool_schema(
        [{"type": "function", "function": {"name": "read_file"}}]
    )
    budget.record_skill_index(
        prompt="## Skills (mandatory)\n<available_skills>\n- x\n</available_skills>"
    )
    budget.mark_model_request_start()
    budget.mark_model_first_byte()
    budget.mark_model_request_end()
    budget.add_tool_execution(["read_file"], 0.123)

    with caplog.at_level(logging.INFO):
        payload = budget.log_agent_turn(
            logger=logging.getLogger("tests.request_budget"),
            reason="text_response",
            api_calls=1,
        )

    assert payload["model_ttfb_ms"] is not None
    assert payload["tool_schema_tokens"] > 0
    assert payload["skill_index_tokens"] > 0
    assert payload["tool_execution_ms"] == 123
    assert "request_budget.v1" in caplog.text
    logged = caplog.records[-1].getMessage().split("request_budget.v1 ", 1)[1]
    assert json.loads(logged)["session_id"] == "session-1"
