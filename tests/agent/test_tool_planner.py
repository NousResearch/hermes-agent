from unittest.mock import MagicMock, patch

from agent.tool_planner import (
    DEFAULT_SEQUENCE,
    build_default_plan,
    parse_planner_response,
    plan_retrieval_tool_use,
    should_skip_retrieval_planner,
)


def test_should_skip_planner_for_explicit_read_file_path():
    assert should_skip_retrieval_planner("read_file", {"path": "/tmp/demo.txt"}) is True
    assert should_skip_retrieval_planner("search_files", {"pattern": "demo", "path": "."}) is False


def test_build_default_plan_for_session_search_is_narrow():
    plan = build_default_plan(
        user_message="What did we discuss about NC ITPE last time?",
        function_name="session_search",
        function_args={"query": "NC OR ITPE", "limit": 3},
        max_retrieval_calls=4,
        allow_broad_search=False,
    )

    assert plan.recommended_sequence == ["session_search"]
    assert plan.allow_broad_search is False
    assert plan.max_retrieval_calls == 4


def test_parse_planner_response_falls_back_on_malformed_json():
    plan = parse_planner_response(
        "not json",
        user_message="find the packet",
        function_name="search_files",
        function_args={"pattern": "packet", "path": "."},
        max_retrieval_calls=4,
        allow_broad_search=False,
    )

    assert plan.source == "fallback"
    assert plan.recommended_sequence == DEFAULT_SEQUENCE
    assert plan.allow_broad_search is False


def test_parse_planner_response_filters_unknown_stages():
    content = """{
      \"needs_retrieval\": true,
      \"goal\": \"Find the packet\",
      \"recommended_sequence\": [\"exact_path\", \"weird_stage\", \"session_search\"],
      \"max_retrieval_calls\": 3,
      \"allow_broad_search\": false,
      \"stop_if\": [\"found_exact_path\"]
    }"""

    plan = parse_planner_response(
        content,
        user_message="find the packet",
        function_name="search_files",
        function_args={"pattern": "packet", "path": "."},
        max_retrieval_calls=4,
        allow_broad_search=False,
    )

    assert plan.source == "planner"
    assert plan.recommended_sequence == ["exact_path", "session_search"]
    assert plan.max_retrieval_calls == 3


def test_plan_retrieval_tool_use_uses_auxiliary_llm_when_enabled():
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = """{
      \"needs_retrieval\": true,
      \"goal\": \"Recover the exact packet path\",
      \"recommended_sequence\": [\"exact_path\", \"known_subtree\", \"session_search\"],
      \"max_retrieval_calls\": 3,
      \"allow_broad_search\": false,
      \"stop_if\": [\"found_exact_path\"]
    }"""

    with patch("agent.tool_planner.call_llm", return_value=mock_response) as mock_call:
        plan = plan_retrieval_tool_use(
            user_message="Recover the exact specialist packet path.",
            function_name="search_files",
            function_args={"pattern": "packet", "path": "./profiles/work"},
            recent_messages=[{"role": "user", "content": "Recover the exact specialist packet path."}],
            max_retrieval_calls=4,
            allow_broad_search=False,
        )

    assert plan.source == "planner"
    assert plan.recommended_sequence == ["exact_path", "known_subtree", "session_search"]
    assert plan.max_retrieval_calls == 3
    assert mock_call.call_args.kwargs["task"] == "planner"
