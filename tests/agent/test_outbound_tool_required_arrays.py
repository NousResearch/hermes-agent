import copy

from agent.chat_completion_helpers import _ensure_required_arrays_for_openai_tools


def _normalize_one(parameters):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "test_tool",
                "parameters": parameters,
            },
        }
    ]
    return _ensure_required_arrays_for_openai_tools(copy.deepcopy(tools))[0]["function"]["parameters"]


def test_missing_required_becomes_empty_array():
    params = _normalize_one(
        {
            "type": "object",
            "properties": {},
        }
    )

    assert params["required"] == []


def test_required_none_becomes_empty_array():
    params = _normalize_one(
        {
            "type": "object",
            "properties": {},
            "required": None,
        }
    )

    assert params["required"] == []


def test_required_non_list_becomes_empty_array():
    params = _normalize_one(
        {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": "query",
        }
    )

    assert params["required"] == []


def test_valid_required_is_preserved():
    params = _normalize_one(
        {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
    )

    assert params["required"] == ["query"]


def test_invalid_required_entries_are_filtered_but_key_is_kept():
    params = _normalize_one(
        {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query", "missing"],
        }
    )

    assert params["required"] == ["query"]


def test_all_invalid_required_entries_become_empty_array():
    params = _normalize_one(
        {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["missing"],
        }
    )

    assert params["required"] == []


def test_nested_object_required_is_added():
    params = _normalize_one(
        {
            "type": "object",
            "properties": {
                "payload": {
                    "type": "object",
                    "properties": {},
                }
            },
        }
    )

    assert params["required"] == []
    assert params["properties"]["payload"]["required"] == []


def test_normalizer_does_not_mutate_original_tools():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "test_tool",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        }
    ]
    before = copy.deepcopy(tools)

    normalized = _ensure_required_arrays_for_openai_tools(copy.deepcopy(tools))

    assert tools == before
    assert normalized[0]["function"]["parameters"]["required"] == []


def test_input_schema_variants_are_normalized_defensively():
    tools = [
        {
            "name": "anthropic_style",
            "input_schema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "bedrock_style",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
    ]

    normalized = _ensure_required_arrays_for_openai_tools(copy.deepcopy(tools))

    assert normalized[0]["input_schema"]["required"] == []
    assert normalized[1]["inputSchema"]["required"] == []
