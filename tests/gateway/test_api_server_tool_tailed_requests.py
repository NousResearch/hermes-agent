"""Tool-tailed OpenAI-format chat requests (Open WebUI built-in tools).

Open WebUI's agent path pre-executes built-in tools and sends the results back
in the same ``/v1/chat/completions`` body, so the request legitimately ends
with a ``tool`` row after an empty assistant tool-call carrier:

    system, user, assistant(content="", tool_calls=[...]), tool(content=result)

The gateway must parse those rows and treat the last visible USER message as
the current turn instead of rejecting the request with
``No user message found in messages``.
"""

from __future__ import annotations

from gateway.platforms.api_server_openai_routes import (
    _parse_chat_messages,
    _select_current_turn,
)

SKILLS_RESULT = "<available_skills>ascii-art, github, plan\n</available_skills>"


def _owui_tool_tailed_messages() -> list:
    """Open WebUI agent-chat shape: system, user, assistant(tool_calls), tool."""
    return [
        {"role": "system", "content": "You are a test agent."},
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "fc_1",
                    "type": "function",
                    "function": {
                        "name": "list_available_skills",
                        "arguments": '{"skill_name": ""}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "fc_1",
            "name": "list_available_skills",
            "content": SKILLS_RESULT,
        },
    ]


def test_parse_preserves_tool_rows_and_tool_calls():
    system_prompt, conversation, err = _parse_chat_messages(_owui_tool_tailed_messages())
    assert err is None
    assert system_prompt == "You are a test agent."
    assert [m["role"] for m in conversation] == ["user", "assistant", "tool"]
    assert conversation[0]["content"] == "hi"
    # The assistant tool-call carrier keeps its tool_calls payload.
    assert conversation[1]["tool_calls"][0]["function"]["name"] == "list_available_skills"
    # The tool result is preserved as replayable context.
    assert "ascii-art" in conversation[2]["content"]


def test_tool_tailed_request_selects_last_visible_user_turn():
    _, conversation, err = _parse_chat_messages(_owui_tool_tailed_messages())
    assert err is None
    user_message, history = _select_current_turn(conversation)
    assert user_message == "hi"
    # The tool-call carrier and its result replay as history.
    assert [m["role"] for m in history] == ["assistant", "tool"]


def test_legacy_single_user_message_unchanged():
    _, conversation, err = _parse_chat_messages([{"role": "user", "content": "ping"}])
    assert err is None
    user_message, history = _select_current_turn(conversation)
    assert user_message == "ping"
    assert history == []


def test_legacy_assistant_tailed_continuation_unchanged():
    _, conversation, err = _parse_chat_messages(
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "partial reply"},
        ]
    )
    assert err is None
    user_message, history = _select_current_turn(conversation)
    # Legacy behavior: the last visible message (assistant continuation) is the turn.
    assert user_message == "partial reply"
    assert [m["role"] for m in history] == ["user"]


def test_system_only_request_has_no_visible_user_turn():
    _, conversation, err = _parse_chat_messages([{"role": "system", "content": "sys"}])
    assert err is None
    user_message, history = _select_current_turn(conversation)
    assert user_message is None
    assert history is None


def test_tool_only_request_has_no_visible_user_turn():
    _, conversation, err = _parse_chat_messages(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "fc_1", "type": "function", "function": {"name": "x", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "fc_1", "content": "result"},
        ]
    )
    assert err is None
    user_message, _ = _select_current_turn(conversation)
    assert user_message is None


def test_empty_user_content_with_tool_tail_still_400s():
    """A user row with no visible payload is not a turn, even before tool rows."""
    _, conversation, err = _parse_chat_messages(
        [
            {"role": "user", "content": ""},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "fc_1", "type": "function", "function": {"name": "x", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "fc_1", "content": "result"},
        ]
    )
    assert err is None
    user_message, _ = _select_current_turn(conversation)
    assert user_message is None
