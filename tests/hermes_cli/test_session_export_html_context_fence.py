from hermes_cli.session_export_html import (
    _generate_messages_html,
    _sanitize_message_content_for_html,
    generate_html_export,
)


def test_html_export_fences_persisted_reasoning_context():
    private = "<memory-context>\nPRIVATE_HTML_REASONING_81312\n</memory-context>"

    rendered = _generate_messages_html(
        [
            {
                "role": "assistant",
                "content": "visible answer",
                "reasoning": private,
            }
        ]
    )

    assert "visible answer" in rendered
    assert "PRIVATE_HTML_REASONING_81312" not in rendered
    assert "memory-context" not in rendered


def test_html_export_projects_tool_arguments_without_mutating_source():
    raw_arguments = '{"query":"<memory-context>PRIVATE_HTML_ARG</memory-context>visible"}'
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "web_search",
                    "arguments": raw_arguments,
                }
            }
        ],
    }

    rendered = _generate_messages_html([message])

    assert "PRIVATE_HTML_ARG" not in rendered
    assert "memory-context" not in rendered
    assert "visible" in rendered
    assert message["tool_calls"][0]["function"]["arguments"] == raw_arguments


def test_html_export_fails_closed_for_malformed_fenced_tool_arguments():
    rendered = _generate_messages_html(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "web_search",
                            "arguments": '{"query":"<memory-context>PRIVATE_MALFORMED',
                        }
                    }
                ],
            }
        ]
    )

    assert "PRIVATE_MALFORMED" not in rendered
    assert "memory-context" not in rendered
    assert "query" not in rendered


def test_html_export_fences_assistant_and_tool_content():
    private = "<memory-context>PRIVATE_HTML_CONTENT</memory-context>"

    rendered = _generate_messages_html(
        [
            {"role": "assistant", "content": f"visible {private} tail"},
            {"role": "tool", "content": f"tool {private} result"},
        ]
    )

    assert "PRIVATE_HTML_CONTENT" not in rendered
    assert "memory-context" not in rendered
    assert "visible  tail" in rendered
    assert "tool  result" in rendered


def test_html_export_fails_closed_for_sentence_shaped_user_close_reopen():
    private = (
        "Visible </memory-context>INJECTED<memory-context>"
        " PRIVATE HTML payload leaked."
    )

    rendered = _generate_messages_html([{"role": "user", "content": private}])

    assert "Visible" in rendered
    assert "INJECTED" not in rendered
    assert "PRIVATE" not in rendered


def test_html_export_fails_closed_for_user_close_reopen_beyond_inline_cap():
    private = (
        "Visible </memory-context>PRIVATE_HTML_CAP"
        + ("x" * 513)
        + "<memory-context>hidden</memory-context> tail"
    )

    rendered = _generate_messages_html([{"role": "user", "content": private}])

    assert "Visible" in rendered
    assert "tail" in rendered
    assert "PRIVATE_HTML_CAP" not in rendered


def test_html_export_fences_structured_content_recursively():
    rendered = _generate_messages_html(
        [
            {
                "role": "assistant",
                "content": {
                    "type": "text",
                    "text": "<memory-context>PRIVATE_HTML_STRUCTURED</memory-context>",
                },
            }
        ]
    )

    assert "PRIVATE_HTML_STRUCTURED" not in rendered
    assert "memory-context" not in rendered


def test_html_export_keeps_distinct_entries_after_key_fencing():
    private_key = "<memory-context>PRIVATE_HTML_KEY</memory-context>"
    projected = _sanitize_message_content_for_html(
        "assistant",
        {private_key: "first", "<memory-context>OTHER_KEY</memory-context>": "second"},
    )

    assert list(projected.values()) == ["first", "second"]
    assert all("memory-context" not in str(key) for key in projected)


def test_html_export_fences_persisted_session_title():
    private = "<memory-context>PRIVATE_TITLE_81312</memory-context>Visible title"

    rendered = generate_html_export(
        {"id": "session-title", "title": private, "messages": []}
    )

    assert "PRIVATE_TITLE_81312" not in rendered
    assert "memory-context" not in rendered
    assert "Visible title" in rendered


def test_html_export_fences_persisted_system_prompt():
    private = "<memory-context>PRIVATE_SYSTEM_PROMPT_81312</memory-context>visible persona"

    rendered = generate_html_export(
        {
            "id": "session-system-prompt",
            "title": "Visible title",
            "system_prompt": private,
            "messages": [],
        }
    )

    assert "PRIVATE_SYSTEM_PROMPT_81312" not in rendered
    assert "memory-context" not in rendered
    assert "visible persona" in rendered
