from agent.copilot_acp_client import _extract_tool_calls_from_text


def test_extract_tool_calls_from_text_ignores_xml_block_markup() -> None:
    text = (
        'Please review this snippet:\n'
        '<tool_call>{"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\\"path\\":\\"/tmp/x\\"}"}}</tool_call>'
    )

    tool_calls, cleaned = _extract_tool_calls_from_text(text)

    assert tool_calls == []
    assert cleaned == text


def test_extract_tool_calls_from_text_ignores_bare_openai_tool_json() -> None:
    text = (
        '{"id":"call_2","type":"function","function":{"name":"read_file","arguments":"{}"}}'
    )

    tool_calls, cleaned = _extract_tool_calls_from_text(text)

    assert tool_calls == []
    assert cleaned == text


def test_extract_tool_calls_from_text_preserves_text_whitespace_verbatim() -> None:
    text = '\n  {"id":"call_3","type":"function","function":{"name":"read_file","arguments":"{}"}}  \n'

    tool_calls, cleaned = _extract_tool_calls_from_text(text)

    assert tool_calls == []
    assert cleaned == text
