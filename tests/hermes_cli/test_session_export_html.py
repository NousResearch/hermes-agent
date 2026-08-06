"""Tests for the HTML session export renderer."""

from hermes_cli.session_export_html import (
    _generate_messages_html,
    generate_multi_session_html_export,
)


def test_tool_call_name_is_escaped():
    """A tool-call name is attacker-influenced (a prompt-injected model can emit
    an arbitrary name), so it must be HTML-escaped like every sibling field."""
    payload = '<img src=x onerror="alert(1)">'
    html = _generate_messages_html([
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": payload, "arguments": "{}"},
                }
            ],
        }
    ])
    assert payload not in html
    assert "&lt;img src=x onerror=" in html


def test_tool_call_arguments_stay_escaped():
    html = _generate_messages_html([
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "<b>x</b>"},
                }
            ],
        }
    ])
    assert "&lt;b&gt;x&lt;/b&gt;" in html
    assert "<b>x</b>" not in html


def test_multi_session_export_keeps_switcher_script():
    """The multi-session export drives session switching with an inline script,
    so the escaping fix must not remove or block that script."""
    sessions = [
        {"id": "aaaa1111", "title": "First", "started_at": 0,
         "messages": [{"role": "user", "content": "one"}]},
        {"id": "bbbb2222", "title": "Second", "started_at": 0,
         "messages": [{"role": "user", "content": "two"}]},
    ]
    html = generate_multi_session_html_export(sessions)
    assert "function showSession" in html
    assert 'data-id="aaaa1111"' in html
    assert 'id="view-bbbb2222"' in html
