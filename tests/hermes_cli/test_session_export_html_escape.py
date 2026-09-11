import re

from hermes_cli.session_export_html import _generate_messages_html


def test_tool_call_name_is_escaped_in_html_export():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "timestamp": 1700000000,
            "tool_calls": [
                {
                    "function": {
                        "name": "<script>alert(1)</script>",
                        "arguments": "{}",
                    }
                }
            ],
        }
    ]

    html = _generate_messages_html(messages)

    # Raw, executable markup must never reach the standalone artifact.
    assert "<script>alert(1)</script>" not in html
    # The escaped form must be present instead.
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_role_is_escaped_in_html_export():
    messages = [
        {
            "role": "<img src=x onerror=alert(document.domain)>",
            "content": "hello",
            "timestamp": 1700000000,
        }
    ]

    html = _generate_messages_html(messages)

    assert "<img src=x onerror=alert(document.domain)>" not in html
    assert "&lt;img src=x onerror=alert(document.domain)&gt;" in html
    # The class attribute must remain a single, well-formed token: a crafted
    # role must not break out of it nor split into several unintended classes.
    class_value = re.search(r'class="(message message-[^"]*active)"', html)
    assert class_value is not None
    assert " message-" in class_value.group(1)  # exactly one message-<role> class
    assert class_value.group(1).count("message-") == 1


def test_known_role_keeps_its_css_class():
    html = _generate_messages_html(
        [{"role": "assistant", "content": "hi", "timestamp": 1700000000}]
    )
    assert 'class="message message-assistant active"' in html
