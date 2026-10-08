"""Durable transcript projection must not store a blank row for screenshot tool results."""

from agent.session_persistence import _durable_content


def test_user_multimodal_list_still_projects_text_and_placeholder():
    content = [
        {"type": "text", "text": "Describe this screenshot"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    assert _durable_content(content) == "Describe this screenshot\n[screenshot]"


def test_image_only_list_does_not_persist_empty():
    content = [
        {"type": "text", "text": ""},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    assert _durable_content(content) == "[screenshot]"


def test_multimodal_envelope_empty_summary_does_not_persist_empty():
    content = {
        "_multimodal": True,
        "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}],
        "text_summary": "",
    }
    stored = _durable_content(content)
    assert stored
    assert stored.strip()


def test_plain_string_passthrough():
    assert _durable_content("hello") == "hello"
    assert _durable_content("") == ""
