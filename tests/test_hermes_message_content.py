from hermes_message_content import (
    STRUCTURED_CONTENT_FORMAT,
    content_to_text,
    deserialize_message_content,
    normalize_replay_content,
    serialize_message_content,
)


def test_content_to_text_redacts_inline_attachment_payloads():
    content = [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "DATA:image/png;base64,raw-image"}},
        {"type": "input_image", "source": {"type": "base64", "data": "raw-input-image"}},
        {"type": "file", "file": {"filename": "notes.pdf", "file_data": "raw-file"}},
        {"type": "document", "source": {"type": "base64", "data": "raw-document"}},
    ]

    text = content_to_text(content)

    assert text == (
        "look\n[image attachment]\n[image attachment]\n"
        "[file attachment]\n[attachment]"
    )
    assert "raw-image" not in text
    assert "raw-input-image" not in text
    assert "raw-file" not in text
    assert "raw-document" not in text


def test_content_to_text_supports_compact_cli_style():
    content = [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "https://example.com/p.png"}},
        {"type": "file", "file": {"filename": "notes.pdf"}},
    ]

    assert content_to_text(content, separator=" ", attachment_style="compact") == "look [image] [file]"


def test_structured_content_serializes_display_text_and_replay_payload():
    content = [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "DATA:image/png;base64,raw-image"}},
    ]

    text, content_format, payload = serialize_message_content(content)

    assert text == "look\n[image attachment]"
    assert content_format == STRUCTURED_CONTENT_FORMAT
    assert "raw-image" not in text
    assert "raw-image" in payload
    assert deserialize_message_content(text, content_format, payload) == content


def test_normalize_replay_content_wraps_single_content_part():
    content = {
        "type": "document",
        "source": {"type": "base64", "data": "raw-document"},
    }

    assert normalize_replay_content(content) == [content]


def test_normalize_replay_content_falls_back_for_non_content_dict():
    assert normalize_replay_content({"metadata": {"source": "integration"}}) == "[structured content]"


def test_normalize_replay_content_sanitizes_memory_context():
    content = [{"type": "text", "text": "before <memory-context>secret</memory-context> after"}]

    assert normalize_replay_content(content) == [{"type": "text", "text": "before  after"}]
