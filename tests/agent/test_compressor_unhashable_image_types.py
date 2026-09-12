"""Regression tests for non-scalar content-part types."""

from agent.context_compressor import (
    ContextCompressor,
    _content_has_images,
)


def test_non_scalar_types_are_safe_and_not_images():
    for value in ({"type": {"type": "string"}}, {"type": ["string", "null"]}):
        content = [value]
        assert _content_has_images(content) is False

        compressor = ContextCompressor.__new__(ContextCompressor)
        serialized = compressor._serialize_for_summary([{"role": "user", "content": content}])
        assert serialized.startswith("[USER]: ")


def test_scalar_content_types_keep_existing_behavior():
    compressor = ContextCompressor.__new__(ContextCompressor)
    assert _content_has_images([{"type": "image_url", "image_url": {"url": "https://example.test/x"}}]) is True
    assert _content_has_images([{"type": "text", "text": "hello"}]) is False
    serialized = compressor._serialize_for_summary([{"role": "user", "content": [{"type": "text", "text": "hello"}]}])
    assert "hello" in serialized
