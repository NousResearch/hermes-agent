"""Regression tests for memory/plugin context injection into multimodal user messages.

The injection block in ``run_conversation`` (``agent/conversation_loop.py``)
was gated on ``isinstance(_base, str)``, which silently dropped external memory
prefetch results and ``pre_llm_call`` plugin context whenever the current user
message carried multimodal content (a list of blocks).

The fix adds an ``elif isinstance(_base, list)`` branch that prepends the
injected context as a leading ``{"type": "text", ...}`` block so vision-capable
providers receive the memory context alongside the image parts.

These tests exercise the fixed injection logic in isolation so the full
``run_conversation`` machinery is not required.
"""

from agent.memory_manager import build_memory_context_block


def _simulate_injection(base_content, raw_memory_context: str):
    """Replicate the injection block from conversation_loop.py.

    Returns the api_msg["content"] value after applying all injections.
    """
    _fenced = build_memory_context_block(raw_memory_context)
    _injections = [_fenced] if _fenced else []
    if not _injections:
        return base_content

    _base = base_content
    if isinstance(_base, str):
        return _base + "\n\n" + "\n\n".join(_injections)
    elif isinstance(_base, list):
        injection_text = "\n\n".join(_injections)
        return [{"type": "text", "text": injection_text}, *_base]
    return _base


class TestMemoryInjectMultimodal:
    """Memory context injection reaches the model for multimodal user messages."""

    def test_string_content_injection_unchanged(self):
        result = _simulate_injection("What is 2+2?", "User prefers concise answers.")
        assert isinstance(result, str)
        assert "What is 2+2?" in result
        assert "User prefers concise answers." in result

    def test_list_content_injection_prepends_text_block(self):
        """External memory must not be silently dropped for multimodal messages."""
        base = [
            {"type": "text", "text": "What do you see?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,ABC=="}},
        ]
        result = _simulate_injection(base, "User prefers Python over JavaScript.")
        assert isinstance(result, list), "result must stay a list for multimodal providers"
        # Injected context arrives as the first block
        assert result[0]["type"] == "text"
        assert "User prefers Python" in result[0]["text"]

    def test_list_content_original_blocks_preserved(self):
        """Image parts and original text must survive injection."""
        base = [
            {"type": "text", "text": "Analyze this"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        ]
        result = _simulate_injection(base, "context")
        # Image part still present
        image_parts = [p for p in result if p.get("type") == "image_url"]
        assert len(image_parts) == 1
        assert image_parts[0]["image_url"]["url"] == "https://example.com/img.png"
        # Original text still present
        text_parts = [p for p in result if p.get("type") == "text" and "Analyze" in p.get("text", "")]
        assert len(text_parts) == 1

    def test_empty_memory_context_skips_injection(self):
        """No injection should happen when the memory context is empty."""
        base = [
            {"type": "text", "text": "Hello"},
            {"type": "image_url", "image_url": {"url": "x"}},
        ]
        result = _simulate_injection(base, "")
        # Content must be unchanged when there's nothing to inject
        assert result is base

    def test_list_length_increases_by_one(self):
        """Exactly one prepended text block — no duplicate or missing parts."""
        base = [
            {"type": "text", "text": "Describe the chart"},
            {"type": "image_url", "image_url": {"url": "https://example.com/chart.png"}},
        ]
        result = _simulate_injection(base, "User works in finance.")
        assert len(result) == len(base) + 1
