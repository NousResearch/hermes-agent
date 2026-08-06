"""Regression tests for None-dereference guards on ``.get(key, "").method()``
patterns (#55997 salvage + config-derived sibling sites).

``dict.get(key, default)`` only returns the default when the key is ABSENT;
a present-but-null value sails through as None and crashes any chained
method call.
"""

from agent.anthropic_adapter import _convert_user_message
from agent.moa_loop import _slot_label


class TestAnthropicNullTextBlock:
    def test_null_text_block_treated_as_empty(self):
        """A text block with ``text: null`` must not crash the empty-message
        check in _convert_user_message (#55997)."""
        result = _convert_user_message([{"type": "text", "text": None}])
        # Must not raise; result is a valid user message dict
        assert result["role"] == "user"

    def test_mixed_null_and_real_text_blocks(self):
        result = _convert_user_message(
            [
                {"type": "text", "text": None},
                {"type": "text", "text": "hello"},
            ]
        )
        assert result["role"] == "user"


class TestMoaSlotLabelNullFields:
    def test_null_provider_and_model(self):
        """MoA slot with ``provider: null`` in config.yaml must not crash
        label construction."""
        assert _slot_label({"provider": None, "model": None}) == ":"  # type: ignore[arg-type]

    def test_normal_slot(self):
        assert _slot_label({"provider": "openrouter", "model": "m"}) == "openrouter:m"
