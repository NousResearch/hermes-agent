"""Tests for YAML boolean coercion in gateway config fields.

YAML parses bare ``off`` as boolean ``False``, not the string ``"off"``.
Config fields that expect string values like ``"off"`` must coerce
booleans to avoid silent misinterpretation.
"""

from gateway.config import StreamingConfig, PlatformConfig


class TestStreamingTransportBoolean:
    """StreamingConfig.from_dict must coerce boolean transport values."""

    def test_bare_off_parsed_as_false_becomes_off_string(self):
        """YAML ``transport: off`` → Python ``False`` → should be ``"off"``."""
        config = StreamingConfig.from_dict({"enabled": True, "transport": False})
        assert config.transport == "off"

    def test_bare_on_parsed_as_true_becomes_edit(self):
        """YAML ``transport: on`` → Python ``True`` → should default to ``"edit"``."""
        config = StreamingConfig.from_dict({"enabled": True, "transport": True})
        assert config.transport == "edit"

    def test_string_off_preserved(self):
        """Quoted ``transport: "off"`` stays as ``"off"``."""
        config = StreamingConfig.from_dict({"enabled": True, "transport": "off"})
        assert config.transport == "off"

    def test_string_edit_preserved(self):
        config = StreamingConfig.from_dict({"enabled": True, "transport": "edit"})
        assert config.transport == "edit"

    def test_missing_transport_defaults_to_edit(self):
        config = StreamingConfig.from_dict({"enabled": True})
        assert config.transport == "edit"


class TestReplyToModeBoolean:
    """PlatformConfig.from_dict must coerce boolean reply_to_mode values."""

    def test_bare_off_parsed_as_false_becomes_off_string(self):
        """YAML ``reply_to_mode: off`` → Python ``False`` → should be ``"off"``."""
        config = PlatformConfig.from_dict({"reply_to_mode": False})
        assert config.reply_to_mode == "off"

    def test_string_off_preserved(self):
        config = PlatformConfig.from_dict({"reply_to_mode": "off"})
        assert config.reply_to_mode == "off"

    def test_string_first_preserved(self):
        config = PlatformConfig.from_dict({"reply_to_mode": "first"})
        assert config.reply_to_mode == "first"

    def test_string_all_preserved(self):
        config = PlatformConfig.from_dict({"reply_to_mode": "all"})
        assert config.reply_to_mode == "all"

    def test_missing_defaults_to_first(self):
        config = PlatformConfig.from_dict({})
        assert config.reply_to_mode == "first"
