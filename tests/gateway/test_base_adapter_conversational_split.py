"""Tests for the shared conversational-split infrastructure on the base adapter.

The ``split_outgoing_*`` extra keys, the fence-aware blank-line splitter
(``_outgoing_message_parts``), and the ``config.extra`` coercion helpers were
lifted from the Telegram/WhatsApp adapters onto ``BasePlatformAdapter`` so
every platform (Telegram, WhatsApp, Photon/iMessage) shares one
implementation. These tests exercise the shared methods directly against a
minimal concrete subclass, independent of any specific platform.
"""
from typing import Any, Dict, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult


class _StubAdapter(BasePlatformAdapter):
    """Minimal concrete adapter — just enough to instantiate the base class."""

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        pass

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        return SendResult(success=True)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {}


def _make_adapter(**extra) -> _StubAdapter:
    adapter = _StubAdapter(
        PlatformConfig(enabled=True, token="t", extra=dict(extra)),
        Platform.TELEGRAM,
    )
    adapter._init_conversational_split_config()
    return adapter


class TestCoerceBoolExtra:
    def test_missing_key_returns_default(self):
        adapter = _make_adapter()
        assert adapter._coerce_bool_extra("nope", False) is False
        assert adapter._coerce_bool_extra("nope", True) is True

    def test_string_spellings(self):
        for raw in ("true", "1", "yes", "on", " True "):
            assert _make_adapter(k=raw)._coerce_bool_extra("k") is True
        for raw in ("false", "0", "no", "off", " OFF "):
            assert _make_adapter(k=raw)._coerce_bool_extra("k") is False

    def test_unrecognized_string_falls_back_to_default(self):
        assert _make_adapter(k="sometimes")._coerce_bool_extra("k", True) is True
        assert _make_adapter(k="sometimes")._coerce_bool_extra("k", False) is False

    def test_non_string_values_are_boolified(self):
        assert _make_adapter(k=1)._coerce_bool_extra("k") is True
        assert _make_adapter(k=0)._coerce_bool_extra("k", True) is False


class TestCoerceFloatExtra:
    def test_missing_and_unparseable_fall_back_to_default(self):
        assert _make_adapter()._coerce_float_extra("k", 0.6) == 0.6
        assert _make_adapter(k="not-a-number")._coerce_float_extra("k", 0.6) == 0.6

    def test_parses_numeric_strings(self):
        assert _make_adapter(k="1.25")._coerce_float_extra("k", 0.6) == 1.25

    def test_non_finite_values_fall_back_to_default(self):
        assert _make_adapter(k="nan")._coerce_float_extra("k", 0.6) == 0.6
        assert _make_adapter(k="inf")._coerce_float_extra("k", 0.6) == 0.6
        assert (
            _make_adapter(k="nan")._coerce_float_extra("k", 0.6, min_value=0.0) == 0.6
        )

    def test_min_and_max_clamp(self):
        adapter = _make_adapter(k="-5")
        assert adapter._coerce_float_extra("k", 0.6, min_value=0.0) == 0.0
        adapter = _make_adapter(k="900")
        assert (
            adapter._coerce_float_extra("k", 30.0, min_value=1.0, max_value=300.0)
            == 300.0
        )

    def test_negative_without_explicit_floor_falls_back_to_default(self):
        # WhatsApp semantics: delays feed asyncio.sleep(), so a negative
        # value without a declared floor is rejected rather than clamped.
        assert _make_adapter(k="-5")._coerce_float_extra("k", 5.0) == 5.0


class TestInitConversationalSplitConfig:
    def test_defaults_are_opt_out_with_documented_values(self):
        adapter = _make_adapter()
        assert adapter._split_outgoing_on_blank_lines is False
        assert adapter._split_outgoing_delay_seconds == 0.6
        assert adapter._split_outgoing_max_parts == 4

    def test_values_are_config_backed_and_invalid_values_fall_back_safely(self):
        configured = _make_adapter(
            split_outgoing_on_blank_lines="yes",
            split_outgoing_delay_seconds="1.25",
            split_outgoing_max_parts="7",
        )
        invalid = _make_adapter(
            split_outgoing_on_blank_lines="sometimes",
            split_outgoing_delay_seconds="not-a-number",
            split_outgoing_max_parts=-2,
        )
        malformed_max = _make_adapter(split_outgoing_max_parts="many")

        assert configured._split_outgoing_on_blank_lines is True
        assert configured._split_outgoing_delay_seconds == 1.25
        assert configured._split_outgoing_max_parts == 7
        assert invalid._split_outgoing_on_blank_lines is False
        assert invalid._split_outgoing_delay_seconds == 0.6
        assert invalid._split_outgoing_max_parts == 4
        assert malformed_max._split_outgoing_max_parts == 4


class TestOutgoingMessageParts:
    def test_opt_out_returns_input_unchanged(self):
        adapter = _make_adapter()
        assert adapter._outgoing_message_parts("one\n\ntwo") == ["one\n\ntwo"]

    def test_blank_lines_split_but_single_newlines_do_not(self):
        adapter = _make_adapter(split_outgoing_on_blank_lines=True)
        assert adapter._outgoing_message_parts("one\n\ntwo\nthree") == [
            "one",
            "two\nthree",
        ]

    def test_single_paragraph_passes_through_verbatim(self):
        adapter = _make_adapter(split_outgoing_on_blank_lines=True)
        assert adapter._outgoing_message_parts("\none\ntwo\n") == ["\none\ntwo\n"]

    def test_blank_lines_inside_fenced_code_do_not_split(self):
        adapter = _make_adapter(split_outgoing_on_blank_lines=True)
        content = "intro\n\n```python\nfirst\n\nsecond\n```\n\noutro"
        assert adapter._outgoing_message_parts(content) == [
            "intro",
            "```python\nfirst\n\nsecond\n```",
            "outro",
        ]

    def test_max_parts_merges_remainder_into_final_part(self):
        adapter = _make_adapter(
            split_outgoing_on_blank_lines=True, split_outgoing_max_parts=3
        )
        assert adapter._outgoing_message_parts("one\n\ntwo\n\nthree\n\nfour") == [
            "one",
            "two",
            "three\n\nfour",
        ]

    def test_uninitialized_adapter_defaults_to_passthrough(self):
        # send() paths must stay safe for adapters built via __new__ in tests
        # (no __init__, so no _split_outgoing_* attributes).
        adapter = _StubAdapter.__new__(_StubAdapter)
        assert adapter._outgoing_message_parts("one\n\ntwo") == ["one\n\ntwo"]
