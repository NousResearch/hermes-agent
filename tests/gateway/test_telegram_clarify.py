"""Tests for gateway.platforms.telegram._clarify_choice_to_text.

The helper exists to defend against the model hallucinating structured
``choices`` (dicts / lists) for the ``clarify`` tool, which would otherwise
be rendered as raw ``repr()`` in the Telegram chat.  These tests pin the
behaviour so the fix doesn't regress.
"""

from __future__ import annotations

import pytest

from gateway.platforms.telegram import _clarify_choice_to_text


# ---------------------------------------------------------------------------
# Plain strings — the documented happy path
# ---------------------------------------------------------------------------


class TestPlainStrings:
    def test_simple_string(self):
        assert _clarify_choice_to_text("Yes") == "Yes"

    def test_empty_string(self):
        assert _clarify_choice_to_text("") == ""

    def test_unicode_string(self):
        assert _clarify_choice_to_text("繼續") == "繼續"

    def test_long_string_is_preserved(self):
        s = "x" * 1000
        assert _clarify_choice_to_text(s) == s


# ---------------------------------------------------------------------------
# Dict shapes — the actual bug we hit
# ---------------------------------------------------------------------------


class TestDictShapes:
    def test_item_with_list(self):
        # The exact shape that caused the bug in the wild
        assert _clarify_choice_to_text({"item": ["Yes"]}) == "Yes"

    def test_item_with_string(self):
        assert _clarify_choice_to_text({"item": "No"}) == "No"

    def test_label_key(self):
        assert _clarify_choice_to_text({"label": "Maybe", "id": 2}) == "Maybe"

    def test_text_key(self):
        assert _clarify_choice_to_text({"text": "Continue"}) == "Continue"

    def test_name_key(self):
        assert _clarify_choice_to_text({"name": "Stop"}) == "Stop"

    def test_title_key(self):
        assert _clarify_choice_to_text({"title": "Retry"}) == "Retry"

    def test_dict_with_multiple_recognised_keys_prefers_first(self):
        # ``item`` wins over ``label`` because it's first in the priority
        # order (most common in the wild).
        assert (
            _clarify_choice_to_text({"item": ["First"], "label": "Second"})
            == "First"
        )

    def test_dict_with_no_recognised_key_falls_back_to_first_string_value(self):
        assert _clarify_choice_to_text({"foo": "bar", "baz": "qux"}) == "bar"

    def test_dict_with_only_non_string_values_falls_back_to_str(self):
        # Should not raise; should produce a string
        result = _clarify_choice_to_text({"a": 1, "b": 2})
        assert isinstance(result, str)
        assert result  # non-empty

    def test_empty_dict_falls_back_to_str(self):
        result = _clarify_choice_to_text({})
        assert isinstance(result, str)
        assert result == "{}"


# ---------------------------------------------------------------------------
# List / tuple shapes
# ---------------------------------------------------------------------------


class TestListShapes:
    def test_list_of_strings(self):
        assert _clarify_choice_to_text(["First"]) == "First"

    def test_list_of_dicts(self):
        assert _clarify_choice_to_text([{"item": ["Yes"]}]) == "Yes"

    def test_tuple_of_strings(self):
        assert _clarify_choice_to_text(("First",)) == "First"

    def test_empty_list_falls_back_to_str(self):
        result = _clarify_choice_to_text([])
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_none(self):
        assert _clarify_choice_to_text(None) == "None"

    def test_int(self):
        assert _clarify_choice_to_text(42) == "42"

    def test_nested_dict_in_list(self):
        # The wildest shape we've seen: nested wrapper around a dict
        assert (
            _clarify_choice_to_text([{"item": {"label": "Deep"}}])
            == "Deep"
        )
