"""Unit tests for the _hex_to_truecolor helper in cli.py.

Guards against the TypeError edge case where a user's skin YAML has a
non-string color value (e.g. ``banner_text: 12345``) which would crash
the streaming hot path if the helper attempted to slice the raw int.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cli import _hex_to_truecolor


class TestValidInputs:
    def test_valid_rrggbb_fg(self):
        assert _hex_to_truecolor("#FFD700", "fg") == "38;2;255;215;0"

    def test_valid_rrggbb_bg(self):
        assert _hex_to_truecolor("#FFD700", "bg") == "48;2;255;215;0"

    def test_default_layer_is_fg(self):
        assert _hex_to_truecolor("#FFD700") == "38;2;255;215;0"

    def test_lowercase_valid(self):
        assert _hex_to_truecolor("#aabbcc", "fg") == "38;2;170;187;204"

    def test_mixed_case(self):
        assert _hex_to_truecolor("#AaBbCc", "fg") == "38;2;170;187;204"

    def test_black(self):
        assert _hex_to_truecolor("#000000", "fg") == "38;2;0;0;0"

    def test_white(self):
        assert _hex_to_truecolor("#FFFFFF", "bg") == "48;2;255;255;255"


class TestMalformedInputs:
    def test_empty_string(self):
        assert _hex_to_truecolor("", "fg") == ""

    def test_none(self):
        # Non-string input must not raise TypeError — this is the
        # regression guard for the original bug.
        assert _hex_to_truecolor(None, "fg") == ""  # type: ignore[arg-type]

    def test_integer(self):
        # Exact footgun from the code-review comment: a user's skin
        # YAML with ``banner_text: 12345`` should degrade gracefully,
        # not crash with TypeError inside the streaming hot path.
        assert _hex_to_truecolor(12345, "fg") == ""  # type: ignore[arg-type]

    def test_short_form(self):
        # #ABC style — unsupported by the helper, should degrade.
        assert _hex_to_truecolor("#ABC", "fg") == ""

    def test_garbage_word(self):
        assert _hex_to_truecolor("red", "fg") == ""

    def test_missing_hash_prefix_but_valid_hex(self):
        # Without the leading '#', slicing [1:3][3:5][5:7] off a 6-char
        # string leaves the final component empty → ValueError → "".
        assert _hex_to_truecolor("FFD700", "fg") == ""

    def test_too_short(self):
        assert _hex_to_truecolor("#FFD70", "fg") == ""

    def test_non_hex_chars(self):
        assert _hex_to_truecolor("#ZZZZZZ", "fg") == ""

    def test_list_input(self):
        assert _hex_to_truecolor([1, 2, 3], "fg") == ""  # type: ignore[arg-type]

    def test_dict_input(self):
        assert _hex_to_truecolor({"r": 1}, "fg") == ""  # type: ignore[arg-type]


class TestLayerSelection:
    def test_bg_layer_uses_48(self):
        out = _hex_to_truecolor("#102030", "bg")
        assert out.startswith("48;2;")

    def test_fg_layer_uses_38(self):
        out = _hex_to_truecolor("#102030", "fg")
        assert out.startswith("38;2;")

    def test_unknown_layer_defaults_to_bg(self):
        # Current contract: anything other than "fg" becomes the bg code.
        # This guards the behavior so callers know what they get.
        out = _hex_to_truecolor("#102030", "something-else")
        assert out.startswith("48;2;")


@pytest.mark.parametrize(
    "hex_str,expected",
    [
        ("#000000", "38;2;0;0;0"),
        ("#FF0000", "38;2;255;0;0"),
        ("#00FF00", "38;2;0;255;0"),
        ("#0000FF", "38;2;0;0;255"),
        ("#808080", "38;2;128;128;128"),
    ],
)
def test_valid_fg_parametrized(hex_str, expected):
    assert _hex_to_truecolor(hex_str, "fg") == expected