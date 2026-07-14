"""Regression tests for clarify_tool._flatten_choice (issue #62146).

Some models (GLM-5.2 via xiaomi) emit choices as [{"value": "A. Option text"}]
instead of the canonical label/text keys. _flatten_choice must surface the
human-readable text from `value` as a last-resort fallback without breaking
models that use the canonical keys.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.clarify_tool import _flatten_choice


def test_canonical_label_wins_over_value():
    assert _flatten_choice({"label": "L", "value": "V"}) == "L"


def test_value_fallback_for_glm_style_choices():
    # The reported bug: GLM-5.2 emits value-only choice dicts.
    assert _flatten_choice({"value": "A. Option text"}) == "A. Option text"


def test_bare_string_passthrough():
    assert _flatten_choice("plain") == "plain"


def test_unknown_key_dropped():
    assert _flatten_choice({"unknown": "x"}) == ""


def test_mixed_list_flattened():
    assert _flatten_choice([{"value": "X"}, "Y"]) == "X Y"


def test_none_dropped():
    assert _flatten_choice(None) == ""
