"""Tests for is_truthy_value() config boolean parsing.

Verifies that string "false", "0", "off", "" are treated as False
when used in config paths, instead of bool() which treats any
non-empty string as True.
"""
import pytest
from utils import is_truthy_value


class TestIsTruthyValue:
    def test_boolean_true(self):
        assert is_truthy_value(True) is True

    def test_boolean_false(self):
        assert is_truthy_value(False) is False

    def test_string_true(self):
        assert is_truthy_value("true") is True
        assert is_truthy_value("True") is True
        assert is_truthy_value("TRUE") is True

    def test_string_false(self):
        assert is_truthy_value("false") is False
        assert is_truthy_value("False") is False
        assert is_truthy_value("FALSE") is False

    def test_string_zero(self):
        assert is_truthy_value("0") is False

    def test_string_one(self):
        assert is_truthy_value("1") is True

    def test_string_off(self):
        assert is_truthy_value("off") is False
        assert is_truthy_value("Off") is False

    def test_string_on(self):
        assert is_truthy_value("on") is True

    def test_string_no(self):
        assert is_truthy_value("no") is False

    def test_string_yes(self):
        assert is_truthy_value("yes") is True

    def test_empty_string(self):
        assert is_truthy_value("") is False

    def test_none_with_default_false(self):
        assert is_truthy_value(None, default=False) is False

    def test_none_with_default_true(self):
        assert is_truthy_value(None, default=True) is True

    def test_int_zero(self):
        assert is_truthy_value(0) is False

    def test_int_one(self):
        assert is_truthy_value(1) is True

    def test_bool_vs_is_truthy_divergence(self):
        """The core bug: bool('false') is True, but is_truthy_value('false') is False."""
        assert bool("false") is True  # the bug
        assert is_truthy_value("false") is False  # the fix

    def test_bool_vs_is_truthy_zero_string(self):
        """bool('0') is True, but is_truthy_value('0') is False."""
        assert bool("0") is True  # the bug
        assert is_truthy_value("0") is False  # the fix
