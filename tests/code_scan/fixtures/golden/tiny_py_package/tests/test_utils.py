"""Tests for tiny package."""
from tiny_pkg.utils import format_msg


def test_format_msg_default():
    assert format_msg() == "Hello, world!"


def test_format_msg_custom():
    assert format_msg("Alice") == "Hello, Alice!"
