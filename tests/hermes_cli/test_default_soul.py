"""Tests for the default SOUL template."""

from hermes_cli.default_soul import DEFAULT_SOUL_MD


def test_default_soul_marks_optional_memory_layers_as_optional():
    assert "up to four layers" in DEFAULT_SOUL_MD
    assert "optional large on-demand memory when configured" in DEFAULT_SOUL_MD
    assert "optional searchable archive of past conversations when the tool is available" in DEFAULT_SOUL_MD