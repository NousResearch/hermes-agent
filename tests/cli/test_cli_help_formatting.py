"""Tests for stable, non-wrapped top-level CLI help output."""

from hermes_cli._parser import build_top_level_parser


def test_top_level_help_does_not_insert_terminal_width_wraps():
    parser, _, _ = build_top_level_parser()

    help_text = parser.format_help()

    assert "No banner, no spinner, no tool previews" in help_text
    assert "No banner, no spinner,\n" not in help_text
