"""Tests for hermes_cli/_parser.py — CLI argument parser helpers."""


def test_import_works():
    from hermes_cli._parser import build_top_level_parser
    parser = build_top_level_parser()
    assert parser is not None


def test_parser_has_help():
    from hermes_cli._parser import build_top_level_parser
    parser = build_top_level_parser()
    help_text = parser.format_help()
    assert len(help_text) > 0
    assert "hermes" in help_text.lower()
