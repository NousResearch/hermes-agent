"""Tests for hermes_cli/portal_cli.py — Portal CLI auth helpers."""


def test_parse_token_simple():
    from hermes_cli.portal_cli import _parse_portal_token
    result = _parse_portal_token("tok_abc123")
    assert result == "tok_abc123"


def test_parse_token_from_env_style():
    from hermes_cli.portal_cli import _parse_portal_token
    result = _parse_portal_token("PORTAL_TOKEN=tok_xyz")
    assert result == "tok_xyz"


def test_parse_token_strips_whitespace():
    from hermes_cli.portal_cli import _parse_portal_token
    assert _parse_portal_token("  tok_space  ") == "tok_space"
