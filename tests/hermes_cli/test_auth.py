"""Tests for hermes_cli/auth.py — auth and model resolution helpers."""


def test_resolve_model_returns_string():
    from hermes_cli.auth import resolve_model
    result = resolve_model("gpt-4")
    assert isinstance(result, str)


def test_resolve_model_empty_returns_empty():
    from hermes_cli.auth import resolve_model
    result = resolve_model("")
    assert result == ""
