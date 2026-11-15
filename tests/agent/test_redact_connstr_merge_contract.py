"""Assigned connection strings retain the newer template and quote handling."""

import pytest

from agent.redact import redact_terminal_output


@pytest.mark.parametrize("prefix", ["Server=db;", "X=host=db;"])
def test_nested_password_in_template_shaped_value_is_masked(prefix):
    line = prefix + "Password=${a'Password=synthetic_sample}"
    out = redact_terminal_output(line, "cat app.py", force=True)
    assert "synthetic_sample" not in out


@pytest.mark.parametrize("prefix", ["Server=db;", "X=host=db;"])
def test_quote_inside_bare_password_and_wrapper_suffix(prefix):
    line = 'conn = "' + prefix + 'Password=synthetic\'sample"+suffix'
    assert redact_terminal_output(line, "cat app.py", force=True) == (
        'conn = "' + prefix + 'Password=***"+suffix')
