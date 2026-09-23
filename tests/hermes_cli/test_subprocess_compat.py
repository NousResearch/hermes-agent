"""Tests for hermes_cli/_subprocess_compat.py — subprocess compatibility helpers."""


def test_hidden_child_options_windows():
    from hermes_cli._subprocess_compat import hidden_child_windows_options
    result = hidden_child_windows_options({}, is_windows=True)
    assert isinstance(result, dict)


def test_hidden_child_options_noop_on_non_windows():
    from hermes_cli._subprocess_compat import hidden_child_windows_options
    result = hidden_child_windows_options({"env": {}}, is_windows=False)
    assert result == {"env": {}}
