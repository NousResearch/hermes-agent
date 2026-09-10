"""Tests for hermes_cli/focus_view.py — focus mode and tool progress helpers."""


def test_normalize_tool_progress_all():
    from hermes_cli.focus_view import normalize_tool_progress_mode
    assert normalize_tool_progress_mode("all") == "all"
    assert normalize_tool_progress_mode(None) == "all"


def test_normalize_tool_progress_off():
    from hermes_cli.focus_view import normalize_tool_progress_mode
    assert normalize_tool_progress_mode("off") == "off"


def test_resolve_focus_arg_on():
    from hermes_cli.focus_view import resolve_focus_arg
    mode, current = resolve_focus_arg("on", False)
    assert mode == "on"
    assert current is None


def test_effective_tool_progress_mode_hidden_when_off():
    from hermes_cli.focus_view import effective_tool_progress_mode
    assert effective_tool_progress_mode(True, None) in ("all", "tool_names_only", "off")
