"""Regression tests confirming the setup model/provider/reasoning pickers route
through the shared curses radiolist (ESC + arrow-key handling that works across
terminals, incl. Ghostty) instead of simple_term_menu.

Guards against silently regressing back to simple_term_menu, whose ESC/arrow
handling was unreliable in `hermes setup` (the provider->model sub-menu).
"""
from unittest.mock import patch


def test_prompt_model_selection_uses_curses_radiolist():
    from hermes_cli.auth import _prompt_model_selection
    from hermes_cli.curses_ui import radio_item_plain

    seen = {}

    def _fake(title, items, *, selected=0, cancel_returns=None, description=None, searchable=False):
        seen["title"] = title
        seen["items"] = items
        return 1  # pick second model

    with patch("hermes_cli.curses_ui.curses_radiolist", side_effect=_fake), \
         patch("builtins.print"):
        result = _prompt_model_selection(["model-a", "model-b"])

    assert result == "model-b"
    assert seen["title"] == "Select default model:"
    # Items are the models plus the custom/skip entries. Model rows may be
    # rich (text, style) segments for sale chrome — compare the plain text.
    plain = [radio_item_plain(item) for item in seen["items"]]
    assert plain[:2] == ["model-a", "model-b"]
    assert "Skip (keep current)" in plain


def test_prompt_model_selection_esc_cancels():
    from hermes_cli.auth import _prompt_model_selection

    # curses_radiolist returns the cancel sentinel (-1) on ESC.
    with patch("hermes_cli.curses_ui.curses_radiolist", return_value=-1), \
         patch("builtins.print"):
        result = _prompt_model_selection(["model-a", "model-b"])

    assert result is None


def test_reasoning_effort_uses_curses_radiolist():
    from hermes_cli.main import _prompt_reasoning_effort_selection

    with patch("hermes_cli.curses_ui.curses_radiolist", return_value=2), \
         patch("builtins.print"):
        result = _prompt_reasoning_effort_selection(["low", "medium", "high"], current_effort="")

    assert result == "high"


def test_reasoning_effort_esc_cancels():
    from hermes_cli.main import _prompt_reasoning_effort_selection

    with patch("hermes_cli.curses_ui.curses_radiolist", return_value=-1), \
         patch("builtins.print"):
        result = _prompt_reasoning_effort_selection(["low", "medium", "high"], current_effort="")

    assert result is None


def test_model_selection_with_pricing_passes_description():
    """When pricing is supplied, the aligned header is passed as the curses
    description (multi-line text above the list), not lost."""
    from hermes_cli.auth import _prompt_model_selection

    seen = {}

    def _fake(title, items, *, selected=0, cancel_returns=None, description=None, searchable=False):
        seen["description"] = description
        return len(items) - 1  # Skip

    pricing = {
        "model-a": {"prompt": "0.000001", "completion": "0.000002"},
        "model-b": {"prompt": "0.000003", "completion": "0.000004"},
    }
    with patch("hermes_cli.curses_ui.curses_radiolist", side_effect=_fake), \
         patch("builtins.print"):
        _prompt_model_selection(["model-a", "model-b"], pricing=pricing)

    # The description should carry the In/Out price header.
    assert seen["description"] is not None
    assert "In" in seen["description"] and "Out" in seen["description"]
