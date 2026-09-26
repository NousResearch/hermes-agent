from hermes_cli.main_provider_setup import _prompt_reasoning_effort_selection
from hermes_cli.setup import _current_reasoning_effort


def test_reasoning_menu_orders_minimal_before_low(monkeypatch):
    captured = {}

    def _fake_radiolist(title, items, *, selected=0, cancel_returns=None, description=None):
        captured["items"] = items
        captured["selected"] = selected
        return selected  # pick the pre-selected (current) entry

    monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", _fake_radiolist)

    selected = _prompt_reasoning_effort_selection(
        ["low", "minimal", "medium", "high"],
        current_effort="medium",
    )

    assert selected == "medium"
    assert [item.split()[0] for item in captured["items"][:4]] == [
        "minimal",
        "low",
        "medium",
        "high",
    ]


def test_current_reasoning_effort_reads_dict_form():
    """The setup wizard's "currently in use" lookup must see the dict form's tier (or `none`
    when it disables thinking), never `str(dict)`."""
    assert _current_reasoning_effort({"agent": {"reasoning_effort": {"enabled": True, "effort": "Thinking"}}}) == "thinking"
    assert _current_reasoning_effort({"agent": {"reasoning_effort": {"enabled": False}}}) == "none"
    assert _current_reasoning_effort({"agent": {"reasoning_effort": "high"}}) == "high"
