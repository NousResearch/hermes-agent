from hermes_cli.main import _prompt_reasoning_effort_selection


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
    assert captured["items"][:4] == [
        "minimal",
        "low",
        "medium  ← currently in use",
        "high",
    ]
