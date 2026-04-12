from hermes_cli.main import _prompt_reasoning_effort_selection


def test_reasoning_menu_orders_minimal_before_low(monkeypatch):
    captured = {}

    def _fake_single_select(title, items, default_index=0, *, cancel_label="Cancel"):
        captured["title"] = title
        captured["items"] = items
        captured["default_index"] = default_index
        captured["cancel_label"] = cancel_label
        return default_index

    monkeypatch.setattr("hermes_cli.curses_ui.curses_single_select", _fake_single_select)

    selected = _prompt_reasoning_effort_selection(
        ["low", "minimal", "medium", "high"],
        current_effort="medium",
    )

    assert selected == "medium"
    assert captured["title"] == "Select reasoning effort:"
    assert captured["items"] == [
        "minimal",
        "low",
        "medium  ← currently in use",
        "high",
        "Disable reasoning",
    ]
    assert captured["default_index"] == 2
    assert captured["cancel_label"] == "Skip (keep current)"
