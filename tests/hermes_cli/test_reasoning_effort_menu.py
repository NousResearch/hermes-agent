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


def test_reasoning_menu_can_hide_disable_for_mandatory_models(monkeypatch):
    captured = {}

    def _fake_radiolist(title, items, *, selected=0, cancel_returns=None, description=None):
        captured["items"] = items
        return len(items) - 1

    monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", _fake_radiolist)

    assert _prompt_reasoning_effort_selection(
        ["low", "medium", "high"], allow_disable=False
    ) is None
    assert captured["items"] == ["low", "medium", "high", "Skip (keep current)"]
