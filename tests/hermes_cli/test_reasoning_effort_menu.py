import sys
import types


from hermes_cli.main import _prompt_reasoning_effort_selection


class _FakeTerminalMenu:
    last_choices = None

    def __init__(self, choices, **kwargs):
        _FakeTerminalMenu.last_choices = choices
        self._cursor_index = kwargs.get("cursor_index")

    def show(self):
        return self._cursor_index


def test_reasoning_menu_orders_minimal_before_low(monkeypatch):
    fake_module = types.SimpleNamespace(TerminalMenu=_FakeTerminalMenu)
    monkeypatch.setitem(sys.modules, "simple_term_menu", fake_module)

    selected = _prompt_reasoning_effort_selection(
        ["low", "minimal", "medium", "high"],
        current_effort="medium",
    )

    assert selected == "medium"
    assert _FakeTerminalMenu.last_choices[:4] == [
        "  minimal",
        "  low",
        "  medium  ← currently in use",
        "  high",
    ]
