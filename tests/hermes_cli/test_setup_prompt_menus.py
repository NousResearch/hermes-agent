from hermes_cli import setup as setup_mod


def test_prompt_choice_uses_curses_helper(monkeypatch):
    monkeypatch.setattr(setup_mod, "_curses_prompt_choice", lambda question, choices, default=0: 1)

    idx = setup_mod.prompt_choice("Pick one", ["a", "b", "c"], default=0)

    assert idx == 1


def test_prompt_choice_falls_back_to_numbered_input(monkeypatch):
    monkeypatch.setattr(setup_mod, "_curses_prompt_choice", lambda question, choices, default=0: -1)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "2")

    idx = setup_mod.prompt_choice("Pick one", ["a", "b", "c"], default=0)

    assert idx == 1


def test_prompt_checklist_uses_shared_curses_checklist(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.curses_ui.curses_checklist",
        lambda title, items, selected, cancel_returns=None: {0, 2},
    )

    selected = setup_mod.prompt_checklist("Pick tools", ["one", "two", "three"], pre_selected=[1])

    assert selected == [0, 2]
