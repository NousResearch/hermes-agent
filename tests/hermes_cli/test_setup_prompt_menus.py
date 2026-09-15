from types import SimpleNamespace

import pytest

from hermes_cli import setup as setup_mod


def test_prompt_choice_escape_keeps_default_without_numbered_fallback(monkeypatch):
    monkeypatch.setattr(
        setup_mod,
        "_curses_prompt_choice",
        lambda question, choices, default=0, description=None: -1,
    )
    monkeypatch.setattr(
        "builtins.input",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Escape must not enter the numbered fallback")
        ),
    )

    assert setup_mod.prompt_choice("Pick one", ["a", "b"], default=1) == 1


def test_setup_navigation_escape_cancels_and_left_goes_back():
    state = setup_mod._SetupNavigationState(section_index=1)
    token = setup_mod._SETUP_NAVIGATION.set(state)
    try:
        assert setup_mod._handle_setup_menu_navigation(
            setup_mod.MenuNavigationEvent.BEGIN
        ).allow_back is True
        with pytest.raises(setup_mod._SetupCancelled):
            setup_mod._handle_setup_menu_navigation(
                setup_mod.MenuNavigationEvent.CANCEL
            )

        assert setup_mod._handle_setup_menu_navigation(
            setup_mod.MenuNavigationEvent.BEGIN
        ).allow_back is True
        with pytest.raises(setup_mod._SetupGoBack) as exc_info:
            setup_mod._handle_setup_menu_navigation(setup_mod.MenuNavigationEvent.BACK)
        assert exc_info.value.prompt_index == 1
    finally:
        setup_mod._SETUP_NAVIGATION.reset(token)


def test_setup_yes_no_uses_navigable_menu(monkeypatch):
    calls = []
    state = setup_mod._SetupNavigationState(section_index=1)
    token = setup_mod._SETUP_NAVIGATION.set(state)
    monkeypatch.setattr(
        setup_mod,
        "_curses_prompt_choice",
        lambda question, choices, default=0, description=None: calls.append(
            (question, choices, default)
        )
        or 1,
    )
    try:
        assert setup_mod.prompt_yes_no("Enable it?", default=True) is False
    finally:
        setup_mod._SETUP_NAVIGATION.reset(token)

    assert calls == [("Enable it?", ["Yes", "No"], 0)]


def test_setup_steps_move_to_previous_section_or_restart_current_section():
    calls = []
    terminal_attempts = 0
    gateway_attempts = 0

    def model():
        calls.append("model")

    def terminal():
        nonlocal terminal_attempts
        calls.append("terminal")
        terminal_attempts += 1
        if terminal_attempts == 1:
            raise setup_mod._SetupGoBack(prompt_index=0)

    def gateway():
        nonlocal gateway_attempts
        calls.append("gateway")
        gateway_attempts += 1
        if gateway_attempts == 1:
            raise setup_mod._SetupGoBack(prompt_index=1)

    state = setup_mod._SetupNavigationState()
    token = setup_mod._SETUP_NAVIGATION.set(state)
    try:
        setup_mod._run_setup_steps(
            [("Model", model), ("Terminal", terminal), ("Gateway", gateway)]
        )
    finally:
        setup_mod._SETUP_NAVIGATION.reset(token)

    assert calls == [
        "model",
        "terminal",
        "model",
        "terminal",
        "gateway",
        "gateway",
    ]


def test_nested_back_reopens_only_the_immediately_previous_prompt():
    shown = []
    attempts = 0

    def model_provider_flow():
        nonlocal attempts
        attempts += 1
        for label in ("provider", "auth method", "existing or reauthenticate"):
            start = setup_mod._handle_setup_menu_navigation(
                setup_mod.MenuNavigationEvent.BEGIN
            )
            if start.should_replay:
                setup_mod._handle_setup_menu_navigation(
                    setup_mod.MenuNavigationEvent.RESOLVE, start.replay_value
                )
                continue
            shown.append(label)
            if label == "existing or reauthenticate" and attempts == 1:
                setup_mod._handle_setup_menu_navigation(
                    setup_mod.MenuNavigationEvent.BACK
                )
            setup_mod._handle_setup_menu_navigation(
                setup_mod.MenuNavigationEvent.RESOLVE, label
            )

    state = setup_mod._SetupNavigationState()
    token = setup_mod._SETUP_NAVIGATION.set(state)
    try:
        setup_mod._run_setup_steps([("Model & Provider", model_provider_flow)])
    finally:
        setup_mod._SETUP_NAVIGATION.reset(token)

    assert shown[:4] == [
        "provider",
        "auth method",
        "existing or reauthenticate",
        "auth method",
    ]


def test_section_specific_model_setup_can_go_back_from_model_to_provider(
    tmp_path, monkeypatch
):
    """``hermes setup model`` must retain nested setup navigation."""
    shown = []
    attempts = 0

    def model_flow(_config):
        nonlocal attempts
        attempts += 1
        for label in ("provider", "model"):
            start = setup_mod._handle_setup_menu_navigation(
                setup_mod.MenuNavigationEvent.BEGIN
            )
            if start.should_replay:
                setup_mod._handle_setup_menu_navigation(
                    setup_mod.MenuNavigationEvent.RESOLVE, start.replay_value
                )
                continue
            shown.append(label)
            if label == "model" and attempts == 1:
                # This is what the menu does for Left: it only dispatches the
                # back event when its setup context enables ``← previous``.
                if start.allow_back:
                    setup_mod._handle_setup_menu_navigation(
                        setup_mod.MenuNavigationEvent.BACK
                    )
                return
            setup_mod._handle_setup_menu_navigation(
                setup_mod.MenuNavigationEvent.RESOLVE, label
            )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(setup_mod, "is_interactive_stdin", lambda: True)
    monkeypatch.setattr(
        setup_mod,
        "SETUP_SECTIONS",
        [("model", "Model & Provider", model_flow)],
    )

    setup_mod.run_setup_wizard(
        SimpleNamespace(
            section="model",
            reset=False,
            reconfigure=False,
            quick=False,
            portal=False,
            non_interactive=False,
        )
    )

    assert shown == ["provider", "model", "provider", "model"]


def test_prompt_strips_bracketed_paste_markers(monkeypatch):
    monkeypatch.setattr(
        "builtins.input",
        lambda _prompt="": "\x1b[200~sk-ant-api-key\x1b[201~",
    )

    value = setup_mod.prompt("API key")

    assert value == "sk-ant-api-key"




def test_prompt_choice_uses_curses_helper(monkeypatch):
    monkeypatch.setattr(setup_mod, "_curses_prompt_choice", lambda question, choices, default=0, description=None: 1)

    idx = setup_mod.prompt_choice("Pick one", ["a", "b", "c"], default=0)

    assert idx == 1
