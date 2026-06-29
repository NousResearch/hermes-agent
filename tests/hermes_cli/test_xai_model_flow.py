import argparse


def test_xai_model_flow_reauth_uses_standard_radio_prompt(monkeypatch):
    from hermes_cli import main as main_mod

    captured = {"login_calls": 0}

    monkeypatch.setattr(
        "hermes_cli.auth.get_xai_oauth_auth_status",
        lambda: {"logged_in": True},
    )
    monkeypatch.setattr(
        "hermes_cli.setup._curses_prompt_choice",
        lambda title, choices, default, description=None: 1,
    )

    def _fake_login(args, provider, force_new_login=False):
        captured["login_calls"] += 1
        captured["force_new_login"] = force_new_login
        captured["args"] = args

    monkeypatch.setattr("hermes_cli.auth._login_xai_oauth", _fake_login)
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_xai_oauth_runtime_credentials",
        lambda *args, **kwargs: {"base_url": "https://api.x.ai/v1"},
    )
    monkeypatch.setattr(
        "hermes_cli.auth._prompt_model_selection",
        lambda model_ids, current_model="": None,
    )

    main_mod._model_flow_xai_oauth(
        {},
        current_model="grok-build-0.1",
        args=argparse.Namespace(manual_paste=True, no_browser=True, timeout=3),
    )

    assert captured["login_calls"] == 1
    assert captured["force_new_login"] is True
    assert captured["args"].manual_paste is True
    assert captured["args"].no_browser is True
    assert captured["args"].timeout == 3


def test_xai_model_flow_cancel_skips_reauth(monkeypatch):
    from hermes_cli import main as main_mod

    monkeypatch.setattr(
        "hermes_cli.auth.get_xai_oauth_auth_status",
        lambda: {"logged_in": True},
    )
    monkeypatch.setattr(
        "hermes_cli.setup._curses_prompt_choice",
        lambda title, choices, default, description=None: 2,
    )
    monkeypatch.setattr(
        "hermes_cli.auth._login_xai_oauth",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not reauthenticate")),
    )
    monkeypatch.setattr(
        "hermes_cli.auth._prompt_model_selection",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not pick a model")),
    )

    main_mod._model_flow_xai_oauth({}, current_model="grok-build-0.1")


def test_auth_credentials_choice_falls_back_to_numbered_prompt(monkeypatch):
    from hermes_cli import main as main_mod

    monkeypatch.setattr(
        "hermes_cli.setup._curses_prompt_choice",
        lambda title, choices, default, description=None: -1,
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": "2")

    assert main_mod._prompt_auth_credentials_choice("Credentials:") == "reauth"
