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
        args=argparse.Namespace(no_browser=True, timeout=3),
    )

    assert captured["login_calls"] == 1
    assert captured["force_new_login"] is True
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
    from hermes_cli import model_setup_flows_common as main_mod

    monkeypatch.setattr(
        "hermes_cli.setup._curses_prompt_choice",
        lambda title, choices, default, description=None: -1,
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": "2")

    assert main_mod._prompt_auth_credentials_choice("Credentials:") == "reauth"


def test_xai_reauth_cancel_keeps_existing_main_route(tmp_path, monkeypatch):
    """A re-auth whose model picker is cancelled must not rewrite the main route:
    the provider/base_url write belongs to _activate_provider_model, which runs
    only after a completed model selection."""
    import yaml

    home = tmp_path / "hermes"
    home.mkdir()
    cfg = home / "config.yaml"
    cfg.write_text(
        "model:\n  default: gpt-5.5\n  provider: openai-codex\n"
        "  base_url: https://chatgpt.com/backend-api/codex\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import main as main_mod
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    token = set_hermes_home_override(str(home))
    try:
        monkeypatch.setattr("hermes_cli.auth.get_xai_oauth_auth_status", lambda: {"logged_in": True})
        # radio: reauthenticate
        monkeypatch.setattr("hermes_cli.setup._curses_prompt_choice",
                            lambda title, choices, default, description=None: 1)
        monkeypatch.setattr("hermes_cli.auth._save_xai_oauth_tokens", lambda *a, **k: None)
        monkeypatch.setattr("hermes_cli.auth.unsuppress_credential_source", lambda *a, **k: None)
        monkeypatch.setattr(
            "hermes_cli.auth._xai_oauth_device_code_login",
            lambda **k: {
                "tokens": {"access_token": "a", "refresh_token": "r"},
                "base_url": "https://api.x.ai/v1",
            },
        )
        # user cancels the model picker
        monkeypatch.setattr("hermes_cli.auth._prompt_model_selection", lambda *a, **k: None)

        main_mod._model_flow_xai_oauth({}, "gpt-5.5", args=argparse.Namespace())

        route = yaml.safe_load(cfg.read_text())["model"]
        assert route["provider"] == "openai-codex", route
        assert route["base_url"] == "https://chatgpt.com/backend-api/codex", route
        assert route["default"] == "gpt-5.5", route
    finally:
        reset_hermes_home_override(token)
