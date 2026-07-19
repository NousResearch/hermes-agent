import argparse
import json

import yaml


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

    def _fake_login(args, provider, force_new_login=False, update_config=True):
        captured["login_calls"] += 1
        captured["force_new_login"] = force_new_login
        captured["update_config"] = update_config
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
    assert captured["update_config"] is False
    assert captured["args"].no_browser is True
    assert captured["args"].timeout == 3


def test_xai_model_flow_initial_login_is_auth_only_until_model_selection(monkeypatch):
    from hermes_cli import main as main_mod

    captured = {}
    monkeypatch.setattr(
        "hermes_cli.auth.get_xai_oauth_auth_status",
        lambda: {"logged_in": False},
    )

    def _fake_login(args, provider, force_new_login=False, update_config=True):
        captured["force_new_login"] = force_new_login
        captured["update_config"] = update_config

    monkeypatch.setattr("hermes_cli.auth._login_xai_oauth", _fake_login)
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_xai_oauth_runtime_credentials",
        lambda *args, **kwargs: {"base_url": "https://api.x.ai/v1"},
    )
    monkeypatch.setattr(
        "hermes_cli.auth._prompt_model_selection",
        lambda model_ids, current_model="": None,
    )
    monkeypatch.setattr(
        "hermes_cli.auth._update_config_for_provider",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("skipping model selection must preserve global provider")
        ),
    )

    main_mod._model_flow_xai_oauth(
        {},
        current_model="gpt-5.6-sol",
        args=argparse.Namespace(no_browser=True, timeout=3),
    )

    assert captured["force_new_login"] is False
    assert captured["update_config"] is False


def test_xai_model_flow_selection_switches_config_and_active_provider(
    tmp_path, monkeypatch
):
    from hermes_cli import main as main_mod

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True)
    auth_path = hermes_home / "auth.json"
    auth_path.write_text(
        json.dumps({"version": 1, "active_provider": "openrouter", "providers": {}})
    )
    config_path = hermes_home / "config.yaml"
    config_path.write_text(
        "model:\n  provider: openrouter\n  default: openai/gpt-5.6-sol\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(
        "hermes_cli.auth.get_xai_oauth_auth_status",
        lambda: {"logged_in": True},
    )
    monkeypatch.setattr(
        "hermes_cli.model_setup_flows._prompt_auth_credentials_choice",
        lambda *args, **kwargs: "use",
    )
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_xai_oauth_runtime_credentials",
        lambda *args, **kwargs: {"base_url": "https://api.x.ai/v1"},
    )
    monkeypatch.setattr(
        "hermes_cli.auth._prompt_model_selection",
        lambda *args, **kwargs: "grok-4.5",
    )

    main_mod._model_flow_xai_oauth({}, current_model="openai/gpt-5.6-sol")

    config = yaml.safe_load(config_path.read_text())
    auth_store = json.loads(auth_path.read_text())
    assert config["model"]["provider"] == "xai-oauth"
    assert config["model"]["default"] == "grok-4.5"
    assert auth_store["active_provider"] == "xai-oauth"


def test_xai_model_flow_selection_uses_one_coherent_config_write(
    tmp_path, monkeypatch
):
    from hermes_cli import auth as auth_mod
    from hermes_cli import main as main_mod
    from utils import atomic_yaml_write as real_atomic_yaml_write

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True)
    (hermes_home / "auth.json").write_text(
        json.dumps({"version": 1, "active_provider": "openrouter", "providers": {}})
    )
    config_path = hermes_home / "config.yaml"
    config_path.write_text(
        "model:\n"
        "  provider: openrouter\n"
        "  base_url: https://openrouter.ai/api/v1\n"
        "  default: existing-direct-model\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(
        auth_mod, "get_xai_oauth_auth_status", lambda: {"logged_in": True}
    )
    monkeypatch.setattr(
        "hermes_cli.model_setup_flows._prompt_auth_credentials_choice",
        lambda *args, **kwargs: "use",
    )
    monkeypatch.setattr(
        auth_mod,
        "resolve_xai_oauth_runtime_credentials",
        lambda *args, **kwargs: {"base_url": "https://api.x.ai/v1/"},
    )
    monkeypatch.setattr(
        auth_mod,
        "_prompt_model_selection",
        lambda *args, **kwargs: "grok-4.5",
    )
    monkeypatch.setattr(
        auth_mod,
        "_save_model_choice",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("xAI selection must not perform a separate model write")
        ),
    )

    writes = []

    def _recording_write(path, data, **kwargs):
        writes.append(yaml.safe_load(yaml.safe_dump(data)))
        return real_atomic_yaml_write(path, data, **kwargs)

    monkeypatch.setattr(auth_mod, "atomic_yaml_write", _recording_write)

    main_mod._model_flow_xai_oauth({}, current_model="existing-direct-model")

    assert len(writes) == 1
    written_model = writes[0]["model"]
    assert written_model["provider"] == "xai-oauth"
    assert written_model["base_url"] == "https://api.x.ai/v1"
    assert written_model["default"] == "grok-4.5"
    assert yaml.safe_load(config_path.read_text())["model"] == written_model


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


def test_xai_model_flow_terminal_root_pool_refresh_then_cancel_preserves_route(
    tmp_path, monkeypatch
):
    from agent.credential_pool import CredentialPool
    from hermes_cli import auth as auth_mod
    from hermes_cli import main as main_mod

    hermes_home = tmp_path / "hermes"
    profile_path = tmp_path / "profiles" / "work" / "auth.json"
    root_path = tmp_path / "root" / "auth.json"
    profile_path.parent.mkdir(parents=True)
    root_path.parent.mkdir(parents=True)
    hermes_home.mkdir(parents=True)
    profile_path.write_text(
        json.dumps(
            {"version": 1, "active_provider": "openrouter", "providers": {}}
        )
    )
    root_path.write_text(
        json.dumps(
            {
                "version": 1,
                "active_provider": "openai-codex",
                "providers": {
                    "xai-oauth": {
                        "tokens": {
                            "access_token": "dead-access",
                            "refresh_token": "dead-refresh",
                        },
                        "discovery": {
                            "client_id": "xai-client",
                            "token_endpoint": "https://auth.x.ai/oauth/token",
                            "redirect_uri": "http://localhost:1455/auth/callback",
                        },
                    }
                },
            }
        )
    )
    config_path = hermes_home / "config.yaml"
    config_path.write_text(
        "model:\n  provider: openrouter\n  default: openai/gpt-5.6-sol\n"
    )
    config_before = config_path.read_bytes()

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HOME", str(tmp_path / "not-the-root"))
    monkeypatch.setattr(auth_mod, "_auth_file_path", lambda: profile_path)
    monkeypatch.setattr(auth_mod, "_global_auth_file_path", lambda: root_path)
    monkeypatch.setattr(CredentialPool, "_entry_needs_refresh", lambda *_args: True)

    def _terminal_refresh(*_args, **_kwargs):
        raise auth_mod.AuthError(
            "Refresh session has been revoked",
            provider="xai-oauth",
            code="xai_refresh_failed",
            relogin_required=True,
        )

    monkeypatch.setattr(auth_mod, "refresh_xai_oauth_pure", _terminal_refresh)
    login_calls = []

    def _cancel_login(*_args, **_kwargs):
        login_calls.append(True)
        raise SystemExit(1)

    monkeypatch.setattr(auth_mod, "_login_xai_oauth", _cancel_login)

    main_mod._model_flow_xai_oauth(
        {},
        current_model="openai/gpt-5.6-sol",
        args=argparse.Namespace(no_browser=True, timeout=3),
    )

    assert login_calls == [True]
    assert config_path.read_bytes() == config_before
    profile = json.loads(profile_path.read_text())
    root = json.loads(root_path.read_text())
    assert profile["active_provider"] == "openrouter"
    assert "xai-oauth" not in profile["providers"]
    assert root["active_provider"] == "openai-codex"
    root_state = root["providers"]["xai-oauth"]
    assert not root_state["tokens"].get("access_token")
    assert not root_state["tokens"].get("refresh_token")
    assert root_state["last_auth_error"]["reason"] == "credential_pool_refresh_failure"


def test_auth_credentials_choice_falls_back_to_numbered_prompt(monkeypatch):
    from hermes_cli import main as main_mod

    monkeypatch.setattr(
        "hermes_cli.setup._curses_prompt_choice",
        lambda title, choices, default, description=None: -1,
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": "2")

    assert main_mod._prompt_auth_credentials_choice("Credentials:") == "reauth"
