import argparse
import json

import hermes_yaml as yaml
import pytest


def test_xai_model_flow_reauth_uses_standard_radio_prompt(monkeypatch):
    from hermes_cli import model_setup_flows as main_mod

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
    from hermes_cli import model_setup_flows as main_mod

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
    from hermes_cli import model_setup_flows as main_mod

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
        "hermes_cli.model_setup_flows_common._prompt_auth_credentials_choice",
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
    from hermes_cli import model_setup_flows as main_mod
    from hermes_cli.config import atomic_config_replace as real_atomic_config_replace

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
        "hermes_cli.model_setup_flows_common._prompt_auth_credentials_choice",
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
        return real_atomic_config_replace(path, data, **kwargs)

    monkeypatch.setattr(auth_mod, "atomic_config_replace", _recording_write)

    main_mod._model_flow_xai_oauth({}, current_model="existing-direct-model")

    assert len(writes) == 1
    written_model = writes[0]["model"]
    assert written_model["provider"] == "xai-oauth"
    assert written_model["base_url"] == "https://api.x.ai/v1"
    assert written_model["default"] == "grok-4.5"
    assert yaml.safe_load(config_path.read_text())["model"] == written_model


def test_xai_model_flow_cancel_skips_reauth(monkeypatch):
    from hermes_cli import model_setup_flows as main_mod

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


def test_xai_model_flow_cancel_does_not_refresh_expired_root_grant(tmp_path, monkeypatch):
    """Current main's status is read-only: Cancel must not rotate even an expired grant."""
    import base64
    from pathlib import Path
    from hermes_cli import auth
    from hermes_cli.model_setup_flows import _model_flow_xai_oauth

    root = tmp_path / ".hermes"
    profile = root / "profiles" / "work"
    profile.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    expired = "e30." + base64.urlsafe_b64encode(b'{"exp": 1}').decode().rstrip("=") + ".sig"
    root_path = root / "auth.json"
    root_path.write_text(json.dumps({
        "version": 1, "active_provider": "openai-codex",
        "providers": {"xai-oauth": {"tokens": {"access_token": expired, "refresh_token": "root-refresh"}}},
    }), encoding="utf-8")
    config_path = profile / "config.yaml"
    config_path.write_text("model:\n  provider: openrouter\n  default: old-model\n", encoding="utf-8")
    before = (root_path.read_bytes(), config_path.read_bytes())
    monkeypatch.setattr(
        "hermes_cli.model_setup_flows_common._prompt_auth_credentials_choice", lambda *_a, **_kw: "cancel",
    )

    def no_refresh(*_args, **_kwargs):
        pytest.fail("read-only status and Cancel must not refresh")

    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", no_refresh)
    _model_flow_xai_oauth({}, current_model="old-model")
    assert (root_path.read_bytes(), config_path.read_bytes()) == before
    assert not (profile / "auth.json").exists()


def test_auth_credentials_choice_falls_back_to_numbered_prompt(monkeypatch):
    from hermes_cli import model_setup_flows_common as main_mod

    monkeypatch.setattr(
        "hermes_cli.setup._curses_prompt_choice",
        lambda title, choices, default, description=None: -1,
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": "2")

    assert main_mod._prompt_auth_credentials_choice("Credentials:") == "reauth"
