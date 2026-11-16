"""The selected store occupies the profile .env layer, never an overlay above it."""

import os

import pytest

from agent import secret_scope as ss
from agent.runtime_scope import bind_state, scope_for_profile
import hermes_secret_routing as routing
from tests.conformance.secrets.reference_view import MemoryViewProvider


@pytest.fixture(params=["local", "memview"])
def profile_store(request, tmp_path, monkeypatch):
    home = tmp_path / "profile-store-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_SECRET_STORE", request.param)
    routing.reset_for_tests()
    provider = MemoryViewProvider() if request.param == "memview" else None
    if provider:
        monkeypatch.setitem(routing._PROVIDERS, "memview", provider)

    def seed(values):
        if provider:
            provider._remote(home)["env"] = dict(values)
            provider.refresh(home)
        else:
            from hermes_cli.config import _quote_env_value
            (home / ".env").write_text("".join(
                f"{k}={_quote_env_value(v)}\n" for k, v in values.items()), encoding="utf-8")
    yield home, seed
    routing.reset_for_tests()


def test_dotenv_overrides_launch_value(profile_store, monkeypatch):
    home, seed = profile_store
    seed({"OPENAI_API_KEY": "profile-marker"})
    monkeypatch.setenv("OPENAI_API_KEY", "launch-marker")
    from hermes_cli.env_loader import load_hermes_dotenv
    load_hermes_dotenv(hermes_home=home, load_external_secrets=False)
    assert os.environ["OPENAI_API_KEY"] == "profile-marker"


def test_dotenv_expansion_and_project_fallback(profile_store, tmp_path, monkeypatch):
    home, seed = profile_store
    seed({"EXPAND_BASE": "profile", "EXPANDED": "${EXPAND_BASE}/suffix"})
    monkeypatch.setenv("EXPAND_BASE", "launch")
    monkeypatch.delenv("EXPANDED", raising=False)
    project = tmp_path / "project.env"
    project.write_text("PROJECT_REF=${EXPANDED}\nEXPAND_BASE=project\n", encoding="utf-8")
    from hermes_cli.env_loader import load_hermes_dotenv
    load_hermes_dotenv(hermes_home=home, project_env=project, load_external_secrets=False)
    assert os.environ["EXPANDED"] == "profile/suffix"
    assert os.environ["PROJECT_REF"] == "profile/suffix"
    assert os.environ["EXPAND_BASE"] == "profile"


def test_managed_external_profile_and_caller_rank(profile_store, monkeypatch):
    home, seed = profile_store
    seed({"RANK_KEY": "profile", "EXTERNAL_KEY": "profile", "PROFILE_KEY": "profile", "EMPTY": ""})
    monkeypatch.setattr("hermes_cli.env_loader.get_secret_source_values", lambda home: {
        "RANK_KEY": "external", "EXTERNAL_KEY": "external"})
    monkeypatch.setattr("hermes_cli.managed_scope.load_managed_env", lambda: {"RANK_KEY": "managed"})
    scope = ss.build_profile_secret_scope(home)
    assert scope["RANK_KEY"] == "managed"
    assert scope["EXTERNAL_KEY"] == "external"
    assert scope["PROFILE_KEY"] == "profile"
    assert scope["EMPTY"] == ""
    with bind_state(scope_for_profile(home), secrets=ss.overlay_secret_scope(scope, {"RANK_KEY": "caller"}),
                    terminal=False):
        assert ss.get_secret("RANK_KEY") == "caller"
        from hermes_cli.config import _env_ref_lookup
        assert _env_ref_lookup("PROFILE_KEY") == "profile"


def test_explicit_deny_beats_yaml_grant(profile_store, monkeypatch):
    home, seed = profile_store
    seed({"GATEWAY_ALLOW_ALL_USERS": "false"})
    monkeypatch.setattr("gateway.config_loader.bridged_allow_all_users", lambda: "true")
    monkeypatch.setattr(ss, "_is_process_home", lambda home: True)
    assert ss.build_profile_secret_scope(home)["GATEWAY_ALLOW_ALL_USERS"] == "false"


def test_send_yaml_bridge_writes_installed_overlay(profile_store, monkeypatch):
    home, seed = profile_store
    seed({})
    (home / "config.yaml").write_text("TELEGRAM_HOME_CHANNEL: yaml-marker\n", encoding="utf-8")
    monkeypatch.setattr(ss, "_MULTIPLEX_ACTIVE", True)
    from hermes_cli.send_cmd import _load_hermes_env
    with bind_state(scope_for_profile(home), terminal=False, hydrate_secrets=False):
        _load_hermes_env()
        assert ss.get_secret("TELEGRAM_HOME_CHANNEL") == "yaml-marker"


@pytest.mark.parametrize("remove", [False, True])
def test_local_bootstrap_write_updates_bound_scope(profile_store, monkeypatch, remove):
    home, seed = profile_store
    seed({})
    (home / ".env").write_text("VIEW_STORE_TOKEN=old-marker\n", encoding="utf-8")
    monkeypatch.setattr(ss, "_MULTIPLEX_ACTIVE", True)
    from hermes_cli.config import remove_env_value, save_env_value
    with bind_state(scope_for_profile(home), terminal=False, hydrate_secrets=False):
        assert ss.get_secret("VIEW_STORE_TOKEN") == "old-marker"
        if remove:
            remove_env_value("VIEW_STORE_TOKEN")
        else:
            save_env_value("VIEW_STORE_TOKEN", "new-marker")
        assert ss.get_secret("VIEW_STORE_TOKEN") == (None if remove else "new-marker")


def test_nested_lookup_does_not_materialize_lower_scope(profile_store, monkeypatch):
    home, seed = profile_store
    seed({"PROFILE_KEY": "profile"})
    scope = ss.overlay_secret_scope(ss.build_profile_secret_scope(home), {"CALLER_KEY": "caller"})
    def refuse(*args):
        raise AssertionError("one-key lookup materialized a complete scope")
    monkeypatch.setattr(ss.LiveSecretScope, "copy", refuse)
    assert scope["CALLER_KEY"] == "caller"
    assert scope["PROFILE_KEY"] == "profile"
