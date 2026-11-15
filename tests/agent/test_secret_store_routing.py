"""The process-wide credential paths reach the secret store ``HERMES_SECRET_STORE`` selects.

With the local store (unset, ``local``) every path is today's file code; the existing env, config,
auth and secret-scope suites pin that. These tests select an in-memory store that keeps secrets
off disk (``tests/conformance/secrets/reference_view.py``) and check each choke point serves it:
the profile ``.env`` read, the launch load into ``os.environ``, ``save_env_value`` /
``remove_env_value``, the ``auth.json`` store and its direct readers, plus selection itself
(read from the environment and the launch ``.env``, failing closed on a store that is missing).
"""

import json
import os

import pytest

import hermes_secret_routing as routing
from tests.conformance.secrets.reference_view import MemoryViewProvider

MARK = "routing-marker-9d41"


@pytest.fixture
def store(tmp_path, monkeypatch):
    """The in-memory store selected for a fresh profile home (``HERMES_HOME``)."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    provider = MemoryViewProvider()
    routing.reset_for_tests()
    monkeypatch.setitem(routing._PROVIDERS, "memview", provider)
    monkeypatch.setenv("HERMES_SECRET_STORE", "memview")
    yield provider, home
    routing.reset_for_tests()


def test_unset_and_local_select_the_files(monkeypatch):
    monkeypatch.delenv("HERMES_SECRET_STORE", raising=False)
    assert routing.routed_provider() is None
    monkeypatch.setenv("HERMES_SECRET_STORE", "local")
    assert routing.routed_provider() is None and routing.supports_file_tooling()


def test_a_missing_store_fails_closed_on_every_secret_path(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text("OPENAI_API_KEY=from-the-file\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_SECRET_STORE", "no-such-store")
    routing.reset_for_tests()
    from hermes_cli import auth, config
    for read in (lambda: routing.read_profile_env(home / ".env"), config.load_env,
                 lambda: config.save_env_value("OPENAI_API_KEY", "x"), auth._load_auth_store):
        with pytest.raises(routing.SecretStoreUnavailable) as refused:
            read()
        assert "no-such-store" in str(refused.value.code) and "from-the-file" not in str(refused.value.code)


def test_the_profile_env_is_the_stores_view_plus_local_bootstrap_names(store):
    provider, home = store
    (home / ".env").write_text("VIEW_STORE_TOKEN=boot\nSTALE_KEY=left-on-disk\nTERMINAL_ENV=local\n"
                               "Hermes_Managed_Dir=/launch-only\n", encoding="utf-8")
    provider.seed_external(home, {"OPENAI_API_KEY": MARK, "TERMINAL_ENV": "from-the-store"})
    from agent.secret_scope import build_profile_secret_scope
    from hermes_cli.config import get_env_value_prefer_dotenv, load_env
    expected = {"VIEW_STORE_TOKEN": "boot", "TERMINAL_ENV": "local", "Hermes_Managed_Dir": "/launch-only",
                "OPENAI_API_KEY": MARK}
    assert routing.read_profile_env(home / ".env") == expected
    assert load_env() == expected
    assert build_profile_secret_scope(home)["OPENAI_API_KEY"] == MARK
    assert "STALE_KEY" not in build_profile_secret_scope(home)
    assert get_env_value_prefer_dotenv("OPENAI_API_KEY") == MARK


def test_save_and_remove_go_to_the_store_and_never_the_file(store, monkeypatch):
    provider, home = store
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from hermes_cli.config import remove_env_value, save_env_value
    save_env_value("OPENAI_API_KEY", MARK)
    assert provider.remote[provider._key(home)]["env"]["OPENAI_API_KEY"] == MARK
    assert not (home / ".env").exists() or MARK not in (home / ".env").read_text(encoding="utf-8")
    assert os.environ["OPENAI_API_KEY"] == MARK
    assert remove_env_value("OPENAI_API_KEY") is True
    assert "OPENAI_API_KEY" not in provider.remote[provider._key(home)]["env"]
    assert "OPENAI_API_KEY" not in os.environ
    assert remove_env_value("OPENAI_API_KEY") is False


def test_a_bootstrap_name_is_written_to_the_file_and_never_sent(store):
    provider, home = store
    from hermes_cli.config import save_env_value
    save_env_value("VIEW_STORE_TOKEN", "boot-token")
    save_env_value("TERMINAL_ENV", "docker")
    text = (home / ".env").read_text(encoding="utf-8")
    assert "VIEW_STORE_TOKEN=boot-token" in text and "TERMINAL_ENV=docker" in text
    assert provider.writes == []


def test_the_launch_load_publishes_the_view_and_no_stale_local_name(store, monkeypatch):
    provider, home = store
    (home / ".env").write_text("VIEW_STORE_TOKEN=boot\nSTALE_ONLY_ON_DISK=stale\n", encoding="utf-8")
    provider.seed_external(home, {"ROUTED_API_KEY": MARK})
    for name in ("VIEW_STORE_TOKEN", "STALE_ONLY_ON_DISK", "ROUTED_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    from hermes_cli.env_loader import launch_dotenv_keys, load_hermes_dotenv
    load_hermes_dotenv(hermes_home=home)
    assert provider.booted == [provider._key(home)]
    assert os.environ["ROUTED_API_KEY"] == MARK and os.environ["VIEW_STORE_TOKEN"] == "boot"
    assert "STALE_ONLY_ON_DISK" not in os.environ
    assert {"ROUTED_API_KEY", "VIEW_STORE_TOKEN"} <= launch_dotenv_keys()  # launch residue for routed children


def test_a_view_change_moves_os_environ_for_the_published_home_only(store, monkeypatch):
    provider, home = store
    provider.seed_external(home, {"ROTATING_KEY": "old", "GONE_KEY": "gone"})
    monkeypatch.delenv("ROTATING_KEY", raising=False)
    monkeypatch.delenv("GONE_KEY", raising=False)
    from hermes_cli.env_loader import load_hermes_dotenv
    load_hermes_dotenv(hermes_home=home)
    monkeypatch.setenv("SHELL_OWNED", "mine")
    routing.view_changed(home, {"ROTATING_KEY": MARK})
    assert os.environ["ROTATING_KEY"] == MARK and "GONE_KEY" not in os.environ
    routing.view_changed(home.parent / "other", {"SHELL_OWNED": "theirs"})
    assert os.environ["SHELL_OWNED"] == "mine"


def test_the_auth_store_keeps_bootstrap_records_local_and_the_rest_in_the_store(store):
    provider, home = store
    from hermes_cli import auth
    with auth._auth_store_lock():
        st = auth._load_auth_store()
        st["providers"]["nous"] = {"access_token": "boot-nous"}
        st["providers"]["openai-codex"] = {"tokens": {"access_token": MARK}}
        st["active_provider"] = "openai-codex"
        auth._save_auth_store(st)
    on_disk = json.loads((home / "auth.json").read_text(encoding="utf-8"))
    assert on_disk["providers"] == {"nous": {"access_token": "boot-nous"}} and MARK not in json.dumps(on_disk)
    held = provider.remote[provider._key(home)]["auth"]
    assert held["openai-codex"] == {"tokens": {"access_token": MARK}}
    assert held[routing.STORE_META_RECORD] == {"active_provider": "openai-codex"} and "nous" not in held
    loaded = auth._load_auth_store()
    assert loaded["providers"]["openai-codex"]["tokens"]["access_token"] == MARK
    assert loaded["providers"]["nous"] == {"access_token": "boot-nous"} and loaded["active_provider"] == "openai-codex"
    assert routing.read_auth_json(home / "auth.json")["active_provider"] == "openai-codex"


def test_an_auth_save_sends_only_what_the_caller_changed(store):
    provider, home = store
    from hermes_cli import auth
    provider._remote(home)["auth"].update({"a-provider": {"v": 1}, "b-provider": {"v": 1}})
    provider.refresh(home)
    st = auth._load_auth_store()
    provider.remote[provider._key(home)]["auth"]["b-provider"] = {"v": 2}  # another writer, meanwhile
    provider.refresh(home)
    st["providers"]["a-provider"] = {"v": 9}
    auth._save_auth_store(st)
    held = provider.remote[provider._key(home)]["auth"]
    assert held["a-provider"] == {"v": 9} and held["b-provider"] == {"v": 2}


def test_file_tooling_is_refused_and_the_deployment_is_never_residue(store):
    from hermes_cli.config_backend import deployment_env_names, require_profile_file_tooling
    with pytest.raises(routing.SecretStoreUnavailable):
        require_profile_file_tooling("Profile clone")
    assert {"HERMES_SECRET_STORE", "VIEW_STORE_URL"} <= deployment_env_names()


def test_the_selector_in_the_launch_env_is_published_before_the_first_read(store, monkeypatch):
    _provider, home = store
    (home / ".env").write_text("HERMES_SECRET_STORE=memview\nVIEW_STORE_URL=https://store.test\n", encoding="utf-8")
    monkeypatch.delenv("HERMES_SECRET_STORE", raising=False)
    monkeypatch.delenv("VIEW_STORE_URL", raising=False)
    from hermes_cli.env_loader import apply_config_bootstrap_env
    apply_config_bootstrap_env(lambda kind: frozenset(), home, None)
    assert os.environ["HERMES_SECRET_STORE"] == "memview" and os.environ["VIEW_STORE_URL"] == "https://store.test"
