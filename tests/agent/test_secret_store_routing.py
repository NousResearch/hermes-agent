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


def _hold_scope_on_a_thread(home, names):
    """Install *home*'s scope on a worker thread (its own context), park it there, and return a
    function that wakes it and reports what ``get_secret`` resolves for *names* in that scope."""
    import threading
    from agent.secret_scope import build_profile_secret_scope, get_secret, reset_secret_scope, set_secret_scope
    installed, wake, done = threading.Event(), threading.Event(), threading.Event()
    seen = {}

    def worker():
        token = set_secret_scope(build_profile_secret_scope(home), profile_home=str(home))
        try:
            installed.set()
            wake.wait(10)
            seen.update({n: get_secret(n) for n in names})
        finally:
            reset_secret_scope(token)
            done.set()

    threading.Thread(target=worker, daemon=True).start()
    assert installed.wait(10)

    def read():
        wake.set()
        assert done.wait(10)
        return seen
    return read


def test_a_view_change_reaches_a_scope_already_installed_on_another_thread(store, monkeypatch):
    provider, home = store
    for name in ("ROTATING_KEY", "REVOKED_KEY", "ADDED_KEY"):
        monkeypatch.delenv(name, raising=False)
    provider.seed_external(home, {"ROTATING_KEY": "before", "REVOKED_KEY": "revoked"})
    provider.refresh(home)
    read = _hold_scope_on_a_thread(home, ("ROTATING_KEY", "REVOKED_KEY", "ADDED_KEY"))
    provider.remote[provider._key(home)]["env"] = {"ROTATING_KEY": MARK, "ADDED_KEY": "added"}
    provider.refresh(home)
    assert read() == {"ROTATING_KEY": MARK, "REVOKED_KEY": None, "ADDED_KEY": "added"}


def test_a_view_change_keeps_a_value_a_higher_layer_put_in_the_scope(store, monkeypatch):
    provider, home = store
    from hermes_cli import managed_scope
    monkeypatch.setattr(managed_scope, "load_managed_env", lambda: {"POLICY_KEY": "managed-wins"})
    provider.seed_external(home, {"POLICY_KEY": "store-old"})
    provider.refresh(home)
    from agent.secret_scope import build_profile_secret_scope, current_secret_scope, reset_secret_scope, set_secret_scope
    scope = build_profile_secret_scope(home)
    token = set_secret_scope(scope, profile_home=str(home))
    try:
        provider.remote[provider._key(home)]["env"] = {"POLICY_KEY": "store-new"}
        provider.refresh(home)
        assert current_secret_scope()["POLICY_KEY"] == "managed-wins"
        provider.remote[provider._key(home)]["env"] = {}
        provider.refresh(home)
        assert current_secret_scope()["POLICY_KEY"] == "managed-wins"
    finally:
        reset_secret_scope(token)


def test_a_finished_scope_is_collectable_without_store_registration(store):
    import gc
    import weakref
    _provider, home = store
    from agent import secret_scope
    scope = secret_scope.build_profile_secret_scope(home)
    ref = weakref.ref(scope)
    token = secret_scope.set_secret_scope(scope, profile_home=str(home))
    secret_scope.reset_secret_scope(token)
    del token, scope
    gc.collect()
    assert ref() is None


def test_under_multiplexing_a_view_change_never_writes_os_environ(store, monkeypatch):
    provider, home = store
    for name in ("ROTATING_KEY", "REVOKED_KEY"):
        monkeypatch.delenv(name, raising=False)
    provider.seed_external(home, {"ROTATING_KEY": "before", "REVOKED_KEY": "revoked"})
    from hermes_cli.env_loader import load_hermes_dotenv
    load_hermes_dotenv(hermes_home=home)
    assert os.environ["ROTATING_KEY"] == "before"
    from agent import secret_scope
    from tui_gateway import launch_profile_policy
    monkeypatch.setattr(launch_profile_policy, "_snapshot", None)
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    launch_profile_policy.capture_launch_env()
    environ_before = dict(os.environ)
    read = _hold_scope_on_a_thread(home, ("ROTATING_KEY", "REVOKED_KEY"))
    provider.remote[provider._key(home)]["env"] = {"ROTATING_KEY": MARK}
    provider.refresh(home)
    assert dict(os.environ) == environ_before
    assert read() == {"ROTATING_KEY": MARK, "REVOKED_KEY": None}
    launch_scope = launch_profile_policy.launch_secret_scope(home)
    assert launch_scope["ROTATING_KEY"] == MARK and "REVOKED_KEY" not in launch_scope


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


def test_a_view_change_reaches_a_bound_profile_scope(store, monkeypatch):
    provider, home = store
    monkeypatch.delenv("ROTATING_KEY", raising=False)
    provider.seed_external(home, {"ROTATING_KEY": "before"})
    provider.refresh(home)
    from agent.runtime_scope import bind_profile_scope
    from agent.secret_scope import get_secret
    with bind_profile_scope(home):
        assert get_secret("ROTATING_KEY") == "before"
        provider.remote[provider._key(home)]["env"] = {"ROTATING_KEY": MARK}
        provider.refresh(home)
        assert get_secret("ROTATING_KEY") == MARK


def test_the_global_auth_fallback_follows_the_stores_view_not_the_local_file(store, tmp_path, monkeypatch):
    provider, _home = store
    from hermes_cli import auth
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr(auth, "_global_auth_file_path", lambda: root / "auth.json")
    monkeypatch.setattr(auth, "_global_auth_store_cache", None)
    provider._remote(root)["auth"]["openai-codex"] = {"tokens": {"access_token": "first"}}
    provider.refresh(root)
    # No local bootstrap file at the root: the store's records are still the root's.
    assert not (root / "auth.json").exists()
    assert auth._load_global_auth_store()["providers"]["openai-codex"]["tokens"]["access_token"] == "first"
    provider._remote(root)["auth"]["openai-codex"] = {"tokens": {"access_token": MARK}}
    provider._remote(root)["version"] += 1
    provider.refresh(root)
    assert auth._load_global_auth_store()["providers"]["openai-codex"]["tokens"]["access_token"] == MARK
    del provider._remote(root)["auth"]["openai-codex"]
    provider._remote(root)["version"] += 1
    provider.refresh(root)
    assert "openai-codex" not in auth._load_global_auth_store().get("providers", {})


def test_the_global_auth_fallback_with_the_local_store_still_keys_on_the_file(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_SECRET_STORE", raising=False)
    from hermes_cli import auth
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr(auth, "_global_auth_file_path", lambda: root / "auth.json")
    monkeypatch.setattr(auth, "_global_auth_store_cache", None)
    assert auth._load_global_auth_store() == {}
    (root / "auth.json").write_text(json.dumps({"providers": {"x": {"k": 1}}}), encoding="utf-8")
    assert auth._load_global_auth_store()["providers"] == {"x": {"k": 1}}
    assert auth._global_auth_store_cache[1] == __import__("utils").file_cache_version(root / "auth.json")


def test_the_project_env_sees_the_stores_view_where_it_saw_the_profile_env(store, tmp_path, monkeypatch):
    provider, home = store
    (home / ".env").write_text("VIEW_STORE_TOKEN=boot\n", encoding="utf-8")
    provider.seed_external(home, {"ROUTED_API_KEY": MARK, "SHARED_NAME": "from-the-store"})
    project = tmp_path / "project.env"
    project.write_text("DERIVED=${ROUTED_API_KEY:-fallback}\nSHARED_NAME=from-the-project\n", encoding="utf-8")
    for name in ("ROUTED_API_KEY", "SHARED_NAME", "DERIVED", "VIEW_STORE_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    from hermes_cli.env_loader import load_hermes_dotenv
    load_hermes_dotenv(hermes_home=home, project_env=project)
    assert os.environ["DERIVED"] == MARK
    assert os.environ["SHARED_NAME"] == "from-the-store"  # the profile's value fills first, as its .env did


def test_with_no_local_env_file_the_stores_view_still_outranks_the_project_env(store, tmp_path, monkeypatch):
    provider, home = store
    provider.seed_external(home, {"SHARED_NAME": "from-the-store"})
    project = tmp_path / "project.env"
    project.write_text("SHARED_NAME=from-the-project\nPROJECT_ONLY=filled\n", encoding="utf-8")
    for name in ("SHARED_NAME", "PROJECT_ONLY"):
        monkeypatch.delenv(name, raising=False)
    from hermes_cli.env_loader import load_hermes_dotenv
    load_hermes_dotenv(hermes_home=home, project_env=project)
    assert os.environ["SHARED_NAME"] == "from-the-store" and os.environ["PROJECT_ONLY"] == "filled"


def test_the_read_only_dotenv_resolution_reads_the_stores_view(store, monkeypatch):
    provider, home = store
    (home / ".env").write_text("VIEW_STORE_TOKEN=boot\nSTALE_ONLY_ON_DISK=stale\nTERMINAL_ENV=docker\n", encoding="utf-8")
    provider.seed_external(home, {"ROUTED_API_KEY": MARK})
    monkeypatch.delenv("STALE_ONLY_ON_DISK", raising=False)
    from hermes_cli.env_loader import resolve_home_dotenv
    resolved = resolve_home_dotenv(home)
    assert resolved["ROUTED_API_KEY"] == MARK and resolved["VIEW_STORE_TOKEN"] == "boot"
    assert resolved["TERMINAL_ENV"] == "docker" and "STALE_ONLY_ON_DISK" not in resolved
