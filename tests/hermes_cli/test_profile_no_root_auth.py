"""Regression for #120219: fresh profiles can opt out of the shared auth store."""

import json
from pathlib import Path


def test_fresh_profile_opt_out_is_persistent_and_scoped(tmp_path, monkeypatch):
    from hermes_cli import auth
    from hermes_cli.profiles import create_profile

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    (root / "auth.json").write_text(json.dumps({
        "providers": {"xai-oauth": {"tokens": {"access_token": "root", "refresh_token": "root-refresh"}}},
        "credential_pool": {"openai-codex": [{"id": "root-codex", "auth_type": "oauth", "access_token": "root"}]},
    }))
    isolated = create_profile("isolated", no_skills=True, no_root_auth=True)
    shared = create_profile("shared", no_skills=True)
    for home, expected in ((shared, True), (isolated, False), (shared, True)):
        monkeypatch.setenv("HERMES_HOME", str(home))
        assert bool(auth.read_credential_pool("openai-codex")) is expected
        assert bool(auth.get_provider_auth_state("xai-oauth")) is expected
        assert (auth._global_auth_file_path() is not None) is expected
    assert json.loads((root / "auth.json").read_text())["providers"]["xai-oauth"]["tokens"]["access_token"] == "root"


def test_isolated_profile_keeps_its_own_auth_without_root_writethrough(tmp_path, monkeypatch):
    from hermes_cli import auth
    from hermes_cli.profiles import create_profile

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    root_auth = root / "auth.json"
    root_auth.write_text(json.dumps({"providers": {"nous": {"access_token": "root"}}}))
    isolated = create_profile("isolated", no_skills=True, no_root_auth=True)
    monkeypatch.setenv("HERMES_HOME", str(isolated))
    auth._save_active_provider_state("nous", {"access_token": "local"})
    assert auth.get_provider_auth_state("nous")["access_token"] == "local"
    assert json.loads(root_auth.read_text())["providers"]["nous"]["access_token"] == "root"


def test_isolated_nous_refresh_never_uses_shared_store(tmp_path, monkeypatch):
    from hermes_cli import auth
    from hermes_cli.auth_nous import _NousRuntimeResolve, _clear_shared_nous_state
    from hermes_cli.profiles import create_profile

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    # Even an explicit process-wide override must not defeat a profile's isolation marker.
    shared_dir = tmp_path / "shared-override"
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(shared_dir))
    root_state = {"access_token": "root-access", "refresh_token": "root-refresh"}
    auth._write_shared_nous_state(root_state)
    shared_file = shared_dir / "nous_auth.json"
    original_shared = shared_file.read_bytes()
    isolated = create_profile("isolated", no_skills=True, no_root_auth=True)
    ordinary = create_profile("ordinary", no_skills=True)

    for home, enabled in ((ordinary, True), (isolated, False), (ordinary, True)):
        monkeypatch.setenv("HERMES_HOME", str(home))
        state = {"access_token": "local-access", "refresh_token": "local-refresh"}
        assert (auth._read_shared_nous_state() is not None) is enabled
        with auth._nous_shared_store_lock():
            assert auth._merge_shared_nous_oauth_state(state) is enabled
        assert state["refresh_token"] == ("root-refresh" if enabled else "local-refresh")

    monkeypatch.setenv("HERMES_HOME", str(isolated))
    assert auth._try_import_shared_nous_state() is None
    auth._save_active_provider_state("nous", {
        "access_token": "local-access", "refresh_token": "local-refresh"})
    store = auth._load_auth_store()
    local_state = dict(store["providers"]["nous"])
    run = _NousRuntimeResolve(
        store, local_state, isolated / "auth.json", force_refresh=False,
        stale_access_token=None, timeout_seconds=1.0)
    local_state.update(access_token="local-rotated", refresh_token="local-rotated-refresh")
    with run.shared_lock():
        run.persist("post_refresh_access_token")
    assert auth.get_provider_auth_state("nous")["refresh_token"] == "local-rotated-refresh"
    _clear_shared_nous_state("isolated_terminal_refresh_failure")
    assert shared_file.read_bytes() == original_shared

    monkeypatch.setenv("HERMES_HOME", str(ordinary))
    assert auth._read_shared_nous_state()["refresh_token"] == "root-refresh"
