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
