"""PM caches are writable; shipped dependency environments stay immutable."""

from __future__ import annotations

from pathlib import Path

import pytest

import pm.packages as pkgs

@pytest.fixture(autouse=True)
def isolated_machine_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))



def test_uv_env_pins_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(pkgs, "uv_cache_dir", lambda: tmp_path / "c")
    env = pkgs.uv_env({"UV_CACHE_DIR": "/ambient/user/cache", "PATH": "x"})
    assert env["UV_CACHE_DIR"] == str(tmp_path / "c")
    # ambient UV_ vars are stripped, not inherited
    assert "UV_PROJECT_ENVIRONMENT" not in env


def test_uv_cache_dir_seeds_from_payload(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    payload = tmp_path / "payload"
    (payload / "uv-cache" / "wheels-v5").mkdir(parents=True)
    (payload / "uv-cache" / "wheels-v5" / "some.pkg").write_text("x", encoding="utf-8")

    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: home)
    import pm.paths as paths_mod

    monkeypatch.setattr(paths_mod, "store_root", lambda: payload / "tools")

    machine = pkgs.uv_cache_dir()
    assert machine == home / "cache" / "uv"
    # seed copied out
    assert (machine / "wheels-v5" / "some.pkg").read_text(encoding="utf-8") == "x"
    # seeded marker written → second call doesn't re-copy
    assert (machine / ".seeded").is_file()
    (payload / "uv-cache" / "wheels-v5" / "some.pkg").write_text("changed", encoding="utf-8")
    pkgs.uv_cache_dir()
    assert (machine / "wheels-v5" / "some.pkg").read_text(encoding="utf-8") == "x"


def test_uv_cache_dir_cold_machine_no_payload(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: home)
    import pm.paths as paths_mod

    monkeypatch.setattr(paths_mod, "store_root", lambda: tmp_path / "nowhere" / "tools")

    machine = pkgs.uv_cache_dir()
    assert machine == home / "cache" / "uv"
    assert (machine / ".seeded").is_file()


@pytest.mark.parametrize("lazy_allowed", [False, True])
def test_bundle_uses_shipped_environment_until_an_extension_is_committed(monkeypatch, tmp_path, lazy_allowed):
    import json
    import importlib
    import pm.paths as paths

    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    payload = tmp_path / "payload"
    core = payload / "hermes-agent"
    core.mkdir(parents=True)
    shipped = payload / "venv"
    shipped.mkdir()
    (payload / "manifest.json").write_text(json.dumps({"repo": "hermes-agent", "venv": "venv"}))
    monkeypatch.setattr(paths, "repo_root", lambda: core)
    monkeypatch.setattr(importlib.import_module("pm.ensure"), "lazy_installs_allowed", lambda: lazy_allowed)
    assert pkgs.Venv().venv_dir() == shipped
    assert not home.exists(), "selecting shipped dependencies must not copy or mutate them"
