"""pm uv cache: hermes-owned UV_CACHE_DIR + sealed-venv bootstrap seed.

The cache ships with bundles and is copied out to the writable machine
cache on first use; every pm-internal uv invocation pins UV_CACHE_DIR.
The mutable venv on sealed installs lives in the machine hermes root,
seeded from the payload's shipped venv when lazy installs are off.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import pm.packages as pkgs


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


def test_venv_dir_sealed_goes_to_machine_root(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    import sys

    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: home)
    ensure_mod = sys.modules["pm.ensure"]
    monkeypatch.setattr(ensure_mod, "sealed", lambda: True)
    venv = pkgs.Venv()
    assert venv.venv_dir() == home / "venv"


def test_seed_mutable_venv_copies_payload_venv(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    payload = tmp_path / "payload"
    (payload / "venv" / "Scripts").mkdir(parents=True)
    (payload / "venv" / "Scripts" / "python.exe").write_text("bin", encoding="utf-8")

    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: home)
    import sys

    ensure_mod = sys.modules["pm.ensure"]
    import pm.paths as paths_mod

    monkeypatch.setattr(ensure_mod, "sealed", lambda: True)
    monkeypatch.setattr(ensure_mod, "lazy_installs_allowed", lambda: False)
    monkeypatch.setattr(paths_mod, "store_root", lambda: payload / "tools")

    venv = pkgs.Venv()
    reason = venv.seed_mutable_venv()
    assert reason is None
    seeded = home / "venv"
    assert (seeded / "Scripts" / "python.exe").read_text(encoding="utf-8") == "bin"
    # idempotent: second call is a no-op
    assert venv.seed_mutable_venv() is None


def test_seed_mutable_venv_lazy_on_skips_copy(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    payload = tmp_path / "payload"
    (payload / "venv").mkdir(parents=True)

    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: home)
    import sys

    ensure_mod = sys.modules["pm.ensure"]
    import pm.paths as paths_mod

    monkeypatch.setattr(ensure_mod, "sealed", lambda: True)
    monkeypatch.setattr(ensure_mod, "lazy_installs_allowed", lambda: True)
    monkeypatch.setattr(paths_mod, "store_root", lambda: payload / "tools")

    venv = pkgs.Venv()
    assert venv.seed_mutable_venv() is None
    assert not (home / "venv").exists()  # builds fresh later, no seed copy
