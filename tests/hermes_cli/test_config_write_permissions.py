"""Every config.yaml writer drops world access, not only save_config (#59726)."""

import os
import stat

import pytest

import hermes_constants
from hermes_cli.config import atomic_config_replace, atomic_config_write
from hermes_cli.config_backend import write_config_key
from hermes_cli.xai_retirement import apply_migration, find_retired_xai_refs

pytestmark = pytest.mark.platforms("posix")  # POSIX permission bits


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for name in ("HERMES_MANAGED", "HERMES_CONTAINER", "HERMES_SKIP_CHMOD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(hermes_constants, "_detect_container", lambda: False)
    path = home / "config.yaml"
    path.write_text("model:\n  default: old\n", encoding="utf-8")
    return path


def _mode(path):
    return stat.S_IMODE(path.stat().st_mode)


WRITERS = {
    "atomic_config_write": lambda p: atomic_config_write(p, {"model": {"default": "new"}}),
    "atomic_config_replace": lambda p: atomic_config_replace(p, {"model": {"default": "new"}}),
    "write_config_key": lambda p: write_config_key(p, "model.default", "new"),
}


@pytest.mark.parametrize("writer", sorted(WRITERS))
def test_world_writable_config_loses_other_bits(config_path, writer):
    os.chmod(config_path, 0o666)

    WRITERS[writer](config_path)

    assert _mode(config_path) == 0o660
    assert "new" in config_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(("before", "after"), [(0o675, 0o670), (0o2676, 0o2670)], ids=["0675", "setgid-2676"])
def test_group_bits_and_setgid_are_kept(config_path, before, after):
    # Shared-group homes are a documented setup: only "other" access is removed.
    os.chown(config_path, -1, os.getegid())  # setgid on a file sticks only for a group we are in
    os.chmod(config_path, before)

    write_config_key(config_path, "model.default", "new")

    assert _mode(config_path) == after


def test_managed_install_mode_is_left_alone(config_path, monkeypatch):
    monkeypatch.setenv("HERMES_MANAGED", "nixos")
    os.chmod(config_path, 0o644)

    atomic_config_write(config_path, {"model": {"default": "new"}})

    assert _mode(config_path) == 0o644


def test_xai_migration_drops_world_access(config_path):
    config_path.write_text("principal:\n  provider: xai\n  model: grok-code-fast-1\n", encoding="utf-8")
    os.chmod(config_path, 0o666)
    from hermes_cli.config import read_user_config_raw
    issues = find_retired_xai_refs(read_user_config_raw(config_path))
    assert issues

    result = apply_migration(config_path, issues, backup=False)

    assert result.config_changed
    assert _mode(config_path) == 0o660
