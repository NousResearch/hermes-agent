"""Every config.yaml writer drops world access, not only save_config (#59726)."""

import os
import stat

import pytest

import hermes_constants
from hermes_cli.config import atomic_config_replace, atomic_config_write
from hermes_cli.config_backend import write_config_key

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


def test_group_bits_are_kept(config_path):
    # Shared-group homes are a documented setup: only "other" access is removed.
    os.chmod(config_path, 0o675)

    write_config_key(config_path, "model.default", "new")

    assert _mode(config_path) == 0o670


def test_managed_install_mode_is_left_alone(config_path, monkeypatch):
    monkeypatch.setenv("HERMES_MANAGED", "nixos")
    os.chmod(config_path, 0o644)

    atomic_config_write(config_path, {"model": {"default": "new"}})

    assert _mode(config_path) == 0o644
