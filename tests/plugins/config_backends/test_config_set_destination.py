"""`hermes config set/unset` name where the write actually landed.

Remote mode writes the profile level of Remote Config and never a local config.yaml, so the
success line must not point at ``<home>/config.yaml``; the file backend keeps printing the path.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from plugins.config_backends import remote as remote_pkg
from plugins.config_backends.remote import backend as backend_mod
from plugins.config_backends.remote import credentials as cred_mod

from .stub_plane import StubPlane, remote_env


@pytest.fixture
def plane(tmp_path, monkeypatch):
    home = Path(tmp_path) / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    with StubPlane() as p:
        for k, v in remote_env(p).items():
            monkeypatch.setenv(k, v)
        monkeypatch.setattr(backend_mod, "BOOT_RETRY_DELAYS", (0.0, 0.0))
        remote_pkg._reset_for_tests()
        cred_mod._IDP_CACHE.clear()
        p.home = home
        yield p
        remote_pkg._reset_for_tests()
        cred_mod._IDP_CACHE.clear()


@pytest.fixture
def file_home(tmp_path, monkeypatch):
    home = Path(tmp_path) / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_CONFIG_BACKEND", raising=False)
    return home


def test_remote_set_and_unset_name_remote_config_not_a_local_file(plane, capsys):
    from hermes_cli.config import set_config_value, unset_config_value

    set_config_value("agent.max_turns", "7")
    unset_config_value("agent.max_turns")

    out = capsys.readouterr().out
    assert "✓ Set agent.max_turns = 7 in Remote Config (profile 'default' level)" in out
    assert "✓ Unset agent.max_turns from Remote Config (profile 'default' level)" in out
    assert "config.yaml" not in out
    assert not (plane.home / "config.yaml").exists()


def test_file_set_and_unset_still_print_the_config_path(file_home, capsys):
    from hermes_cli.config import set_config_value, unset_config_value

    set_config_value("agent.max_turns", "7")
    unset_config_value("agent.max_turns")

    out = capsys.readouterr().out
    path = file_home / "config.yaml"
    assert f"✓ Set agent.max_turns = 7 in {path}\n" in out
    assert f"✓ Unset agent.max_turns from {path}\n" in out
