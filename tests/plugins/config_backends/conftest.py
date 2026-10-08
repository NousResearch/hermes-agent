"""Fixtures and helpers shared by the remote config backend test modules."""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from hermes_cli.config_backend import get_config_backend
from plugins.config_backends import remote as remote_pkg
from plugins.config_backends.remote import backend as backend_mod
from plugins.config_backends.remote import credentials as cred_mod

from .stub_plane import StubPlane, remote_env


def _config_cmd(capsys, *argv):
    from hermes_cli.config import config_command
    ns = argparse.Namespace(config_command=argv[0], key=argv[1], value=argv[2] if len(argv) > 2 else None,
                            force=False)
    with pytest.raises(SystemExit) as exc:
        config_command(ns)
    return exc.value.code, capsys.readouterr().err


def _gets(plane):
    return [r for r in plane.requests if r["method"] == "GET"]


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
def profile_plane(tmp_path, monkeypatch):
    """``plane`` with a real profiles root (``Path.home()/.hermes``)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    with StubPlane() as p:
        for k, v in remote_env(p).items():
            monkeypatch.setenv(k, v)
        monkeypatch.setattr(backend_mod, "BOOT_RETRY_DELAYS", (0.0, 0.0))
        remote_pkg._reset_for_tests()
        cred_mod._IDP_CACHE.clear()
        p.home = home
        get_config_backend()._state(home)
        yield p
        remote_pkg._reset_for_tests()
        cred_mod._IDP_CACHE.clear()
