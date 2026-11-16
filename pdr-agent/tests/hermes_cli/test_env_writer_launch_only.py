"""Env writers must not touch launch-only keys (#135200).

``HERMES_MANAGED_DIR`` picks the administrator's managed scope; only the real launch environment may
set it. ``save_env_value`` (dashboard Custom Keys, ``hermes config set``) persisted it and mirrored it
into ``os.environ`` live, and ``remove_env_value`` popped the launch value out of the process.
"""

import os

import pytest

from hermes_cli import config, managed_scope


@pytest.fixture
def managed_dirs(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    admin = tmp_path / "admin"
    admin.mkdir()
    (admin / ".env").write_text("SLACK_ALLOWED_USERS=admin-only\n", encoding="utf-8")
    untrusted = tmp_path / "untrusted"
    untrusted.mkdir()
    (untrusted / ".env").write_text("SLACK_ALLOWED_USERS=*\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(admin))
    managed_scope.invalidate_managed_cache()
    yield home, admin, untrusted
    managed_scope.invalidate_managed_cache()
    config.invalidate_env_cache()


def test_save_env_value_refuses_managed_dir(managed_dirs):
    home, admin, untrusted = managed_dirs
    with pytest.raises(ValueError):
        config.save_env_value("HERMES_MANAGED_DIR", str(untrusted))
    managed_scope.invalidate_managed_cache()
    assert os.environ["HERMES_MANAGED_DIR"] == str(admin)
    assert managed_scope.load_managed_env() == {"SLACK_ALLOWED_USERS": "admin-only"}
    env_file = home / ".env"
    assert not env_file.exists() or "HERMES_MANAGED_DIR" not in env_file.read_text(encoding="utf-8")


def test_remove_env_value_keeps_launch_managed_dir(managed_dirs):
    home, admin, _untrusted = managed_dirs
    (home / ".env").write_text("HERMES_MANAGED_DIR=/stale/user/line\n", encoding="utf-8")
    config.remove_env_value("HERMES_MANAGED_DIR")
    assert os.environ["HERMES_MANAGED_DIR"] == str(admin)


def test_remove_env_value_keeps_lowercase_launch_only_value(managed_dirs, monkeypatch):
    """On Windows ``hermes_managed_dir`` IS ``HERMES_MANAGED_DIR``; the publish step must not
    pop it in any letter case."""
    monkeypatch.setenv("hermes_managed_dir", "/launch/value")
    config.remove_env_value("hermes_managed_dir")
    assert os.environ["hermes_managed_dir"] == "/launch/value"
