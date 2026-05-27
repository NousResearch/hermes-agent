from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import uninstall


class Completed:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_uninstall_profile_checks_gateway_subprocess_before_deleting(tmp_path, monkeypatch):
    profile_home = tmp_path / "profiles" / "alpha"
    profile_home.mkdir(parents=True)
    alias_path = tmp_path / "bin" / "alpha"
    alias_path.parent.mkdir()
    alias_path.write_text("#!/bin/sh\n")
    profile = SimpleNamespace(name="alpha", path=profile_home, alias_path=alias_path)

    monkeypatch.setattr(
        uninstall.subprocess,
        "run",
        lambda *args, **kwargs: Completed(returncode=1, stderr="module import failed"),
    )

    assert uninstall._uninstall_profile(profile) is False
    assert profile_home.exists()
    assert alias_path.exists()


def test_run_uninstall_removes_named_profiles_before_project_root(tmp_path, monkeypatch):
    project_root = tmp_path / "hermes-agent"
    project_root.mkdir()
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    profile_home = hermes_home / "profiles" / "alpha"
    profile_home.mkdir(parents=True)
    alias_path = tmp_path / "bin" / "alpha"
    alias_path.parent.mkdir()
    alias_path.write_text("#!/bin/sh\n")
    profile = SimpleNamespace(name="alpha", path=profile_home, alias_path=alias_path)

    events = []
    inputs = iter(["2", "y", "yes"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setattr(uninstall, "get_project_root", lambda: project_root)
    monkeypatch.setattr(uninstall, "get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr(uninstall, "_is_default_hermes_home", lambda home: True)
    monkeypatch.setattr(uninstall, "_discover_named_profiles", lambda: [profile])
    monkeypatch.setattr(uninstall, "uninstall_gateway_service", lambda: False)
    monkeypatch.setattr(uninstall, "remove_path_from_shell_configs", lambda: [])
    monkeypatch.setattr(uninstall, "remove_wrapper_script", lambda: [])
    monkeypatch.setattr(uninstall, "_is_windows", lambda: False)
    monkeypatch.setattr(
        uninstall.subprocess,
        "run",
        lambda *args, **kwargs: events.append(("gateway", project_root.exists())) or Completed(),
    )

    real_rmtree = uninstall.shutil.rmtree

    def record_rmtree(path, *args, **kwargs):
        events.append(("rmtree", Path(path)))
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(uninstall.shutil, "rmtree", record_rmtree)

    uninstall.run_uninstall(SimpleNamespace())

    assert ("gateway", True) in events
    first_project_delete = events.index(("rmtree", project_root))
    last_gateway = max(i for i, event in enumerate(events) if event[0] == "gateway")
    assert last_gateway < first_project_delete


def test_run_uninstall_aborts_when_named_profile_cleanup_fails(tmp_path, monkeypatch):
    project_root = tmp_path / "hermes-agent"
    project_root.mkdir()
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    profile_home = hermes_home / "profiles" / "alpha"
    profile_home.mkdir(parents=True)
    profile = SimpleNamespace(name="alpha", path=profile_home, alias_path=None)

    inputs = iter(["2", "y", "yes"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setattr(uninstall, "get_project_root", lambda: project_root)
    monkeypatch.setattr(uninstall, "get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr(uninstall, "_is_default_hermes_home", lambda home: True)
    monkeypatch.setattr(uninstall, "_discover_named_profiles", lambda: [profile])
    monkeypatch.setattr(uninstall, "uninstall_gateway_service", lambda: False)
    monkeypatch.setattr(uninstall, "remove_path_from_shell_configs", lambda: [])
    monkeypatch.setattr(uninstall, "remove_wrapper_script", lambda: [])
    monkeypatch.setattr(uninstall, "_is_windows", lambda: False)
    monkeypatch.setattr(
        uninstall.subprocess,
        "run",
        lambda *args, **kwargs: Completed(returncode=1, stderr="module import failed"),
    )
    monkeypatch.setattr(
        uninstall.shutil,
        "rmtree",
        lambda path, *args, **kwargs: pytest.fail(f"unexpected deletion: {path}"),
    )

    uninstall.run_uninstall(SimpleNamespace())

    assert project_root.exists()
    assert hermes_home.exists()
    assert profile_home.exists()
