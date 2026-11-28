"""Tests for exposing PM's installed git on PATH in ``pm.environment.managed_environment``.

Standalone processes (e.g. ``python -m hermes_cli.default_plugins``) can invoke
``Venv.apply`` -> ``managed_environment`` before ``pm.activate()`` runs. On stock
Windows with no system Git, ``uv lock`` must still find the bundled Git from PM's
tool store to resolve git-sourced requirements in ``pyproject.toml``.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from pm import environment as env_module, install as install_module, paths
from pm.lock import Facts, Lockfile
from pm.packages import Git
from pm.store import Store, current_target

pytestmark = pytest.mark.platforms("windows", "posix")


def _stub_toolchain(tmp_path: Path, monkeypatch):
    uv = tmp_path / "uv"
    py = tmp_path / "foreign-python"
    uv.write_text("", encoding="utf-8")
    py.write_text("", encoding="utf-8")
    monkeypatch.setattr("pm._uv._toolchain", lambda **kw: (uv, py))
    return uv, py


def _install_fake_git_store(tmp_path: Path, monkeypatch) -> tuple[str, str]:
    """Create a real read-only PM store + facts + lockfile recording ``git``."""
    target = current_target()
    version = "2.56.1+1"
    sha = "a" * 64
    lock_path = tmp_path / "lock.json"
    lock = Lockfile(lock_path)
    lock.set_pin("git", version, {target: {"url": "https://example.invalid/PortableGit.7z.exe", "sha256": sha}})
    lock.save()

    store_root = tmp_path / "payload" / "tools"
    store = Store(store_root)
    entry_name = f"git-{version}-{target}"
    entry = store.entry(entry_name)
    (entry / "cmd").mkdir(parents=True)
    (entry / "usr" / "bin").mkdir(parents=True)
    (entry / "cmd" / "git.exe").write_bytes(b"")

    git_pkg = Git()
    facts = Facts(store_root / "facts.json")
    facts.record(
        "git", version, entry_name, git_pkg.env(entry, target), store.root,
        target=target, artifacts=[sha], digest="0" * 64,
    )

    monkeypatch.setattr(paths, "lockfile_path", lambda: lock_path)
    monkeypatch.setattr(paths, "store_root", lambda: store_root)
    monkeypatch.setattr(paths, "writable_store_root", lambda: tmp_path / "writable" / "tools")
    monkeypatch.setattr(paths, "facts_path", lambda: store_root / "facts.json")
    # Reading git's PATH must never trigger downloads or installs.
    monkeypatch.setattr(install_module, "_install", lambda *a, **kw: pytest.fail("must not install"))
    monkeypatch.setattr(install_module, "ensure", lambda *a, **kw: pytest.fail("must not ensure"))
    return str(store_root / entry_name / "cmd"), str(store_root / entry_name / "usr" / "bin")


def test_managed_environment_prepends_installed_git_to_ambient_path(tmp_path, monkeypatch):
    _uv, py = _stub_toolchain(tmp_path, monkeypatch)
    cmd_dir, usr_bin_dir = _install_fake_git_store(tmp_path, monkeypatch)
    ambient = os.pathsep.join(["/ambient/bin", "/ambient/usr/bin"])
    monkeypatch.setenv("PATH", ambient)

    built = env_module.managed_environment(tmp_path / "venv", python=py, cache=tmp_path / "cache")
    parts = built.env["PATH"].split(os.pathsep)
    assert parts[:4] == [cmd_dir, usr_bin_dir, "/ambient/bin", "/ambient/usr/bin"]

    # Ambient PATH missing/empty still gets PM's git when env=None.
    monkeypatch.delenv("PATH", raising=False)
    built_empty = env_module.managed_environment(tmp_path / "venv2", python=py, cache=tmp_path / "cache")
    assert built_empty.env["PATH"].split(os.pathsep) == [cmd_dir, usr_bin_dir]


def test_managed_environment_prepends_installed_git_to_explicit_path_and_dedupes(tmp_path, monkeypatch):
    _uv, py = _stub_toolchain(tmp_path, monkeypatch)
    cmd_dir, usr_bin_dir = _install_fake_git_store(tmp_path, monkeypatch)

    # Explicit env with PATH (e.g. source_build_environment on Windows ARM64)
    explicit = {"PATH": os.pathsep.join(["/msvc/bin", cmd_dir, "/rust/bin"]), "CC": "cl"}
    built = env_module.managed_environment(
        tmp_path / "venv", python=py, cache=tmp_path / "cache", env=explicit,
    )
    parts = built.env["PATH"].split(os.pathsep)
    assert parts == [cmd_dir, usr_bin_dir, "/msvc/bin", "/rust/bin"]
    assert parts.count(cmd_dir) == 1
    assert built.env["CC"] == "cl"
    # Caller's dict was not mutated.
    assert explicit["PATH"] == os.pathsep.join(["/msvc/bin", cmd_dir, "/rust/bin"])

    # Mixed-case "Path" key (PowerShell output on Windows) is updated in place, never duplicated.
    mixed = {"Path": "/msvc/bin"}
    built_mixed = env_module.managed_environment(
        tmp_path / "venv-mixed", python=py, cache=tmp_path / "cache", env=mixed,
    )
    path_keys = [key for key in built_mixed.env if key.upper() == "PATH"]
    assert path_keys == ["Path"]
    assert built_mixed.env["Path"].split(os.pathsep) == [cmd_dir, usr_bin_dir, "/msvc/bin"]


def test_managed_environment_respects_explicit_env_without_path_or_empty_path(tmp_path, monkeypatch):
    _uv, py = _stub_toolchain(tmp_path, monkeypatch)
    _install_fake_git_store(tmp_path, monkeypatch)

    empty_env = env_module.managed_environment(
        tmp_path / "empty", python=py, cache=tmp_path / "cache", env={},
    ).env
    assert not any(k.upper() == "PATH" for k in empty_env)

    no_path_env = env_module.managed_environment(
        tmp_path / "no-path", python=py, cache=tmp_path / "cache", env={"AR": "/custom/ar"},
    ).env
    assert no_path_env["AR"] == "/custom/ar"
    assert not any(k.upper() == "PATH" for k in no_path_env)

    cleared_env = env_module.managed_environment(
        tmp_path / "cleared-path", python=py, cache=tmp_path / "cache", env={"PATH": ""},
    ).env
    assert cleared_env["PATH"] == ""


def test_managed_environment_keeps_ambient_path_when_git_not_installed(tmp_path, monkeypatch):
    _uv, py = _stub_toolchain(tmp_path, monkeypatch)
    empty_store = tmp_path / "empty-store"
    empty_store.mkdir()
    monkeypatch.setattr(paths, "store_root", lambda: empty_store)
    monkeypatch.setattr(paths, "writable_store_root", lambda: empty_store)
    monkeypatch.setattr(paths, "facts_path", lambda: empty_store / "facts.json")
    monkeypatch.setattr(install_module, "_install", lambda *a, **kw: pytest.fail("must not install"))
    ambient = os.pathsep.join(["/usr/bin", "/bin"])
    monkeypatch.setenv("PATH", ambient)

    built = env_module.managed_environment(tmp_path / "venv", python=py, cache=tmp_path / "cache")
    assert built.env["PATH"] == ambient


def test_managed_environment_survives_unreadable_git_lookup(tmp_path, monkeypatch):
    _uv, py = _stub_toolchain(tmp_path, monkeypatch)
    monkeypatch.setattr(
        install_module, "installed_package",
        lambda *a, **kw: (_ for _ in ()).throw(OSError("corrupt facts")),
    )
    ambient = "/usr/bin"
    monkeypatch.setenv("PATH", ambient)

    built = env_module.managed_environment(tmp_path / "venv", python=py, cache=tmp_path / "cache")
    assert built.env["PATH"] == ambient


def test_managed_environment_strips_verbatim_prefix_from_git_path(tmp_path, monkeypatch):
    _uv, py = _stub_toolchain(tmp_path, monkeypatch)
    # UNC verbatim path has no ':' so it splits cleanly with os.pathsep on every OS.
    verbatim = Path(r"\\?\UNC\server\share\git")
    monkeypatch.setattr(
        install_module, "installed_package",
        lambda name, **kw: install_module.InstalledPackage(verbatim, "2.56.1+1", verbatim / "cmd" / "git.exe")
        if name == "git" else None,
    )
    monkeypatch.setenv("PATH", "/usr/bin")

    built = env_module.managed_environment(tmp_path / "venv", python=py, cache=tmp_path / "cache")
    assert r"\\?\\" not in built.env["PATH"]
    assert built.env["PATH"].split(os.pathsep)[0].replace("/", "\\") == r"\\server\share\git\cmd"
