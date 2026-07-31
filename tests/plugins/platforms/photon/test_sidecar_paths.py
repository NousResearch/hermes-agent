"""Tests for the Photon sidecar directory resolver (NS-606).

Hosted/managed images keep the plugin tree under an immutable
``/opt/hermes``; ``resolve_sidecar_dir`` must run in place when the deps are
baked and current, and mirror the sidecar to the writable ``HERMES_HOME``
volume when a runtime install is unavoidable.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import plugins.platforms.photon.sidecar_paths as sidecar_paths


def _seed_source(source: Path, *, with_node_modules: bool = False) -> None:
    source.mkdir(parents=True, exist_ok=True)
    for name in sidecar_paths._MIRROR_FILES:
        (source / name).write_text(f"// {name}\n", encoding="utf-8")
    if with_node_modules:
        (source / "node_modules").mkdir()
        (source / "node_modules" / ".package-lock.json").write_text(
            "{}", encoding="utf-8"
        )


def _freeze_writability(monkeypatch, *, writable: bool) -> None:
    monkeypatch.setattr(sidecar_paths, "_dir_writable", lambda _p: writable)


def test_env_override_wins(tmp_path, monkeypatch) -> None:
    override = tmp_path / "custom"
    monkeypatch.setenv("PHOTON_SIDECAR_DIR", str(override))
    assert sidecar_paths.resolve_sidecar_dir(tmp_path / "src") == override


def test_writable_source_runs_in_place(tmp_path, monkeypatch) -> None:
    """Dev installs: writable tree keeps today's behavior exactly."""
    monkeypatch.delenv("PHOTON_SIDECAR_DIR", raising=False)
    source = tmp_path / "src"
    _seed_source(source)
    _freeze_writability(monkeypatch, writable=True)
    assert sidecar_paths.resolve_sidecar_dir(source) == source


def test_readonly_source_with_baked_fresh_deps_runs_in_place(
    tmp_path, monkeypatch
) -> None:
    """Managed-image happy path: deps baked at build time, no mirror needed."""
    monkeypatch.delenv("PHOTON_SIDECAR_DIR", raising=False)
    source = tmp_path / "src"
    _seed_source(source, with_node_modules=True)
    # Marker newer than lockfile == fresh install.
    lock = source / "package-lock.json"
    marker = source / "node_modules" / ".package-lock.json"
    os.utime(lock, (1000.0, 1000.0))
    os.utime(marker, (2000.0, 2000.0))
    _freeze_writability(monkeypatch, writable=False)
    assert sidecar_paths.resolve_sidecar_dir(source) == source


def test_mirror_refresh_updates_changed_files_and_keeps_node_modules(
    tmp_path, monkeypatch
) -> None:
    """Image update changes index.mjs → re-copied; installed deps survive."""
    monkeypatch.delenv("PHOTON_SIDECAR_DIR", raising=False)
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    source = tmp_path / "src"
    _seed_source(source)
    _freeze_writability(monkeypatch, writable=False)

    mirror = sidecar_paths.resolve_sidecar_dir(source)
    # Simulate a completed npm install in the mirror.
    (mirror / "node_modules").mkdir()
    (mirror / "node_modules" / "installed.txt").write_text("x", encoding="utf-8")

    # Image update rewrites a source file.
    (source / "index.mjs").write_text("// index.mjs v2\n", encoding="utf-8")

    resolved = sidecar_paths.resolve_sidecar_dir(source)

    assert resolved == mirror
    assert (mirror / "index.mjs").read_text(encoding="utf-8") == "// index.mjs v2\n"
    assert (mirror / "node_modules" / "installed.txt").exists()


def test_dir_writable_probe(tmp_path) -> None:
    assert sidecar_paths.dir_writable(tmp_path) is True
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o555)
    try:
        if os.geteuid() == 0:  # pragma: no cover - root ignores perms
            pytest.skip("root bypasses directory permissions")
        assert sidecar_paths.dir_writable(ro) is False
    finally:
        ro.chmod(0o755)


def test_adapter_import_does_not_resolve_sidecar_dir(monkeypatch) -> None:
    """Importing the adapter must not probe the filesystem or mirror files.

    resolve_sidecar_dir() touch/unlink-probes the source tree and may copy
    files to HERMES_HOME; the adapter and CLI resolve lazily on first use so
    a bare import (plugin discovery, `hermes --help`, test collection) has
    no filesystem side effects.
    """
    import importlib

    from plugins.platforms.photon import adapter as photon_adapter
    from plugins.platforms.photon import cli as photon_cli

    def _boom(*args, **kwargs):  # pragma: no cover - failure path
        raise AssertionError("resolve_sidecar_dir called at import time")

    monkeypatch.setattr(sidecar_paths, "resolve_sidecar_dir", _boom)
    try:
        importlib.reload(photon_adapter)
        importlib.reload(photon_cli)
        # Nothing resolved yet.
        assert photon_adapter._SIDECAR_DIR is None
        assert photon_cli._SIDECAR_DIR is None
        # First real use resolves (and would call resolve_sidecar_dir).
        with pytest.raises(AssertionError, match="import time"):
            photon_adapter._sidecar_dir()
        # A monkeypatched _SIDECAR_DIR (the pattern existing tests use) is
        # honored without touching the resolver.
        monkeypatch.setattr(photon_adapter, "_SIDECAR_DIR", Path("/tmp/x"))
        assert photon_adapter._sidecar_dir() == Path("/tmp/x")
        assert photon_adapter._npm_error_log() == Path("/tmp/x/.photon-npm-error.log")
    finally:
        # Restore real bindings for any later test importing these modules.
        monkeypatch.undo()
        importlib.reload(photon_adapter)
        importlib.reload(photon_cli)
