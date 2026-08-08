"""Photon's part of the sidecar resolver.

The rungs themselves are covered by tests/gateway/test_sidecar_runtime.py.
What is Photon-specific, and tested here: the legacy ``PHOTON_SIDECAR_DIR``
override, and that neither the adapter nor the CLI resolves at import time.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import plugins.platforms.photon.sidecar_paths as sidecar_paths


def test_legacy_override_still_works(tmp_path, monkeypatch) -> None:
    """``PHOTON_SIDECAR_DIR`` predates the shared resolver.

    Operators set it, so it keeps working alongside the shared
    ``HERMES_PHOTON_SIDECAR_DIR``.
    """
    override = tmp_path / "custom"
    monkeypatch.delenv("HERMES_PHOTON_SIDECAR_DIR", raising=False)
    monkeypatch.setenv("PHOTON_SIDECAR_DIR", str(override))
    assert sidecar_paths.resolve_sidecar_dir(tmp_path / "src") == override


def test_shared_override_wins_over_the_legacy_one(tmp_path, monkeypatch) -> None:
    """Both set means the operator moved to the shared name. Honour it."""
    shared = tmp_path / "shared"
    monkeypatch.setenv("PHOTON_SIDECAR_DIR", str(tmp_path / "legacy"))
    monkeypatch.setenv("HERMES_PHOTON_SIDECAR_DIR", str(shared))
    assert sidecar_paths.resolve_sidecar_dir(tmp_path / "src") == shared


def test_the_source_dir_is_the_shipped_sidecar(monkeypatch) -> None:
    """A resolve with no argument reads the sidecar inside the plugin."""
    monkeypatch.delenv("PHOTON_SIDECAR_DIR", raising=False)
    monkeypatch.delenv("HERMES_PHOTON_SIDECAR_DIR", raising=False)
    seen = {}

    def _record(name, source):
        seen["name"] = name
        seen["source"] = source
        return source

    monkeypatch.setattr(sidecar_paths, "resolve_sidecar", _record)
    sidecar_paths.resolve_sidecar_dir()
    assert seen["name"] == "photon"
    assert seen["source"] == sidecar_paths.SOURCE_SIDECAR_DIR
    assert seen["source"].name == "sidecar"


def test_adapter_import_does_not_resolve_sidecar_dir(monkeypatch) -> None:
    """Importing the adapter must not probe the filesystem or copy files.

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


def test_dir_writable_probe(tmp_path) -> None:
    """Re-exported for the adapter, which gates npm on it."""
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
