"""Regression tests for the Photon sidecar stale-dependency self-heal.

A `hermes update` that bumps the spectrum-ts pin rewrites the sidecar's
``package-lock.json`` but never reinstalls ``node_modules``, so the sidecar
spawns against stale deps and dies on every reconnect. ``_sidecar_deps_stale``
detects that skew (lockfile newer than npm's install marker) so
``_start_sidecar`` can reinstall before spawning.
"""

from __future__ import annotations

import os
from pathlib import Path

import plugins.platforms.photon.adapter as photon_adapter


def _seed(sidecar: Path, *, lock_mtime: float, marker_mtime: float | None) -> None:
    """Create a fake sidecar dir with a lockfile and (optionally) npm's marker."""
    (sidecar / "node_modules").mkdir(parents=True)
    lock = sidecar / "package-lock.json"
    lock.write_text("{}", encoding="utf-8")
    os.utime(lock, (lock_mtime, lock_mtime))
    if marker_mtime is not None:
        marker = sidecar / "node_modules" / ".package-lock.json"
        marker.write_text("{}", encoding="utf-8")
        os.utime(marker, (marker_mtime, marker_mtime))


def test_stale_when_lockfile_newer_than_marker(tmp_path, monkeypatch) -> None:
    """The update-rewrites-lockfile-but-skips-install case must reinstall."""
    sidecar = tmp_path / "sidecar"
    _seed(sidecar, lock_mtime=2000.0, marker_mtime=1000.0)
    monkeypatch.setattr(photon_adapter, "get_runtime_sidecar_dir", lambda: sidecar)
    assert photon_adapter._sidecar_deps_stale() is True


def test_fresh_when_marker_newer_than_lockfile(tmp_path, monkeypatch) -> None:
    """A normal install (marker at/after lockfile) must NOT trigger a reinstall."""
    sidecar = tmp_path / "sidecar"
    _seed(sidecar, lock_mtime=1000.0, marker_mtime=2000.0)
    monkeypatch.setattr(photon_adapter, "get_runtime_sidecar_dir", lambda: sidecar)
    assert photon_adapter._sidecar_deps_stale() is False


def test_not_stale_when_marker_missing(tmp_path, monkeypatch) -> None:
    """No marker (first run / unreadable) must fail safe to False, never block start."""
    sidecar = tmp_path / "sidecar"
    _seed(sidecar, lock_mtime=2000.0, marker_mtime=None)
    monkeypatch.setattr(photon_adapter, "get_runtime_sidecar_dir", lambda: sidecar)
    assert photon_adapter._sidecar_deps_stale() is False
