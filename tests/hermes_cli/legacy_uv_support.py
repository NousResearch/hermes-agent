"""Fixtures shared by the legacy-uv cleanup and doctor-report suites.

The two consumer suites mirror the split in production: ``legacy_uv`` owns the
deletion mechanics and ``doctor_state`` owns what the user is told. Both need the
same pre-PM binary shapes, so the shape builders and the temp-home fixture live
here instead of in either suite.
"""
from __future__ import annotations

import stat
from pathlib import Path

import pytest

from hermes_cli import doctor


def _uv(path: Path, body: str = "hermes") -> None:
    path.write_text(f"#!/usr/bin/env sh\n# {body}\nexit 0\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _uv_binary(path: Path) -> None:
    """The pre-PM ``uv`` shape: a full astral build carrying uv's own marker."""
    path.write_bytes(b"\x7fELF" + b"\0" * (2 << 20) + b"UV_CACHE_DIR")
    path.chmod(0o755)


def _uvx_binary(path: Path) -> None:
    """The real pre-PM ``uvx``: astral's thin launcher (~330 KiB, its own message,
    none of ``uv``'s markers) — the shape a ``_uv_binary`` fixture would miss."""
    path.write_bytes(
        b"\x7fELF" + b"\0" * (200 << 10)
        + b"Could not determine the location of the `uvx` binary"
    )
    path.chmod(0o755)


def _uv_binary_with_ctrl_z(path: Path, *, magic: bytes = b"\x7fELF") -> None:
    """A pre-PM ``uv`` whose identity marker sits BEYOND a ``0x1A`` byte.

    The shipped fixtures carry no Ctrl-Z, so a text-mode descriptor still reads
    to the end and every assertion passes on Windows — which is exactly how the
    missing ``O_BINARY`` stayed invisible. A real ``uv.exe`` has ``0x1A`` at
    offset 2410 and its marker ~38 MB later, so this shape reproduces that order
    at a size the suite can afford.

    The ``0x1A`` is placed EARLY and the marker late on purpose: text mode stops
    at the first ``0x1A``, so the verdict flips to "no identity" there and back
    once the descriptor is opened binary. No ``sys.platform`` fake is involved —
    on POSIX this file is simply scanned to the end either way.
    """
    body = magic + b"\0" * 4096 + b"\x1a" + b"\0" * (2 << 20) + b"UV_CACHE_DIR"
    path.write_bytes(body)
    path.chmod(0o755)


def _pre_pm_binary(path: Path) -> None:
    """The real pre-PM shape for *path*'s name: thin ``uvx``, full ``uv``."""
    (_uvx_binary if "uvx" in path.name.lower() else _uv_binary)(path)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A temp HERMES_HOME with the ``bin/`` dir a launcher install publishes."""
    hermes_home = tmp_path / "hermes-home"
    (hermes_home / "bin").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(doctor, "HERMES_HOME", hermes_home)
    return hermes_home
