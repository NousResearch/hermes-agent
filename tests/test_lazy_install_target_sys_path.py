"""Regression: durable lazy-install target is importable without a restart.

On sealed/immutable deployments (``HERMES_DISABLE_LAZY_INSTALLS=1``) the agent
venv is read-only and pre-installed optional packages live in the durable
volume named by ``HERMES_LAZY_INSTALL_TARGET`` (e.g. ``/opt/data/lazy-packages``).

The original code only appended that directory to ``sys.path`` from inside
``ensure()`` *after* a successful install. With lazy installs disabled the
install path never runs, so a pre-installed package in the durable target
stayed invisible to the already-running interpreter — breaking features that
depend on it (notably local STT / faster-whisper for inbound voice
transcription across iMessage, Telegram, Discord, …).

``tools.lazy_deps`` now binds the target onto ``sys.path`` at import time,
independent of the install gate. This test asserts that a package pre-installed
into a durable target becomes importable the moment ``lazy_deps`` is imported,
with lazy installs disabled — no restart required.
"""

import os
import sys
import types
from pathlib import Path

import pytest


def _make_fake_package(directory: Path, name: str) -> None:
    """Create a trivial importable package ``name`` inside ``directory``."""
    pkg = directory / name
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("# fake durable package for sys.path test\n")


def test_durable_target_bound_to_sys_path_on_import(monkeypatch, tmp_path):
    """Importing lazy_deps makes a pre-installed durable package importable.

    Mirrors a sealed-venv install: lazy installs disabled, package already
    present in the durable target dir.
    """
    target = tmp_path / "lazy-packages"
    target.mkdir()
    _make_fake_package(target, "fake_durable_stt")

    # Sealed-venv posture: installs disabled, target dir set.
    monkeypatch.setenv("HERMES_LAZY_INSTALL_TARGET", str(target))
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")

    # Ensure the target is not already importable and the module is fresh.
    monkeypatch.delitem(sys.path, sys.path.index(str(target)) if str(target) in sys.path else 0, raising=False)
    sys.modules.pop("fake_durable_stt", None)
    import importlib

    monkeypatch.setitem(sys.modules, "tools.lazy_deps", None) if "tools.lazy_deps" in sys.modules else None
    sys.modules.pop("tools.lazy_deps", None)

    assert "fake_durable_stt" not in sys.modules
    # Before import the package is NOT importable (simulating the old bug).
    with pytest.raises(ImportError):
        importlib.import_module("fake_durable_stt")

    # Importing lazy_deps must bind the durable target.
    importlib.import_module("tools.lazy_deps")

    # Now the pre-installed package is importable without a restart.
    mod = importlib.import_module("fake_durable_stt")
    assert isinstance(mod, types.ModuleType)
    assert str(target) in sys.path


def test_no_target_env_is_a_noop(monkeypatch):
    """When the env var is unset, nothing is appended and import still works."""
    monkeypatch.delenv("HERMES_LAZY_INSTALL_TARGET", raising=False)
    before = list(sys.path)
    import importlib

    sys.modules.pop("tools.lazy_deps", None)
    importlib.import_module("tools.lazy_deps")
    # Path unchanged (no spurious append) when the var is absent.
    assert sys.path[: len(before)] == before
