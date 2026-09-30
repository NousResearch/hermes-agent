"""Regression tests for ``hermes_cli._launchers.resolve_store_python``.

The dispatcher contract: ``resolve_store_python`` MUST return ``None`` when
``facts.json`` carries a Python entry whose target directory does not contain
an executable interpreter (broken PM download, placeholder version string
with unresolved asterisks, failed extraction). The dispatcher then falls
back to ``sys.executable`` — the venv Python that actually has ``hermes_cli``
importable. Returning a non-executable path instead would crash every
dispatched child with ``ModuleNotFoundError`` until PM repairs the entry.

Behavior contract under test: ``None`` when the entry's interpreter is
missing OR not executable; ``Path`` to the interpreter when it is present
and executable. No hardcoded paths, no snapshot of which Python version PM
recorded.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from hermes_cli import _launchers


def _write_facts(runtime_dir: Path, *, entry: str | None) -> Path:
    """Write a minimal facts.json under ``runtime_dir``.

    Mirrors the package shape ``resolve_store_python`` reads:
    ``{"packages": {"python": {"entry": "<name>"}}}``. With ``entry=None``
    the package is omitted to also exercise that branch.
    """
    payload: dict[str, object] = {"packages": {}}
    if entry is not None:
        payload["packages"] = {"python": {"entry": entry}}
    facts = runtime_dir / "facts.json"
    facts.write_text(json.dumps(payload), encoding="utf-8")
    return facts


def _patch_runtime(monkeypatch, runtime: Path):
    """Redirect ``resolve_store_python``'s ``store_root()`` lookup to ``runtime``.

    ``resolve_store_python`` reads the store location via
    ``pm.environments.store_root``, which does NOT honor ``HERMES_RUNTIME_DIR``
    and always returns a hardcoded path under the user's home. To exercise
    each scenario against a clean per-test directory we monkeypatch the
    name where ``resolve_store_python`` binds it — at the top of
    ``_launchers`` — to a lambda that returns our fixture path. Patching
    there (not in ``pm.environments``) keeps the change local to the module
    we are testing and avoids side effects on other tests that rely on the
    real ``store_root``.
    """
    monkeypatch.setattr(_launchers, "store_root", lambda _repo_root: runtime)


@pytest.mark.platforms("posix")
def test_resolve_store_python_falls_back_on_broken_entry(tmp_path, monkeypatch):
    """``facts.json`` with an entry whose target dir has no ``bin/python3``.

    Reproduces the production bug: PM recorded an entry
    (``python-3.14.7+202****0901-linux-x64``) whose directory does not exist
    or lacks ``bin/python3``. ``resolve_store_python`` MUST return ``None``
    so the dispatcher falls back to the venv's ``sys.executable``.
    """
    runtime = tmp_path / "tools"
    runtime.mkdir()
    _write_facts(runtime, entry="python-3.14.7+202****0901-linux-x64")

    _patch_runtime(monkeypatch, runtime)

    assert _launchers.resolve_store_python(tmp_path / "checkout") is None


@pytest.mark.platforms("posix")
def test_resolve_store_python_returns_none_when_interpreter_not_executable(
    tmp_path, monkeypatch
):
    """``facts.json`` points at a directory whose ``bin/python3`` is non-executable.

    Some PM runs leave a stub file at the canonical path but chmod it
    non-executable after a failed extraction. ``is_file()`` is insufficient
    — we must also require execute permission, otherwise the dispatcher
    will still crash trying to spawn it.
    """
    runtime = tmp_path / "tools"
    runtime.mkdir()
    entry = "python-broken-no-exec-bit"
    entry_dir = runtime / entry
    interp = entry_dir / "bin" / "python3"
    interp.parent.mkdir(parents=True)
    interp.write_text("#!/bin/sh\necho stub\n", encoding="utf-8")
    interp.chmod(interp.stat().st_mode & ~stat.S_IXUSR & ~stat.S_IXGRP & ~stat.S_IXOTH)
    _write_facts(runtime, entry=entry)

    _patch_runtime(monkeypatch, runtime)

    assert _launchers.resolve_store_python(tmp_path / "checkout") is None


@pytest.mark.platforms("posix")
def test_resolve_store_python_returns_interpreter_when_executable(
    tmp_path, monkeypatch
):
    """Sanity guard: a valid entry still resolves to its interpreter.

    Pins the contract the dispatcher relies on (positive case); guards
    against an over-eager fix that starts returning ``None`` even for
    good stores.
    """
    runtime = tmp_path / "tools"
    runtime.mkdir()
    entry = "python-good-3.11"
    interp = runtime / entry / "bin" / "python3"
    interp.parent.mkdir(parents=True)
    interp.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    interp.chmod(0o755)
    _write_facts(runtime, entry=entry)

    _patch_runtime(monkeypatch, runtime)

    assert _launchers.resolve_store_python(tmp_path / "checkout") == interp


@pytest.mark.platforms("posix")
def test_resolve_store_python_returns_none_when_no_facts_file(
    tmp_path, monkeypatch
):
    """No ``facts.json`` at all — same outcome as a broken entry."""
    runtime = tmp_path / "tools"
    runtime.mkdir()

    _patch_runtime(monkeypatch, runtime)

    assert _launchers.resolve_store_python(tmp_path / "checkout") is None