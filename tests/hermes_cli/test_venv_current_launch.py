"""A launch through the venv's symlinked interpreter is current, not stale.

PM-managed venvs expose ``bin/python`` as a symlink into the store, so
``sys.executable`` never string-equals the store path even when both point
at the same binary. The identity check must follow symlinks, or every venv
launch pays a pointless ``os.execv`` into isolated mode (#122513).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hermes_cli import venv_sync


def _self_checkout(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    (root / ".git").mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='example'\n")
    (root / "install-stamp.json").write_text('{"updateMechanism": "self"}')
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    return root


@pytest.fixture
def publication(monkeypatch):
    calls = []
    monkeypatch.setattr(venv_sync, "publish_launchers", lambda root: calls.append(root))
    return calls


def test_symlinked_venv_python_is_current_and_never_relaunches(
    tmp_path, monkeypatch, publication
):
    """``venv/bin/python -> store/…/bin/python3``: same binary, no re-exec."""
    import pm
    from hermes_cli import _launchers

    root = _self_checkout(tmp_path, monkeypatch)
    store = tmp_path / "store" / "cpython-3.14"
    (store / "bin").mkdir(parents=True)
    store_python = store / "bin" / "python3"
    store_python.write_text("#!/bin/sh\n")
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    venv_python = venv_bin / "python"
    venv_python.symlink_to(store_python)

    monkeypatch.setattr(pm, "venv_is_current", lambda **kw: True)
    monkeypatch.setattr(_launchers, "resolve_store_python", lambda _: store_python)
    monkeypatch.setattr(sys, "executable", str(venv_python))

    assert venv_sync.prepare_launch(root, []) is None
    assert publication == [], (
        "a current symlinked venv launch must not republish launchers"
    )


def test_stale_interpreter_still_relaunches(tmp_path, monkeypatch, publication):
    """A different binary behind the venv symlink must still re-exec into the store."""
    import pm
    from hermes_cli import _launchers

    root = _self_checkout(tmp_path, monkeypatch)
    store = tmp_path / "store" / "cpython-3.14"
    (store / "bin").mkdir(parents=True)
    store_python = store / "bin" / "python3"
    store_python.write_text("#!/bin/sh\n")
    other = tmp_path / "other-python"
    other.write_text("#!/bin/sh\n")

    monkeypatch.setattr(pm, "venv_is_current", lambda **kw: True)
    monkeypatch.setattr(_launchers, "resolve_store_python", lambda _: store_python)
    monkeypatch.setattr(sys, "executable", str(other))

    assert venv_sync.prepare_launch(root, []) == store_python
    assert publication == [root]
