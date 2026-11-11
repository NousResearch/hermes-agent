"""prepare_launch() recognizes the store interpreter it would re-exec into (#122513).

PM spells the store Python through HERMES_HOME, which may contain '..', while
the OS reports ``sys.executable`` normalized. The identity check must be
lexical: comparing the raw spellings relaunched every child forever, and
following symlinks would treat a venv interpreter as the store interpreter.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hermes_cli import venv_sync


@pytest.fixture
def launch(tmp_path, monkeypatch):
    import pm
    from hermes_cli import _launchers

    root = tmp_path / "checkout"
    (root / ".git").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='example'\n", encoding="utf-8")
    (root / "install-stamp.json").write_text('{"updateMechanism": "self"}', encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    monkeypatch.setattr(pm, "venv_is_current", lambda **kw: True)
    store_python = tmp_path / "store" / "cpython-3.14" / "bin" / "python3"
    store_python.parent.mkdir(parents=True)
    store_python.write_text("#!/bin/sh\n", encoding="utf-8")
    published = []
    monkeypatch.setattr(venv_sync, "publish_launchers", published.append)

    def run(*, store_spelling: Path, executable: Path):
        monkeypatch.setattr(_launchers, "resolve_store_python", lambda _: store_spelling)
        monkeypatch.setattr(sys, "executable", str(executable))
        return venv_sync.prepare_launch(root, []), published

    return tmp_path, store_python, run


def test_store_python_spelled_through_dotdot_home_is_current(launch):
    """HERMES_HOME=<x>/work/../store...: the normalized sys.executable is the same interpreter."""
    tmp_path, store_python, run = launch
    dotted = tmp_path / "work" / ".." / store_python.relative_to(tmp_path)
    (tmp_path / "work").mkdir()

    assert run(store_spelling=dotted, executable=store_python) == (None, [])


@pytest.mark.platforms("posix")
def test_venv_python_symlinked_to_the_store_binary_still_relaunches(launch):
    """Same binary, different interpreter: the venv carries its own sys.prefix."""
    tmp_path, store_python, run = launch
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(store_python)

    target, published = run(store_spelling=store_python, executable=venv_python)
    assert target == store_python
    assert published == [(tmp_path / "checkout").resolve()]
