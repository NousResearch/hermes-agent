"""LSP python discovery must resolve the pm store interpreter, not sniff
VIRTUAL_ENV.

Under no-boot-through-venv the process is the store python
(``sys.prefix == sys.base_prefix``) and ``VIRTUAL_ENV`` is unset in bundled
installs, so the old ambient-env probe silently degraded to project-dir
probing. pm's ``python`` fact (facts.json + store layout) is the authority.
"""

import json
import os
import sys
from pathlib import Path

from agent.lsp.servers import _detect_python, _pm_store_python


def _seed_pm_python(tmp_path, monkeypatch, entry="python-3.11.13"):
    """Stage a pm bundled-install layout: HERMES_RUNTIME_DIR -> store with a
    manifest sibling (bundled), a python entry, and facts recording it."""
    payload = tmp_path / "payload"
    store = payload / "tools"
    python_entry = store / entry
    python_entry.mkdir(parents=True)
    exe = python_entry / ("python.exe" if sys.platform == "win32" else "bin/python3")
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_text("", encoding="utf-8")
    (payload / "manifest.json").write_text("{}", encoding="utf-8")
    (store / "facts.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "packages": {"python": {"entry": entry, "version": "3.11.13"}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(store))
    return exe


def test_pm_store_python_resolved_without_virtual_env(tmp_path, monkeypatch):
    """No VIRTUAL_ENV anywhere — the pm store interpreter is the answer."""
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    exe = _seed_pm_python(tmp_path, monkeypatch)

    assert _pm_store_python() == str(exe)
    assert _detect_python(str(tmp_path / "some-workspace")) == str(exe)


def test_no_pm_python_falls_back_to_project_layouts(tmp_path, monkeypatch):
    """Without a pm python fact, project .venv candidates still resolve."""
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "no-such-store"))

    dot_venv_bin = tmp_path / "proj" / ".venv" / "bin"
    dot_venv_bin.mkdir(parents=True)
    python = dot_venv_bin / "python"
    python.write_text("", encoding="utf-8")

    assert Path(_detect_python(str(tmp_path / "proj"))) == python


def test_missing_store_entry_is_not_an_answer(tmp_path, monkeypatch):
    """A python fact whose store entry is gone resolves to None (pm only
    vouches for what is on disk)."""
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    payload = tmp_path / "payload"
    store = payload / "tools"
    store.mkdir(parents=True)
    (payload / "manifest.json").write_text("{}", encoding="utf-8")
    (store / "facts.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "packages": {"python": {"entry": "python-3.11.13", "version": "3.11.13"}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(store))

    assert _pm_store_python() is None
