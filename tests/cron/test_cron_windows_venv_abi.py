"""Windows cron scripts overlay the committed generation, never the stale in-tree venv.

``cron/scheduler_script.py::_windows_cron_python_invocation`` answers "which interpreter and
which site-packages?" for a cron script child. On a PM-managed install it read the dependency
tree through ``pm.environments.selected_venv``, which falls back to ``base_venv`` and answers
the leftover pre-PM ``<root>/venv`` when no generation is recorded. That tree is built for
whichever interpreter created it, so overlaying it on the managed store Python loads a cp311
``pydantic_core`` on 3.14 and every script dies with
``ModuleNotFoundError: No module named 'pydantic_core._pydantic_core'`` — the cron sibling of
the gateway crash in #122183/#123650, which no PR covered.

Invariant: with a generation committed, that generation is the tree the child gets — never the
in-tree venv.
"""

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.platforms("windows")


def _write_venv(venv: Path) -> Path:
    """A fake Windows venv with the ``Scripts/python.exe`` shim a cron child is handed."""
    (venv / "Lib" / "site-packages").mkdir(parents=True, exist_ok=True)
    (venv / "Scripts").mkdir(parents=True, exist_ok=True)
    (venv / "Scripts" / "python.exe").write_text("", encoding="utf-8")
    return venv


def _site_packages_in(env_overlay: dict) -> Path | None:
    for entry in env_overlay.get("PYTHONPATH", "").split(os.pathsep):
        if Path(entry).name == "site-packages":
            return Path(entry)
    return None


def test_managed_install_uses_the_committed_generation_not_the_in_tree_venv(
    tmp_path, monkeypatch,
):
    """The committed generation wins; the leftover pre-PM venv never reaches the child."""
    from cron import scheduler_script as sched_script

    stale = _write_venv(tmp_path / "repo" / "venv")
    committed = _write_venv(tmp_path / "installs" / "gen" / "venv")
    store = tmp_path / "store" / "python.exe"
    store.parent.mkdir(parents=True)
    store.write_text("", encoding="utf-8")
    child = _write_venv(tmp_path / "child")

    monkeypatch.setattr("hermes_cli._launchers.resolve_store_python", lambda _root: store)
    monkeypatch.setattr("pm.environments.committed_venv", lambda _root: committed)
    # Base read selected_venv; pointing it at the stale venv is what makes this red there.
    monkeypatch.setattr("pm.environments.selected_venv", lambda _root: stale)

    interpreter, env_overlay = sched_script._windows_cron_python_invocation(
        str(child / "Scripts" / "python.exe")
    )

    assert interpreter == str(store)
    overlay = _site_packages_in(env_overlay)
    assert overlay is not None, "the committed generation must supply a site-packages entry"
    assert overlay == committed / "Lib" / "site-packages"
    assert not overlay.is_relative_to(stale)



def test_uv_managed_base_interpreter_falls_back_to_virtual_env(tmp_path, monkeypatch):
    """#153043: a gateway started from a uv-managed base interpreter keeps its venv packages.

    ``interpreter.parent.parent`` is the uv Python directory there, which has no ``pyvenv.cfg``;
    the launcher's ``VIRTUAL_ENV`` names the real venv, so the overlay must still be produced.
    """
    from cron import scheduler_script as sched_script

    uv_python_dir = tmp_path / "uv-python" / "cpython-3.12"
    uv_python_dir.mkdir(parents=True)
    base_python = uv_python_dir / "python.exe"
    base_python.write_text("", encoding="utf-8")

    venv = _write_venv(tmp_path / "venv")
    (venv / "pyvenv.cfg").write_text(f"home = {uv_python_dir}\nuv = 0.9.0\n", encoding="utf-8")

    monkeypatch.setattr("hermes_cli._launchers.resolve_store_python", lambda _root: None)
    monkeypatch.setattr("pm.environments.committed_venv", lambda _root: None)
    monkeypatch.setenv("VIRTUAL_ENV", str(venv))

    interpreter, env_overlay = sched_script._windows_cron_python_invocation(str(base_python))

    assert interpreter == str(base_python)
    assert env_overlay.get("VIRTUAL_ENV") == str(venv)
    overlay = _site_packages_in(env_overlay)
    assert overlay == venv / "Lib" / "site-packages"


def test_virtual_env_from_an_unrelated_venv_is_ignored(tmp_path, monkeypatch):
    """A stray VIRTUAL_ENV must not leak another environment's packages into the cron child."""
    from cron import scheduler_script as sched_script

    uv_python_dir = tmp_path / "uv-python" / "cpython-3.12"
    uv_python_dir.mkdir(parents=True)
    base_python = uv_python_dir / "python.exe"
    base_python.write_text("", encoding="utf-8")

    other_base = tmp_path / "other-python"
    other_base.mkdir()
    (other_base / "python.exe").write_text("", encoding="utf-8")
    unrelated = _write_venv(tmp_path / "unrelated")
    (unrelated / "pyvenv.cfg").write_text(f"home = {other_base}\nuv = 0.9.0\n", encoding="utf-8")

    monkeypatch.setattr("hermes_cli._launchers.resolve_store_python", lambda _root: None)
    monkeypatch.setattr("pm.environments.committed_venv", lambda _root: None)
    monkeypatch.setenv("VIRTUAL_ENV", str(unrelated))

    interpreter, env_overlay = sched_script._windows_cron_python_invocation(str(base_python))

    assert interpreter == str(base_python)
    assert "VIRTUAL_ENV" not in env_overlay
    assert "PYTHONPATH" not in env_overlay
