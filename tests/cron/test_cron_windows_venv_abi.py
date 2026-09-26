"""Windows cron scripts resolve their dependency tree, and fall through when there is none.

``cron/scheduler_script.py::_windows_cron_python_invocation`` answers "which interpreter and
which site-packages?" for a cron script child. On a PM-managed install it read the dependency
tree through ``pm.environments.selected_venv``, which falls back to ``base_venv`` and answers
the leftover pre-PM ``<root>/venv`` when no generation is recorded. That tree is built for
whichever interpreter created it, so overlaying it on the managed store Python loads a cp311
``pydantic_core`` on 3.14 and every script dies with
``ModuleNotFoundError: No module named 'pydantic_core._pydantic_core'`` — the cron sibling of
the gateway crash in #122183/#123650, which no PR covered.

Two invariants:

1. With a generation committed, that generation is the tree the child gets — never the in-tree
   venv.
2. With NO generation, or a record too corrupt to read, the managed branch falls through to the
   uv overlay: the handed venv keeps its own interpreter and its own site-packages. Handing back
   the bare store Python instead would be worse than main (``_require_own_dependencies`` refuses
   it with "hermes pm repair" and a cron script has no bootstrap to refuse cleanly), and it
   strips the overlay's site-packages entry, warning on every spawn.
"""

import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.platforms("windows")


def _write_venv(venv: Path, base: Path, *, version: str | None = None) -> Path:
    """A fake Windows venv with the ``Scripts/python.exe`` shim a cron child is handed."""
    (venv / "Lib" / "site-packages").mkdir(parents=True, exist_ok=True)
    (venv / "Scripts").mkdir(parents=True, exist_ok=True)
    base.mkdir(parents=True, exist_ok=True)
    (venv / "Scripts" / "python.exe").write_text("", encoding="utf-8")
    (base / "python.exe").write_text("", encoding="utf-8")
    lines = [f"home = {base}", "uv = 0.11.14"]
    if version:
        lines.append(f"version_info = {version}")
    (venv / "pyvenv.cfg").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return venv


def _site_packages_in(env_overlay: dict) -> Path | None:
    for entry in env_overlay.get("PYTHONPATH", "").split(os.pathsep):
        if Path(entry).name == "site-packages":
            return Path(entry)
    return None


def _raise_runtime_error(_root):
    raise RuntimeError("cannot read dependency environment: facts.json")


def test_managed_install_uses_the_committed_generation_not_the_in_tree_venv(
    tmp_path, monkeypatch,
):
    """The committed generation wins; the leftover pre-PM venv never reaches the child."""
    from cron import scheduler_script as sched_script

    stale = _write_venv(tmp_path / "repo" / "venv", tmp_path / "base")
    committed = _write_venv(
        tmp_path / "installs" / "gen" / "venv", tmp_path / "genbase",
        version=f"{sys.version_info[0]}.{sys.version_info[1]}.7",
    )
    store = tmp_path / "store" / "python.exe"
    store.parent.mkdir(parents=True)
    store.write_text("", encoding="utf-8")
    child = _write_venv(tmp_path / "child", tmp_path / "childbase")

    monkeypatch.setattr("hermes_cli._launchers.resolve_store_python", lambda _root: store)
    monkeypatch.setattr("pm.environments.committed_venv", lambda _root: committed)
    monkeypatch.setattr("pm.environments.selected_venv", lambda _root: stale)

    interpreter, env_overlay = sched_script._windows_cron_python_invocation(
        str(child / "Scripts" / "python.exe")
    )

    assert interpreter == str(store)
    overlay = _site_packages_in(env_overlay)
    assert overlay is not None, "the committed generation must supply a site-packages entry"
    assert overlay == committed / "Lib" / "site-packages"
    assert not overlay.is_relative_to(stale)


def test_managed_install_without_a_generation_falls_through_to_the_handed_venv(
    tmp_path, monkeypatch,
):
    """No generation, or an unreadable record: the handed venv keeps its own packages.

    Both arms must land identically — the child gets its own base interpreter and its own
    site-packages, not the bare store Python, which cannot serve a script with nothing
    committed. Positive control for the fall-through: the handed venv here IS a uv venv
    carrying its own site-packages, exactly the case that worked on main.
    """
    from cron import scheduler_script as sched_script

    stale = _write_venv(tmp_path / "repo" / "venv", tmp_path / "base")
    store = tmp_path / "store" / "python.exe"
    store.parent.mkdir(parents=True)
    store.write_text("", encoding="utf-8")

    for label, committed in (("none", lambda _root: None),
                             ("unreadable", _raise_runtime_error)):
        case = tmp_path / label
        child = _write_venv(case / "child", case / "childbase")
        monkeypatch.setattr("hermes_cli._launchers.resolve_store_python",
                            lambda _root, s=store: s)
        monkeypatch.setattr("pm.environments.committed_venv", committed)
        monkeypatch.setattr("pm.environments.selected_venv", lambda _root, t=stale: t)

        interpreter, env_overlay = sched_script._windows_cron_python_invocation(
            str(child / "Scripts" / "python.exe")
        )

        assert interpreter == str(case / "childbase" / "python.exe"), label
        assert env_overlay["VIRTUAL_ENV"] == str(child), label
        overlay = _site_packages_in(env_overlay)
        assert overlay is not None, f"{label}: the overlay must carry site-packages"
        assert overlay == child / "Lib" / "site-packages", label
        assert not overlay.is_relative_to(stale), label
        assert interpreter != str(store), label
