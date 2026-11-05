"""Windows cron scripts overlay the committed generation, never the stale in-tree venv.

``cron/scheduler_script.py::_windows_cron_python_invocation`` answers "which interpreter and
which site-packages?" for a cron script child. On a PM-managed install it read the dependency
tree through ``pm.environments.selected_venv``, which falls back to ``base_venv`` and answers
the leftover pre-PM ``<root>/venv`` when no generation is recorded. That tree is built for
whichever interpreter created it, so overlaying it on the managed store Python loads a cp311
``pydantic_core`` on 3.14 and every script dies with
``ModuleNotFoundError: No module named 'pydantic_core._pydantic_core'`` — the cron sibling of
the gateway crash in #122183/#123650, which no PR covered.

Invariants: with a generation committed, that generation is the tree the child gets — never
the in-tree venv — and it is handed over as ``bootstrap_paths`` for in-process ``sys.path``
insertion, never as ``PYTHONPATH`` in the env: everything the script spawns would inherit it
and a foreign interpreter would load the store's compiled extensions (#123440).
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.platforms("windows")


def _write_venv(venv: Path) -> Path:
    """A fake Windows venv with the ``Scripts/python.exe`` shim a cron child is handed."""
    (venv / "Lib" / "site-packages").mkdir(parents=True, exist_ok=True)
    (venv / "Scripts").mkdir(parents=True, exist_ok=True)
    (venv / "Scripts" / "python.exe").write_text("", encoding="utf-8")
    return venv


def test_managed_install_uses_the_committed_generation_not_the_in_tree_venv(
    tmp_path, monkeypatch,
):
    """The committed generation wins; the leftover pre-PM venv never reaches the child —
    and neither tree reaches it via PYTHONPATH (#123440)."""
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

    interpreter, env_overlay, bootstrap_paths = sched_script._windows_cron_python_invocation(
        str(child / "Scripts" / "python.exe")
    )

    assert interpreter == str(store)
    assert "PYTHONPATH" not in env_overlay, (
        "paths must reach the script via the in-process bootstrap — PYTHONPATH in the env "
        "would leak to every child the script spawns (#123440)"
    )
    assert env_overlay.get("HERMES_DISABLE_LAZY_INSTALLS") == "1"
    assert bootstrap_paths is not None, "the committed generation must supply bootstrap paths"
    repo, site_packages_entry = bootstrap_paths
    site_packages = Path(site_packages_entry)
    assert site_packages == committed / "Lib" / "site-packages"
    assert not site_packages.is_relative_to(stale)
    assert Path(repo) == Path(sched_script.__file__).resolve().parents[1], (
        "the bootstrap must receive the live checkout, not the stale tree"
    )
