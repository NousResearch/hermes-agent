"""E2E: the workspace union must INSTALL member deps, not just lock them.

Probed live 2026-09-03: plain `uv sync --frozen` installs only the ROOT
project's dependencies — workspace-member deps are locked by `uv lock`
but never reach site-packages. The union's whole contract is that plugin
deps ride the venv, so lock_and_sync must pass --all-packages. This test
builds a REAL mini-workspace and asserts the member's dep is importable
after lock_and_sync — the exact failure the probe caught (green-looking
lock, empty site-packages) can never silently return.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import pm.workspace as ws


def _uv_available() -> bool:
    import shutil

    return shutil.which("uv") is not None


@pytest.fixture
def mini_workspace(tmp_path, monkeypatch):
    """A real generated workspace: fake core pyproject + a plugin member
    whose dep (pyfiglet) is NOT a root dep. Store paths pointed at tmp."""
    core = tmp_path / "core"
    core.mkdir()
    (core / "pyproject.toml").write_text(
        "[project]\n"
        'name = "fake-core"\n'
        'version = "0.1.0"\n'
        'requires-python = ">=3.11"\n'
        'dependencies = []\n',
        encoding="utf-8",
    )
    plug = tmp_path / "plugins" / "member-plug"
    plug.mkdir(parents=True)
    (plug / "pyproject.toml").write_text(
        "[project]\n"
        'name = "member-plug"\n'
        'version = "0.1.0"\n'
        'requires-python = ">=3.11"\n'
        'dependencies = ["pyfiglet==1.0.2"]\n',
        encoding="utf-8",
    )
    store = tmp_path / "store"
    store.mkdir()
    venv = tmp_path / "venv"

    import pm.paths

    monkeypatch.setattr(pm.paths, "repo_root", lambda: core)
    monkeypatch.setattr(pm.paths, "store_root", lambda: store)
    monkeypatch.setattr(ws.paths, "repo_root", lambda: core)
    monkeypatch.setattr(ws.paths, "store_root", lambda: store)
    return tmp_path, plug, venv


@pytest.mark.skipif(_uv_available() is False, reason="uv not on PATH")
def test_lock_and_sync_installs_member_deps(mini_workspace):
    """The probe scenario: after lock_and_sync, the member's dep MUST be
    importable from the synced venv. Under plain `uv sync --frozen` the
    lock contains the dep but site-packages does not — this fails."""
    _, plug, venv = mini_workspace
    ws.lock_and_sync([plug], [], venv_dir=venv)

    site = venv / "Lib" if (venv / "Lib").exists() else venv / "lib"
    packages = sorted(p.name for p in site.glob("*site-packages"))
    assert packages, "no site-packages in the synced venv"
    sp = site / packages[0]

    # the member's dep landed (this is the --all-packages contract)
    assert (sp / "pyfiglet").is_dir() or any(
        p.name.startswith("pyfiglet") for p in sp.iterdir()
    ), (
        "member dep locked but NOT installed — lock_and_sync must pass "
        "--all-packages (plain `uv sync --frozen` is root-only)"
    )


@pytest.mark.skipif(_uv_available() is False, reason="uv not on PATH")
def test_union_survives_a_resync(mini_workspace):
    """The prune-proof contract: a second lock_and_sync (what an update
    rebuild does) must NOT remove the member's deps — they are in the
    union lock, not pip-guests."""
    _, plug, venv = mini_workspace
    ws.lock_and_sync([plug], [], venv_dir=venv)
    ws.lock_and_sync([plug], [], venv_dir=venv)

    site = venv / "Lib" if (venv / "Lib").exists() else venv / "lib"
    sp = site / next(iter(sorted(p.name for p in site.glob("*site-packages"))))
    assert any(p.name.startswith("pyfiglet") for p in sp.iterdir()), (
        "member deps were stripped by a re-sync — the union lock must own "
        "them across rebuilds"
    )
