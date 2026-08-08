"""Regression tests for wave-1 s1 extraction of ``hermes_cli/main.py``.

Clusters moved verbatim (blind implementer w1b):

* ``hermes_cli/bundled_skills_sync.py`` — Termux bundled-skills startup sync
  (``_termux_bundled_skills_*``, ``_sync_bundled_skills_for_startup``,
  ``_termux_should_prefetch_update_check``).
* ``hermes_cli/tui_launch.py`` — npm-workspace / build-input helpers
  (``_workspace_root``, ``_termux_workspace_install_context``,
  ``_tui_need_npm_install``, ``_iter_tui_build_inputs``, ``_tui_need_rebuild``)
  plus their module constants.

Contract under test:

1. Seam identity — ``hermes_cli.main`` re-exports the moved names, so
   ``main_mod.<name>`` resolves to the SAME object as the new module's.
2. Patch liveness — the moved code must keep seeing monkeypatches applied to
   ``hermes_cli.main`` bindings (lazy imports re-read the CLI module at call
   time). The pre-extraction tests patch ``main_mod`` attributes and the
   moved code must behave identically.
3. Behavior — the pure functions keep their exact semantics in their new
   home modules.
4. No back-import — importing the new modules directly must not pull in
   ``hermes_cli.main``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def main_mod():
    import hermes_cli.main as m

    return m


@pytest.fixture
def bss():
    import hermes_cli.bundled_skills_sync as b

    return b


@pytest.fixture
def tui():
    import hermes_cli.tui_launch as t

    return t


# ── seam identity: re-exports are the same objects ──────────────────────


def test_seam_reexports_bundled_skills_names(main_mod, bss):
    for name in (
        "_mark_termux_bundled_skills_synced",
        "_sync_bundled_skills_for_startup",
        "_termux_bundled_skills_fingerprint",
        "_termux_bundled_skills_stamp_path",
        "_termux_bundled_skills_sync_needed",
        "_termux_should_prefetch_update_check",
    ):
        assert getattr(main_mod, name) is getattr(bss, name), name


def test_seam_reexports_tui_launch_names(main_mod, tui):
    for name in (
        "_NPM_LOCK_RUNTIME_KEYS",
        "_TUI_BUILD_INPUT_DIRS",
        "_TUI_BUILD_INPUT_FILES",
        "_TUI_BUILD_INPUT_SUFFIXES",
        "_iter_tui_build_inputs",
        "_termux_workspace_install_context",
        "_tui_need_npm_install",
        "_tui_need_rebuild",
        "_workspace_root",
    ):
        assert getattr(main_mod, name) is getattr(tui, name), name


def test_tui_lock_runtime_keys_value(tui):
    assert tui._NPM_LOCK_RUNTIME_KEYS == {"ideallyInert", "peer"}


# ── bundled skills sync ─────────────────────────────────────────────────


def test_bundled_sync_stamp_path_uses_main_get_hermes_home(
    monkeypatch, main_mod, bss, tmp_path
):
    # Patch-liveness: the pre-extraction test patches main_mod.get_hermes_home
    # and the moved function must see that patch (no real-home writes).
    monkeypatch.setattr(main_mod, "get_hermes_home", lambda: tmp_path)
    assert (
        bss._termux_bundled_skills_stamp_path()
        == tmp_path / "skills" / ".termux_bundled_sync_stamp"
    )


def test_bundled_sync_needed_outside_termux(monkeypatch, bss):
    monkeypatch.delenv("TERMUX_VERSION", raising=False)
    assert bss._termux_bundled_skills_sync_needed() is True


def test_bundled_sync_stamp_fresh_skips(monkeypatch, main_mod, bss, tmp_path):
    monkeypatch.setenv("TERMUX_VERSION", "1")
    monkeypatch.setattr(main_mod, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(bss, "_termux_bundled_skills_fingerprint", lambda: "fp1")
    bss._mark_termux_bundled_skills_synced()
    stamp = tmp_path / "skills" / ".termux_bundled_sync_stamp"
    assert stamp.read_text(encoding="utf-8").strip() == "fp1"
    assert bss._termux_bundled_skills_sync_needed() is False


def test_bundled_sync_stamp_changed_forces_sync(monkeypatch, main_mod, bss, tmp_path):
    monkeypatch.setenv("TERMUX_VERSION", "1")
    monkeypatch.setattr(main_mod, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(bss, "_termux_bundled_skills_fingerprint", lambda: "fp1")
    bss._mark_termux_bundled_skills_synced()
    monkeypatch.setattr(bss, "_termux_bundled_skills_fingerprint", lambda: "fp2")
    assert bss._termux_bundled_skills_sync_needed() is True


def test_bundled_sync_for_startup_stubs_skills_sync(
    monkeypatch, main_mod, bss, tmp_path
):
    calls = []
    monkeypatch.setenv("TERMUX_VERSION", "1")
    monkeypatch.setattr(main_mod, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(bss, "_termux_bundled_skills_fingerprint", lambda: "fp1")
    monkeypatch.setitem(
        sys.modules,
        "tools.skills_sync",
        types.SimpleNamespace(sync_skills=lambda quiet: calls.append(quiet)),
    )
    bss._mark_termux_bundled_skills_synced()
    # Fresh stamp: cheap skip, no sync.
    assert bss._sync_bundled_skills_for_startup() is False
    assert calls == []
    # Stamp removed: one real sync, then re-stamped.
    (tmp_path / "skills" / ".termux_bundled_sync_stamp").unlink()
    assert bss._sync_bundled_skills_for_startup() is True
    assert calls == [True]


def test_bundled_prefetch_update_check(monkeypatch, bss):
    monkeypatch.setenv("TERMUX_VERSION", "1")
    monkeypatch.delenv("HERMES_TERMUX_PREFETCH_UPDATES", raising=False)
    assert bss._termux_should_prefetch_update_check() is False
    monkeypatch.setenv("HERMES_TERMUX_PREFETCH_UPDATES", "1")
    assert bss._termux_should_prefetch_update_check() is True
    monkeypatch.delenv("TERMUX_VERSION")
    assert bss._termux_should_prefetch_update_check() is True


# ── tui launch: workspace root / install check ──────────────────────────


def test_workspace_root_prefers_parent_lockfile(tui, tmp_path):
    sub = tmp_path / "ui-tui"
    sub.mkdir()
    (sub / "package.json").write_text("{}", encoding="utf-8")
    (tmp_path / "package-lock.json").write_text("{}", encoding="utf-8")
    assert tui._workspace_root(sub) == tmp_path


def test_workspace_root_standalone(tui, tmp_path):
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    assert tui._workspace_root(tmp_path) == tmp_path


def test_termux_workspace_install_context(tui, tmp_path):
    sub = tmp_path / "ui-tui"
    sub.mkdir()
    (sub / "package.json").write_text("{}", encoding="utf-8")
    (tmp_path / "package-lock.json").write_text("{}", encoding="utf-8")

    cwd, args = tui._termux_workspace_install_context(sub)
    assert cwd == tmp_path
    assert args == ("--workspace", "ui-tui", "--include-workspace-root=false")

    for child in ("a", "b"):
        pkg = sub / "packages" / child / "package.json"
        pkg.parent.mkdir(parents=True)
        pkg.write_text("{}", encoding="utf-8")
    (sub / "packages" / "b" / "not-pkg.txt").write_text("x", encoding="utf-8")

    cwd, args = tui._termux_workspace_install_context(
        sub, include_child_workspaces=True
    )
    assert cwd == tmp_path
    assert args == (
        "--workspace",
        "ui-tui",
        "--workspace",
        "ui-tui/packages/a",
        "--workspace",
        "ui-tui/packages/b",
        "--include-workspace-root=false",
    )


def test_tui_need_npm_install_missing_ink(tui, tmp_path):
    assert tui._tui_need_npm_install(tmp_path) is True


def test_tui_need_npm_install_matching_locks(tui, tmp_path):
    lock = json.dumps({"packages": {"": {}, "ink": {"version": "1"}}})
    (tmp_path / "package-lock.json").write_text(lock, encoding="utf-8")
    ink = tmp_path / "node_modules" / "@hermes" / "ink" / "package.json"
    ink.parent.mkdir(parents=True)
    ink.write_text("{}", encoding="utf-8")
    (tmp_path / "node_modules" / ".package-lock.json").write_text(
        lock, encoding="utf-8"
    )
    assert tui._tui_need_npm_install(tmp_path) is False


def test_tui_need_npm_install_missing_nonoptional_dep(tui, tmp_path):
    (tmp_path / "package-lock.json").write_text(
        json.dumps({"packages": {"dep": {"version": "2"}}}), encoding="utf-8"
    )
    ink = tmp_path / "node_modules" / "@hermes" / "ink" / "package.json"
    ink.parent.mkdir(parents=True)
    ink.write_text("{}", encoding="utf-8")
    (tmp_path / "node_modules" / ".package-lock.json").write_text(
        json.dumps({"packages": {}}), encoding="utf-8"
    )
    assert tui._tui_need_npm_install(tmp_path) is True
    # optional/peer entries that npm may skip are tolerated.
    (tmp_path / "package-lock.json").write_text(
        json.dumps({"packages": {"dep": {"optional": True}}}), encoding="utf-8"
    )
    assert tui._tui_need_npm_install(tmp_path) is False


def test_tui_need_npm_install_ignores_runtime_keys(tui, tmp_path):
    # ideallyInert is npm's runtime annotation — must not force a reinstall.
    (tmp_path / "package-lock.json").write_text(
        json.dumps({"packages": {"dep": {"version": "1", "ideallyInert": True}}}),
        encoding="utf-8",
    )
    ink = tmp_path / "node_modules" / "@hermes" / "ink" / "package.json"
    ink.parent.mkdir(parents=True)
    ink.write_text("{}", encoding="utf-8")
    (tmp_path / "node_modules" / ".package-lock.json").write_text(
        json.dumps({"packages": {"dep": {"version": "1"}}}), encoding="utf-8"
    )
    assert tui._tui_need_npm_install(tmp_path) is False


def test_tui_need_npm_install_prebuilt_bundle_skips(tui, tmp_path):
    entry = tmp_path / "dist" / "entry.js"
    entry.parent.mkdir(parents=True)
    entry.write_text("console.log('tui')", encoding="utf-8")
    assert tui._tui_need_npm_install(tmp_path) is False


# ── tui launch: build inputs / rebuild ──────────────────────────────────


def test_iter_tui_build_inputs(tui, tmp_path):
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.ts").write_text("x", encoding="utf-8")
    (src / "a.txt").write_text("x", encoding="utf-8")
    paths = {
        p.relative_to(tmp_path).as_posix()
        for p in tui._iter_tui_build_inputs(tmp_path)
    }
    assert "package.json" in paths
    assert "src/a.ts" in paths
    assert "src/a.txt" not in paths


def test_tui_need_rebuild_missing_entry(tui, tmp_path):
    assert tui._tui_need_rebuild(tmp_path) is True


def test_tui_need_rebuild_up_to_date_and_stale(tui, tmp_path):
    entry = tmp_path / "dist" / "entry.js"
    entry.parent.mkdir()
    entry.write_text("x", encoding="utf-8")
    pkg = tmp_path / "package.json"
    pkg.write_text("{}", encoding="utf-8")
    base = 1_700_000_000
    os.utime(entry, (base, base))
    os.utime(pkg, (base - 10, base - 10))
    assert tui._tui_need_rebuild(tmp_path) is False
    os.utime(pkg, (base + 10, base + 10))
    assert tui._tui_need_rebuild(tmp_path) is True


def test_tui_need_rebuild_force_env(monkeypatch, tui, tmp_path):
    monkeypatch.setenv("HERMES_TUI_FORCE_BUILD", "1")
    assert tui._tui_need_rebuild(tmp_path) is True


# ── no back-import ──────────────────────────────────────────────────────


def test_new_modules_import_without_main_backimport():
    program = (
        "import sys\n"
        "sys.modules['hermes_cli.main'] = None\n"
        "import hermes_cli.bundled_skills_sync\n"
        "import hermes_cli.tui_launch\n"
        "assert sys.modules['hermes_cli.main'] is None, 'hermes_cli.main was imported'\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
