"""_tui_need_npm_install: auto npm when node_modules is behind the lockfile."""

import os
import types
from pathlib import Path

import pytest


@pytest.fixture
def main_mod():
    import hermes_cli.main as m

    return m


def _touch_ink(root: Path) -> None:
    ink = root / "node_modules" / "@hermes" / "ink" / "package.json"
    ink.parent.mkdir(parents=True, exist_ok=True)
    ink.write_text("{}")


def _touch_tui_entry(root: Path) -> None:
    entry = root / "dist" / "entry.js"
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text("console.log('tui')")


def _assert_utf8_replace_capture(kwargs: dict) -> None:
    assert kwargs["text"] is True
    assert kwargs["encoding"] == "utf-8"
    assert kwargs["errors"] == "replace"














def test_make_tui_argv_uses_bundled_tui_when_workspace_missing(
    tmp_path: Path, main_mod, monkeypatch
) -> None:
    """Prebuilt-install regression (#56665): a prebuilt install (Docker
    image, Nix build, or prior `npm run build`) ships
    hermes_cli/tui_dist/entry.js but never ships ui-tui/ (that directory only
    exists in a git checkout). _make_tui_argv must try the bundled entry.js
    BEFORE _ensure_tui_workspace() — requiring the workspace first hard-exits
    every prebuilt dashboard Chat tab connection with `sys.exit(1)` (surfaced
    to the user as the unhelpful "Chat unavailable: 1") despite a perfectly
    runnable bundled TUI on disk. The bundled shortcut must succeed without
    ever touching the (missing) ui-tui workspace or git.
    """
    monkeypatch.delenv("HERMES_TUI_DIR", raising=False)
    monkeypatch.setattr(main_mod, "_ensure_tui_node", lambda: None)

    bundled_entry = tmp_path / "bundled" / "entry.js"
    bundled_entry.parent.mkdir(parents=True)
    bundled_entry.write_text("// bundled TUI")
    monkeypatch.setattr(main_mod, "_find_bundled_tui", lambda: bundled_entry)

    def which(name: str) -> str | None:
        if name == "node":
            return "/usr/bin/node"
        raise AssertionError(f"unexpected shutil.which({name!r}) call — bundled path must not need npm/git")

    monkeypatch.setattr(main_mod.shutil, "which", which)

    def fail_run(*_args, **_kwargs):
        raise AssertionError("bundled TUI path must not spawn any subprocess (no npm install/build, no git restore)")

    monkeypatch.setattr(main_mod.subprocess, "run", fail_run)

    # ui-tui/ deliberately does not exist under tmp_path, and there is no
    # .git either — this mirrors a prebuilt (Docker/Nix) install exactly.
    tui_dir = tmp_path / "ui-tui"
    assert not tui_dir.exists()

    argv, cwd = main_mod._make_tui_argv(tui_dir, tui_dev=False)

    assert argv == ["/usr/bin/node", "--expose-gc", str(bundled_entry)]
    assert cwd == bundled_entry.parent


# ── _workspace_root helper ──────────────────────────────────────────




    # (Smoke test: just confirm _tui_need_npm_install doesn't crash)
    # It won't need install because the lockfile exists and there's no
    # hidden lockfile to compare against, and ink is missing → True.
    # But the key invariant is: ws_root for the need-check == ws_root
    # for the install cwd — both use _workspace_root(sub).


def test_no_stray_lockfiles_in_workspace_subdirs(main_mod) -> None:
    """Workspace sub-directories must not contain their own package-lock.json.

    With a single workspace root lockfile, per-directory lockfiles are
    always accidental (typically from running ``npm install`` inside the
    wrong directory).  They cause ``_workspace_root`` to treat the
    sub-package as standalone, which breaks hoisted ``node_modules``
    resolution and can silently diverge the install cwd from the
    lockfile-check root.

    This is an invariant, not a change-detector: the workspace structure
    is not expected to gain per-dir lockfiles.
    """
    root = main_mod.PROJECT_ROOT
    # Workspace members that live one level below the root and should
    # NOT have their own lockfile.  (ui-tui/packages/* members are
    # two levels deep and even less likely to get accidental lockfiles,
    # but we check them too for completeness.)
    subdirs = [
        root / "ui-tui",
        root / "web",
        root / "apps" / "desktop",
        root / "apps" / "shared",
    ]
    # Also sweep ui-tui/packages/* (hermes-ink etc.)
    tui_pkgs = root / "ui-tui" / "packages"
    if tui_pkgs.is_dir():
        subdirs.extend(d for d in tui_pkgs.iterdir() if d.is_dir())

    stray = [d for d in subdirs if (d / "package-lock.json").is_file()]
    assert not stray, (
        "stray package-lock.json found in workspace sub-directory(es); "
        "delete them and run `npm install` from the repo root instead: "
        + ", ".join(str(d / "package-lock.json") for d in stray)
    )


def test_make_tui_argv_omits_workspace_when_tui_has_own_lockfile(
    tmp_path: Path, main_mod, monkeypatch
) -> None:
    """When ui-tui/ has its own package-lock.json, _workspace_root returns
    tui_dir itself.  npm install --workspace ui-tui would fail in that case
    because npm cannot find a workspace named "ui-tui" inside ui-tui/.
    The fix omits --workspace and runs plain npm install from tui_dir.
    See #42973.
    """
    tui_dir = tmp_path / "ui-tui"
    tui_dir.mkdir()
    (tui_dir / "package.json").write_text("{}")
    # Simulate curl-install layout: tui_dir has its own lockfile
    (tui_dir / "package-lock.json").write_text("{}")
    # Parent also has lockfile (but _workspace_root prefers tui_dir's own)
    (tmp_path / "package-lock.json").write_text("{}")

    monkeypatch.delenv("TERMUX_VERSION", raising=False)
    monkeypatch.setenv("PREFIX", "/usr")
    monkeypatch.setattr(main_mod, "_tui_need_npm_install", lambda _root: True)
    monkeypatch.setattr(main_mod.shutil, "which", lambda name: f"/bin/{name}")
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(main_mod.subprocess, "run", fake_run)

    main_mod._make_tui_argv(tui_dir, tui_dev=False)

    install_cmd = calls[0][0][0]
    # Must NOT contain --workspace when npm_cwd == tui_dir
    assert "--workspace" not in install_cmd, (
        f"npm install should omit --workspace when tui_dir has its own lockfile, got: {install_cmd}"
    )
    assert install_cmd[:2] == ["/bin/npm", "install"]
    # cwd must be tui_dir (standalone), not parent
    assert calls[0][1]["cwd"] == str(tui_dir)


# ── workspace-mode lockfile comparison (#45657) ─────────────────────


import json


def _write_lock(path: Path, packages: dict) -> None:
    path.write_text(json.dumps({"packages": packages}), encoding="utf-8")


def _make_workspace_checkout(tmp_path: Path) -> Path:
    """Minimal npm-workspaces checkout: ui-tui/ has a package.json but no
    lockfile, so _workspace_root(ui-tui) resolves to the repo root where the
    single lockfile and hoisted node_modules/ live."""
    tui_dir = tmp_path / "ui-tui"
    tui_dir.mkdir()
    (tui_dir / "package.json").write_text("{}")
    _touch_ink(tmp_path)
    return tui_dir


_PKG = {
    "version": "1.2.3",
    "resolved": "https://registry.npmjs.org/dep/-/dep-1.2.3.tgz",
    "integrity": "sha512-abc",
}


def test_workspace_mode_ignores_packages_from_other_workspaces(
    tmp_path: Path, main_mod
) -> None:
    """#45657 primary trigger: launch installs only the ui-tui subset
    (``npm install --workspace ui-tui``), so the hidden lockfile never
    contains sibling-workspace entries (apps/desktop, web, …).  Treating
    them as missing forced a spurious "Installing TUI dependencies…" +
    no-op npm install on EVERY launch.  In workspace mode, entries absent
    from the hidden lockfile must be skipped."""
    tui_dir = _make_workspace_checkout(tmp_path)

    full_packages = {
        "": {"name": "monorepo"},
        "ui-tui": {"version": "0.1.0", "license": "MIT"},
        # Sibling workspaces and their deps: installed only by a full-root
        # npm install, intentionally absent from the hidden lockfile.
        "apps/desktop": {"version": "0.1.0", "license": "MIT"},
        "apps/bootstrap-installer": {"version": "0.1.0"},
        "node_modules/electron": dict(_PKG),
        # ui-tui deps: installed, present in both locks.
        "node_modules/@hermes/ink": {"resolved": "ui-tui/packages/hermes-ink", "link": True},
        "node_modules/ink": dict(_PKG),
    }
    installed_packages = {
        "": {"name": "monorepo"},
        "ui-tui": {"version": "0.1.0", "license": "MIT"},
        "node_modules/@hermes/ink": {"resolved": "ui-tui/packages/hermes-ink", "link": True},
        "node_modules/ink": dict(_PKG),
    }
    _write_lock(tmp_path / "package-lock.json", full_packages)
    _write_lock(tmp_path / "node_modules" / ".package-lock.json", installed_packages)

    assert main_mod._tui_need_npm_install(tui_dir) is False


def test_workspace_mode_detects_missing_direct_tui_dependency(
    tmp_path: Path, main_mod
) -> None:
    """A missing package reachable from ui-tui must still trigger repair."""
    tui_dir = _make_workspace_checkout(tmp_path)
    _write_lock(
        tmp_path / "package-lock.json",
        {
            "": {"name": "monorepo"},
            "ui-tui": {"dependencies": {"ink": "1.2.3"}},
            "node_modules/ink": dict(_PKG),
        },
    )
    _write_lock(
        tmp_path / "node_modules" / ".package-lock.json",
        {"": {"name": "monorepo"}, "ui-tui": {"dependencies": {"ink": "1.2.3"}}},
    )

    assert main_mod._tui_need_npm_install(tui_dir) is True


def test_workspace_mode_detects_missing_nested_tui_dependency(
    tmp_path: Path, main_mod
) -> None:
    """A missing transitive package in the ui-tui closure must trigger repair."""
    tui_dir = _make_workspace_checkout(tmp_path)
    _write_lock(
        tmp_path / "package-lock.json",
        {
            "": {"name": "monorepo"},
            "ui-tui": {"dependencies": {"ink": "1.2.3"}},
            "node_modules/ink": {"dependencies": {"nested": "1.0.0"}, **_PKG},
            "node_modules/ink/node_modules/nested": dict(_PKG),
        },
    )
    _write_lock(
        tmp_path / "node_modules" / ".package-lock.json",
        {
            "": {"name": "monorepo"},
            "ui-tui": {"dependencies": {"ink": "1.2.3"}},
            "node_modules/ink": {"dependencies": {"nested": "1.0.0"}, **_PKG},
        },
    )

    assert main_mod._tui_need_npm_install(tui_dir) is True


def test_workspace_mode_ignores_dev_flag_skew(tmp_path: Path, main_mod) -> None:
    """#45657 secondary trigger: npm recomputes ``dev`` against the installed
    subset — a package that is a prod dep of an *uninstalled* workspace has
    no ``dev`` flag in the root lock but gets ``dev: true`` in the hidden
    actualized lock.  That annotation is not real skew."""
    tui_dir = _make_workspace_checkout(tmp_path)

    _write_lock(
        tmp_path / "package-lock.json",
        {"": {"name": "monorepo"}, "node_modules/zod": dict(_PKG)},
    )
    hidden_pkg = {**_PKG, "dev": True}
    _write_lock(
        tmp_path / "node_modules" / ".package-lock.json",
        {"": {"name": "monorepo"}, "node_modules/zod": hidden_pkg},
    )

    assert main_mod._tui_need_npm_install(tui_dir) is False


def test_workspace_mode_still_detects_real_skew(tmp_path: Path, main_mod) -> None:
    """The skip must not blind the check: a ui-tui dependency whose version
    actually moved (e.g. lockfile updated, node_modules stale) is present
    in BOTH locks with differing fields and must still trigger reinstall."""
    tui_dir = _make_workspace_checkout(tmp_path)

    _write_lock(
        tmp_path / "package-lock.json",
        {"": {"name": "monorepo"}, "node_modules/ink": dict(_PKG)},
    )
    stale = dict(_PKG)
    stale["version"] = "1.2.2"
    stale["integrity"] = "sha512-old"
    _write_lock(
        tmp_path / "node_modules" / ".package-lock.json",
        {"": {"name": "monorepo"}, "node_modules/ink": stale},
    )

    assert main_mod._tui_need_npm_install(tui_dir) is True


def test_standalone_mode_still_reinstalls_for_missing_packages(
    tmp_path: Path, main_mod
) -> None:
    """Non-workspace layouts (curl install: ui-tui has its own lockfile)
    keep the strict behaviour — a package missing from the hidden lockfile
    means an incomplete install and must trigger npm install."""
    tui_dir = tmp_path / "ui-tui"
    tui_dir.mkdir()
    (tui_dir / "package.json").write_text("{}")
    # Own lockfile → _workspace_root(tui_dir) == tui_dir (standalone mode).
    _write_lock(
        tui_dir / "package-lock.json",
        {"": {"name": "ui-tui"}, "node_modules/ink": dict(_PKG)},
    )
    _touch_ink(tui_dir)
    _write_lock(
        tui_dir / "node_modules" / ".package-lock.json",
        {"": {"name": "ui-tui"}},  # ink missing from the actualized tree
    )

    assert main_mod._tui_need_npm_install(tui_dir) is True
