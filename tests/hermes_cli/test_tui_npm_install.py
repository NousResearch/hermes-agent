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


# ── digest-stamp fast path (scoped --workspace install) ──────────────


def _write_lock(root: Path, packages: dict) -> Path:
    import json as _json

    lock = root / "package-lock.json"
    lock.write_text(_json.dumps({"packages": packages}))
    return lock


def test_no_reinstall_for_scoped_install_when_stamp_matches(
    tmp_path: Path, main_mod
) -> None:
    """Regression: a scoped ``npm install --workspace ui-tui`` produces a hidden
    lock that legitimately omits the other workspaces (apps/desktop, web), so the
    full root lock lists packages absent from the hidden lock — which used to
    force a reinstall on EVERY launch. A digest stamp from the last install must
    short-circuit that."""
    _touch_ink(tmp_path)
    lock = _write_lock(
        tmp_path,
        {
            "node_modules/foo": {"version": "1.0.0"},
            "apps/desktop": {"version": "1.0.0"},
            "apps/desktop/node_modules/electron": {"version": "30.0.0"},
        },
    )
    # Hidden lock from the scoped install: only the ui-tui-relevant package.
    (tmp_path / "node_modules" / ".package-lock.json").write_text(
        '{"packages":{"node_modules/foo":{"version":"1.0.0"}}}'
    )
    # Without a stamp this is a guaranteed (false-positive) reinstall...
    assert main_mod._tui_need_npm_install(tmp_path) is True
    # ...but once the last install's digest is recorded, it's a no-op.
    main_mod._tui_install_stamp(tmp_path).write_text(main_mod._lock_digest(lock))
    assert main_mod._tui_need_npm_install(tmp_path) is False


def test_stale_stamp_falls_through_to_comparison(tmp_path: Path, main_mod) -> None:
    """A stamp that doesn't match the current lockfile must NOT mask a genuinely
    needed reinstall — the check falls through to the hidden-lock comparison."""
    _touch_ink(tmp_path)
    _write_lock(
        tmp_path,
        {
            "node_modules/foo": {"version": "1.0.0"},
            "node_modules/bar": {"version": "1.0.0"},
        },
    )
    (tmp_path / "node_modules" / ".package-lock.json").write_text(
        '{"packages":{"node_modules/foo":{"version":"1.0.0"}}}'
    )
    main_mod._tui_install_stamp(tmp_path).write_text("stale-digest")
    # bar is required but missing from the hidden lock → still reinstall.
    assert main_mod._tui_need_npm_install(tmp_path) is True


def test_install_writes_digest_stamp(tmp_path: Path, main_mod, monkeypatch) -> None:
    """After a successful TUI install, _make_tui_argv records the lockfile digest
    so the next launch's fast path can skip the comparison."""
    tui_dir = tmp_path / "ui-tui"
    tui_dir.mkdir()
    (tui_dir / "package.json").write_text("{}")
    (tui_dir / "dist").mkdir()
    (tui_dir / "dist" / "entry.js").write_text("console.log('tui')")
    lock = _write_lock(tmp_path, {"node_modules/foo": {"version": "1.0.0"}})

    monkeypatch.delenv("TERMUX_VERSION", raising=False)
    monkeypatch.setenv("PREFIX", "/usr")
    monkeypatch.setattr(main_mod, "_tui_need_npm_install", lambda _root: True)
    monkeypatch.setattr(main_mod, "_tui_need_rebuild", lambda _root: False)
    monkeypatch.setattr(main_mod.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(
        main_mod.subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    main_mod._make_tui_argv(tui_dir, tui_dev=False)

    stamp = main_mod._tui_install_stamp(tmp_path)
    assert stamp.is_file()
    assert stamp.read_text().strip() == main_mod._lock_digest(lock)
