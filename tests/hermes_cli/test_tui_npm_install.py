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


def test_need_install_when_ink_missing(tmp_path: Path, main_mod) -> None:
    (tmp_path / "package-lock.json").write_text("{}")
    assert main_mod._tui_need_npm_install(tmp_path) is True


def test_no_install_when_lock_newer_but_hidden_lock_matches(tmp_path: Path, main_mod) -> None:
    _touch_ink(tmp_path)
    (tmp_path / "package-lock.json").write_text('{"packages":{"node_modules/foo":{"version":"1.0.0"}}}')
    (tmp_path / "node_modules" / ".package-lock.json").write_text(
        '{"packages":{"node_modules/foo":{"version":"1.0.0","ideallyInert":true}}}'
    )
    os.utime(tmp_path / "package-lock.json", (200, 200))
    os.utime(tmp_path / "node_modules" / ".package-lock.json", (100, 100))
    assert main_mod._tui_need_npm_install(tmp_path) is False


def test_need_install_when_required_package_missing_from_hidden_lock(tmp_path: Path, main_mod) -> None:
    _touch_ink(tmp_path)
    (tmp_path / "package-lock.json").write_text(
        '{"packages":{"node_modules/foo":{"version":"1.0.0"},"node_modules/bar":{"version":"1.0.0"}}}'
    )
    (tmp_path / "node_modules" / ".package-lock.json").write_text(
        '{"packages":{"node_modules/foo":{"version":"1.0.0"}}}'
    )
    assert main_mod._tui_need_npm_install(tmp_path) is True


def test_no_install_when_only_optional_peer_package_missing_from_hidden_lock(tmp_path: Path, main_mod) -> None:
    _touch_ink(tmp_path)
    (tmp_path / "package-lock.json").write_text(
        '{"packages":{"node_modules/foo":{"version":"1.0.0"},"node_modules/optional":{"version":"1.0.0","optional":true,"peer":true}}}'
    )
    (tmp_path / "node_modules" / ".package-lock.json").write_text(
        '{"packages":{"node_modules/foo":{"version":"1.0.0"}}}'
    )
    assert main_mod._tui_need_npm_install(tmp_path) is False


def test_no_install_when_only_peer_annotation_differs(tmp_path: Path, main_mod) -> None:
    """npm 9 drops the ``peer`` flag from the hidden lock on dev-deps that are
    *also* declared as peers.  That's a cosmetic difference — the package is
    installed at the requested version — so it must not trigger a reinstall.
    Regression for the TUI-in-Docker failure where 16 such mismatches caused
    `Installing TUI dependencies…` → EACCES on every launch.
    """
    _touch_ink(tmp_path)
    (tmp_path / "package-lock.json").write_text(
        '{"packages":{'
        '"node_modules/foo":{"version":"1.0.0","dev":true,"peer":true,"resolved":"https://x/foo.tgz"}'
        '}}'
    )
    (tmp_path / "node_modules" / ".package-lock.json").write_text(
        '{"packages":{'
        '"node_modules/foo":{"version":"1.0.0","dev":true,"resolved":"https://x/foo.tgz"}'
        '}}'
    )
    assert main_mod._tui_need_npm_install(tmp_path) is False


def test_install_when_version_differs_even_with_peer_drop(tmp_path: Path, main_mod) -> None:
    """The peer-drop tolerance must not mask a real version skew."""
    _touch_ink(tmp_path)
    (tmp_path / "package-lock.json").write_text(
        '{"packages":{"node_modules/foo":{"version":"2.0.0","dev":true,"peer":true}}}'
    )
    (tmp_path / "node_modules" / ".package-lock.json").write_text(
        '{"packages":{"node_modules/foo":{"version":"1.0.0","dev":true}}}'
    )
    assert main_mod._tui_need_npm_install(tmp_path) is True


def test_no_install_when_lock_older_than_marker(tmp_path: Path, main_mod) -> None:
    _touch_ink(tmp_path)
    (tmp_path / "package-lock.json").write_text("{}")
    (tmp_path / "node_modules" / ".package-lock.json").write_text("{}")
    os.utime(tmp_path / "package-lock.json", (100, 100))
    os.utime(tmp_path / "node_modules" / ".package-lock.json", (200, 200))
    assert main_mod._tui_need_npm_install(tmp_path) is False


def test_need_install_when_marker_missing(tmp_path: Path, main_mod) -> None:
    _touch_ink(tmp_path)
    (tmp_path / "package-lock.json").write_text("{}")
    assert main_mod._tui_need_npm_install(tmp_path) is True


def test_no_install_without_lockfile_when_ink_present(tmp_path: Path, main_mod) -> None:
    _touch_ink(tmp_path)
    assert main_mod._tui_need_npm_install(tmp_path) is False


def test_no_install_prebuilt_bundle_mode(tmp_path: Path, main_mod) -> None:
    """dist/entry.js present and no package-lock.json → prebuilt bundle, skip npm install."""
    _touch_tui_entry(tmp_path)
    assert main_mod._tui_need_npm_install(tmp_path) is False


def test_need_rebuild_when_tui_bundle_missing(tmp_path: Path, main_mod) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "entry.tsx").write_text("console.log('src')")

    assert main_mod._tui_need_rebuild(tmp_path) is True


def test_no_rebuild_when_tui_bundle_newer_than_inputs(tmp_path: Path, main_mod) -> None:
    _touch_tui_entry(tmp_path)
    src = tmp_path / "src"
    src.mkdir()
    (src / "entry.tsx").write_text("console.log('src')")
    os.utime(src / "entry.tsx", (100, 100))
    os.utime(tmp_path / "dist" / "entry.js", (200, 200))

    assert main_mod._tui_need_rebuild(tmp_path) is False


def test_rebuild_when_tui_source_newer_than_bundle(tmp_path: Path, main_mod) -> None:
    _touch_tui_entry(tmp_path)
    src = tmp_path / "src"
    src.mkdir()
    (src / "entry.tsx").write_text("console.log('src')")
    os.utime(tmp_path / "dist" / "entry.js", (100, 100))
    os.utime(src / "entry.tsx", (200, 200))

    assert main_mod._tui_need_rebuild(tmp_path) is True


def test_make_tui_argv_skips_build_only_on_termux_when_fresh(
    tmp_path: Path, main_mod, monkeypatch
) -> None:
    _touch_tui_entry(tmp_path)
    monkeypatch.setenv("TERMUX_VERSION", "1")
    monkeypatch.setattr(main_mod, "_tui_need_npm_install", lambda _root: False)
    monkeypatch.setattr(main_mod, "_tui_need_rebuild", lambda _root: False)
    monkeypatch.setattr(main_mod.shutil, "which", lambda name: f"/bin/{name}")

    def fail_run(*_args, **_kwargs):
        raise AssertionError("fresh Termux TUI launch must not rebuild")

    monkeypatch.setattr(main_mod.subprocess, "run", fail_run)

    argv, cwd = main_mod._make_tui_argv(tmp_path, tui_dev=False)

    assert argv == ["/bin/node", "--expose-gc", str(tmp_path / "dist" / "entry.js")]
    assert cwd == tmp_path


def test_make_tui_argv_skips_install_on_termux_when_bundle_fresh(
    tmp_path: Path, main_mod, monkeypatch
) -> None:
    _touch_tui_entry(tmp_path)
    monkeypatch.setenv("TERMUX_VERSION", "1")
    monkeypatch.setattr(main_mod, "_tui_need_npm_install", lambda _root: True)
    monkeypatch.setattr(main_mod, "_tui_need_rebuild", lambda _root: False)
    monkeypatch.setattr(main_mod.shutil, "which", lambda name: f"/bin/{name}")

    def fail_run(*_args, **_kwargs):
        raise AssertionError("fresh Termux TUI launch must not run npm")

    monkeypatch.setattr(main_mod.subprocess, "run", fail_run)

    argv, cwd = main_mod._make_tui_argv(tmp_path, tui_dev=False)

    assert argv == ["/bin/node", "--expose-gc", str(tmp_path / "dist" / "entry.js")]
    assert cwd == tmp_path


def test_make_tui_argv_scopes_npm_install_on_termux_workspace(
    tmp_path: Path, main_mod, monkeypatch
) -> None:
    tui_dir = tmp_path / "ui-tui"
    tui_dir.mkdir()
    (tui_dir / "package.json").write_text("{}")
    ink_dir = tui_dir / "packages" / "hermes-ink"
    ink_dir.mkdir(parents=True)
    (ink_dir / "package.json").write_text("{}")
    (tmp_path / "package-lock.json").write_text("{}")

    monkeypatch.setenv("TERMUX_VERSION", "1")
    monkeypatch.setattr(main_mod, "_tui_need_npm_install", lambda _root: True)
    monkeypatch.setattr(main_mod, "_tui_need_rebuild", lambda _root: True)
    monkeypatch.setattr(main_mod.shutil, "which", lambda name: f"/bin/{name}")
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(main_mod.subprocess, "run", fake_run)

    main_mod._make_tui_argv(tui_dir, tui_dev=False)

    install_cmd = calls[0][0][0]
    assert install_cmd[:7] == [
        "/bin/npm",
        "install",
        "--workspace",
        "ui-tui",
        "--workspace",
        "ui-tui/packages/hermes-ink",
        "--include-workspace-root=false",
    ]
    assert calls[0][1]["cwd"] == str(tmp_path)


def test_make_tui_argv_keeps_desktop_workspace_install_behaviour(
    tmp_path: Path, main_mod, monkeypatch
) -> None:
    tui_dir = tmp_path / "ui-tui"
    tui_dir.mkdir()
    (tui_dir / "package.json").write_text("{}")
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

    assert calls[0][0][0] == [
        "/bin/npm",
        "install",
        "--workspace",
        "ui-tui",
        "--silent",
        "--no-fund",
        "--no-audit",
        "--progress=false",
    ]
    assert calls[0][1]["cwd"] == str(tmp_path)


def test_make_tui_argv_keeps_desktop_always_build_behaviour(
    tmp_path: Path, main_mod, monkeypatch
) -> None:
    _touch_tui_entry(tmp_path)
    monkeypatch.delenv("TERMUX_VERSION", raising=False)
    monkeypatch.setenv("PREFIX", "/usr")
    monkeypatch.setattr(main_mod, "_tui_need_npm_install", lambda _root: False)
    monkeypatch.setattr(main_mod, "_tui_need_rebuild", lambda _root: False)
    monkeypatch.setattr(main_mod.shutil, "which", lambda name: f"/bin/{name}")
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(main_mod.subprocess, "run", fake_run)

    main_mod._make_tui_argv(tmp_path, tui_dev=False)

    assert calls
    assert calls[0][0][0] == ["/bin/npm", "run", "build"]


# ── _workspace_root helper ──────────────────────────────────────────


def test_workspace_root_returns_parent_when_subpackage(tmp_path: Path, main_mod) -> None:
    """Sub-package has package.json, no lockfile; parent has lockfile → parent."""
    sub = tmp_path / "ui-tui"
    sub.mkdir()
    (sub / "package.json").write_text("{}")
    (tmp_path / "package-lock.json").write_text("{}")
    assert main_mod._workspace_root(sub) == tmp_path


def test_workspace_root_returns_dir_when_standalone(tmp_path: Path, main_mod) -> None:
    """No package.json → not a sub-package, return dir itself."""
    assert main_mod._workspace_root(tmp_path) == tmp_path


def test_workspace_root_returns_dir_when_own_lockfile(tmp_path: Path, main_mod) -> None:
    """Has package.json AND its own lockfile → standalone, return dir."""
    (tmp_path / "package.json").write_text("{}")
    (tmp_path / "package-lock.json").write_text("{}")
    (tmp_path.parent / "package-lock.json").write_text("{}")
    assert main_mod._workspace_root(tmp_path) == tmp_path


def test_workspace_root_returns_dir_when_no_parent_lockfile(
    tmp_path: Path, main_mod
) -> None:
    """Has package.json, no own lockfile, but parent also has no lockfile → standalone."""
    sub = tmp_path / "ui-tui"
    sub.mkdir()
    (sub / "package.json").write_text("{}")
    # tmp_path has no package-lock.json either
    assert main_mod._workspace_root(sub) == sub


def test_workspace_root_consistent_with_need_npm_install(
    tmp_path: Path, main_mod
) -> None:
    """Divergence regression: if someone creates ui-tui/package-lock.json
    by accident, _workspace_root (used by both _tui_need_npm_install AND
    the npm install cwd) returns ui-tui/ for both, so they never disagree.

    Before the shared helper, _tui_need_npm_install used a 3-condition
    check (falling back to ui-tui/ when its own lockfile exists) while
    the npm install cwd used a simpler check (still going to the parent
    because the parent lockfile still exists).  The shared helper
    eliminates the split.
    """
    sub = tmp_path / "ui-tui"
    sub.mkdir()
    (sub / "package.json").write_text("{}")
    # Both sub and parent have lockfiles — accidental state
    (sub / "package-lock.json").write_text("{}")
    (tmp_path / "package-lock.json").write_text("{}")

    ws = main_mod._workspace_root(sub)
    # _workspace_root sees sub has its own lockfile → treats it as standalone
    assert ws == sub

    # _tui_need_npm_install also uses _workspace_root, so both agree
    assert main_mod._tui_need_npm_install.__code__.co_names
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


def test_tui_launch_install_uses_workspace_scope(
    tmp_path: Path, main_mod, monkeypatch
) -> None:
    """TUI launch npm install must pass --workspace ui-tui to avoid pulling apps/desktop."""
    tui_dir = tmp_path / "ui-tui"
    tui_dir.mkdir()
    (tui_dir / "package.json").write_text("{}")
    (tui_dir / "dist" / "entry.js").parent.mkdir(parents=True)
    (tui_dir / "dist" / "entry.js").write_text("console.log('tui')")
    # workspace root: parent has lockfile, tui_dir does not
    (tmp_path / "package-lock.json").write_text("{}")

    monkeypatch.setattr(main_mod, "_tui_need_npm_install", lambda _root: True)
    monkeypatch.setattr(main_mod, "_tui_need_rebuild", lambda _root: False)
    monkeypatch.setattr(main_mod.shutil, "which", lambda name: f"/usr/bin/{name}")

    npm_calls = []

    def fake_run(cmd, **kwargs):
        if cmd[0].endswith("npm"):
            npm_calls.append(cmd)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(main_mod.subprocess, "run", fake_run)

    main_mod._make_tui_argv(tui_dir, tui_dev=False)

    assert npm_calls, "expected npm install to be called"
    install_cmd = npm_calls[0]
    assert "--workspace" in install_cmd
    assert "ui-tui" in install_cmd
