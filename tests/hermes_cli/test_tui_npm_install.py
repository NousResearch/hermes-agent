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


# ── workspace-scoped comparison (#66978) ────────────────────────────
#
# In a shared workspace checkout the launch install is scoped to the ui-tui
# workspace, so only its dependency closure lands in the hidden lock while the
# root lock lists every other workspace's deps too. The comparison must ignore
# those unrelated packages instead of reinstalling on every launch.


def _write_ws(root: Path, ws_lock: str, hidden_lock: str) -> Path:
    """Lay out a workspace root + ui-tui member and return the ui-tui dir.

    ``@hermes/ink`` and the marker live at the workspace root (hoisted);
    ``ui-tui/`` has no lockfile of its own so ``_workspace_root`` treats the
    parent as the workspace root and the launch scopes to ``--workspace ui-tui``.
    """
    (root / "package-lock.json").write_text(ws_lock)
    _touch_ink(root)
    (root / "node_modules" / ".package-lock.json").write_text(hidden_lock)
    tui_dir = root / "ui-tui"
    tui_dir.mkdir(parents=True, exist_ok=True)
    # package.json (and no own lockfile) is what makes _workspace_root treat the
    # parent as the workspace root and the launch scope to --workspace ui-tui.
    (tui_dir / "package.json").write_text('{"name":"hermes-tui"}')
    return tui_dir


def test_no_install_when_only_other_workspace_deps_missing(tmp_path: Path, main_mod) -> None:
    """Deps that belong to apps/desktop / web (never installed by the ui-tui
    scoped install) must not trigger a reinstall on every launch (#66978)."""
    tui_dir = _write_ws(
        tmp_path,
        '{"packages":{'
        '"ui-tui":{"dependencies":{"foo":"1.0.0"}},'
        '"node_modules/foo":{"version":"1.0.0"},'
        '"apps/desktop":{"dependencies":{"desktop-only":"1.0.0"}},'
        '"node_modules/desktop-only":{"version":"1.0.0"},'
        '"apps/desktop/node_modules/nested":{"version":"1.0.0"}'
        "}}",
        '{"packages":{'
        '"ui-tui":{"dependencies":{"foo":"1.0.0"}},'
        '"node_modules/foo":{"version":"1.0.0"}'
        "}}",
    )
    assert main_mod._tui_need_npm_install(tui_dir) is False


def test_need_install_when_ui_tui_dep_missing_in_workspace_layout(tmp_path: Path, main_mod) -> None:
    """A genuinely missing ui-tui dependency is still caught after scoping."""
    tui_dir = _write_ws(
        tmp_path,
        '{"packages":{'
        '"ui-tui":{"dependencies":{"foo":"1.0.0","bar":"1.0.0"}},'
        '"node_modules/foo":{"version":"1.0.0"},'
        '"node_modules/bar":{"version":"1.0.0"}'
        "}}",
        '{"packages":{'
        '"ui-tui":{"dependencies":{"foo":"1.0.0","bar":"1.0.0"}},'
        '"node_modules/foo":{"version":"1.0.0"}'
        "}}",
    )
    assert main_mod._tui_need_npm_install(tui_dir) is True


def test_need_install_when_linked_workspace_dep_missing(tmp_path: Path, main_mod) -> None:
    """The closure follows workspace symlinks (@hermes/ink → ui-tui/packages/…)
    so a linked workspace's own missing dep triggers a reinstall."""
    tui_dir = _write_ws(
        tmp_path,
        '{"packages":{'
        '"ui-tui":{"dependencies":{"@hermes/ink":"*"}},'
        '"node_modules/@hermes/ink":{"link":true,"resolved":"ui-tui/packages/hermes-ink"},'
        '"ui-tui/packages/hermes-ink":{"dependencies":{"inkdep":"1.0.0"}},'
        '"node_modules/inkdep":{"version":"1.0.0"}'
        "}}",
        '{"packages":{'
        '"ui-tui":{"dependencies":{"@hermes/ink":"*"}},'
        '"node_modules/@hermes/ink":{"link":true,"resolved":"ui-tui/packages/hermes-ink"},'
        '"ui-tui/packages/hermes-ink":{"dependencies":{"inkdep":"1.0.0"}}'
        "}}",
    )
    assert main_mod._tui_need_npm_install(tui_dir) is True


def test_need_install_when_closure_package_version_drifts(tmp_path: Path, main_mod) -> None:
    """Version drift on an in-closure package still forces a reinstall."""
    tui_dir = _write_ws(
        tmp_path,
        '{"packages":{'
        '"ui-tui":{"dependencies":{"foo":"2.0.0"}},'
        '"node_modules/foo":{"version":"2.0.0"}'
        "}}",
        '{"packages":{'
        '"ui-tui":{"dependencies":{"foo":"2.0.0"}},'
        '"node_modules/foo":{"version":"1.0.0"}'
        "}}",
    )
    assert main_mod._tui_need_npm_install(tui_dir) is True


def test_workspace_closure_includes_dev_deps_of_scoped_workspace(main_mod) -> None:
    """ui-tui's devDependencies (esbuild/typescript build toolchain) are part of
    the closure; a transitive package's devDependencies are not."""
    packages = {
        "ui-tui": {
            "dependencies": {"foo": "1"},
            "devDependencies": {"esbuild": "1"},
        },
        "node_modules/foo": {"devDependencies": {"foo-dev-only": "1"}},
        "node_modules/esbuild": {},
        "node_modules/foo-dev-only": {},
    }
    closure = main_mod._npm_lock_workspace_closure(packages, "ui-tui")
    assert "node_modules/esbuild" in closure
    assert "node_modules/foo-dev-only" not in closure


def test_workspace_closure_returns_none_when_start_absent(main_mod) -> None:
    """Missing workspace key → None so the caller falls back to full compare."""
    assert main_mod._npm_lock_workspace_closure({"node_modules/foo": {}}, "ui-tui") is None


def test_workspace_closure_includes_dev_deps_of_selected_child_workspace(main_mod) -> None:
    """On Termux the install also scopes to ui-tui's child packages/* workspaces,
    so each selected child's devDependencies join the closure — a dev dep unique
    to a child is NOT dropped (regression for the child-scope false-negative)."""
    packages = {
        "ui-tui": {"dependencies": {"@hermes/ink": "*"}},
        "node_modules/@hermes/ink": {
            "link": True,
            "resolved": "ui-tui/packages/hermes-ink",
        },
        "ui-tui/packages/hermes-ink": {"devDependencies": {"child-dev-only": "1"}},
        "node_modules/child-dev-only": {},
    }
    # Only ui-tui selected (desktop): the child's dev dep is not installed.
    desktop = main_mod._npm_lock_workspace_closure(packages, {"ui-tui"})
    assert "node_modules/child-dev-only" not in desktop
    # ui-tui + child selected (Termux): the child's dev dep is in the closure.
    termux = main_mod._npm_lock_workspace_closure(
        packages, {"ui-tui", "ui-tui/packages/hermes-ink"}
    )
    assert "node_modules/child-dev-only" in termux


def test_termux_install_catches_missing_child_workspace_dev_dep(
    tmp_path: Path, main_mod, monkeypatch
) -> None:
    """On Termux the launch install selects ui-tui/packages/* too, installing
    each child's devDependencies.  A child dev dep missing from the hidden lock
    must trigger a reinstall — off Termux (child not selected) it must not."""
    ws_lock = (
        '{"packages":{'
        '"ui-tui":{"dependencies":{"@hermes/ink":"*"}},'
        '"node_modules/@hermes/ink":{"link":true,"resolved":"ui-tui/packages/hermes-ink"},'
        '"ui-tui/packages/hermes-ink":{"devDependencies":{"child-dev-only":"1.0.0"}},'
        '"node_modules/child-dev-only":{"version":"1.0.0"}'
        "}}"
    )
    hidden_lock = (
        '{"packages":{'
        '"ui-tui":{"dependencies":{"@hermes/ink":"*"}},'
        '"node_modules/@hermes/ink":{"link":true,"resolved":"ui-tui/packages/hermes-ink"},'
        '"ui-tui/packages/hermes-ink":{"devDependencies":{"child-dev-only":"1.0.0"}}'
        "}}"
    )
    tui_dir = _write_ws(tmp_path, ws_lock, hidden_lock)
    child = tui_dir / "packages" / "hermes-ink"
    child.mkdir(parents=True, exist_ok=True)
    (child / "package.json").write_text('{"name":"@hermes/ink"}')

    monkeypatch.setattr(main_mod, "_is_termux_startup_environment", lambda: False)
    assert main_mod._tui_need_npm_install(tui_dir) is False

    monkeypatch.setattr(main_mod, "_is_termux_startup_environment", lambda: True)
    assert main_mod._tui_need_npm_install(tui_dir) is True


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
    _assert_utf8_replace_capture(calls[0][1])
    _assert_utf8_replace_capture(calls[1][1])


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
        "--include=dev",
        "--silent",
        "--no-fund",
        "--no-audit",
        "--progress=false",
    ]
    assert calls[0][1]["cwd"] == str(tmp_path)
    _assert_utf8_replace_capture(calls[0][1])
    _assert_utf8_replace_capture(calls[1][1])


def test_make_tui_argv_npm_install_forces_include_dev(
    tmp_path: Path, main_mod, monkeypatch
) -> None:
    """The TUI-launch npm install must force --include=dev: ui-tui's build
    toolchain (esbuild, typescript) lives in devDependencies, and an inherited
    NODE_ENV=production (container shells; a parent TUI sets it on its own
    subprocess env) or an npm `omit=dev` config would silently skip them,
    breaking the TUI build with `tsc`/`esbuild: command not found."""
    tui_dir = tmp_path / "ui-tui"
    tui_dir.mkdir()
    (tui_dir / "package.json").write_text("{}")
    (tmp_path / "package-lock.json").write_text("{}")

    monkeypatch.delenv("TERMUX_VERSION", raising=False)
    monkeypatch.setenv("PREFIX", "/usr")
    monkeypatch.setenv("NODE_ENV", "production")
    monkeypatch.setattr(main_mod, "_tui_need_npm_install", lambda _root: True)
    monkeypatch.setattr(main_mod.shutil, "which", lambda name: f"/bin/{name}")
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(main_mod.subprocess, "run", fake_run)

    main_mod._make_tui_argv(tui_dir, tui_dev=False)

    install_cmd = calls[0][0][0]
    assert install_cmd[:2] == ["/bin/npm", "install"]
    assert "--include=dev" in install_cmd


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
    _assert_utf8_replace_capture(calls[0][1])


def test_make_tui_argv_decodes_dev_prebuild_with_utf8_replace(
    tmp_path: Path, main_mod, monkeypatch
) -> None:
    ink_dir = tmp_path / "packages" / "hermes-ink"
    ink_dir.mkdir(parents=True)
    tsx = tmp_path / "node_modules" / ".bin" / "tsx"
    tsx.parent.mkdir(parents=True)
    tsx.write_text("")

    monkeypatch.setattr(main_mod, "_tui_need_npm_install", lambda _root: False)
    monkeypatch.setattr(main_mod.shutil, "which", lambda name: f"/bin/{name}")
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(main_mod.subprocess, "run", fake_run)

    argv, cwd = main_mod._make_tui_argv(tmp_path, tui_dev=True)

    assert argv == [str(tsx), "src/entry.tsx"]
    assert cwd == tmp_path
    assert calls[0][0][0] == ["/bin/npm", "run", "build"]
    assert calls[0][1]["cwd"] == str(ink_dir)
    _assert_utf8_replace_capture(calls[0][1])


def test_make_tui_argv_exits_with_recovery_hint_when_workspace_unrecoverable(
    tmp_path: Path, main_mod, monkeypatch, capsys
) -> None:
    """Missing ui-tui + no git checkout → clean error, never touches node/npm."""
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


def test_make_tui_argv_omits_workspace_and_scrubs_esbuild_override(
    tmp_path: Path, main_mod, monkeypatch
) -> None:
    """When ui-tui/ has its own package-lock.json, _workspace_root returns
    tui_dir itself.  npm install --workspace ui-tui would fail in that case
    because npm cannot find a workspace named "ui-tui" inside ui-tui/.
    The fix omits --workspace and runs plain npm install from tui_dir.
    See #42973. The npm child must also ignore an inherited esbuild binary
    override: a version mismatch makes esbuild's postinstall abort (#87405).
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
    monkeypatch.setenv("ESBUILD_BINARY_PATH", "/opt/esbuild-0.28.2")
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
    assert Path(install_cmd[0]).name in {"npm", "npm.cmd"}
    assert install_cmd[1] == "install"
    # cwd must be tui_dir (standalone), not parent
    assert calls[0][1]["cwd"] == str(tui_dir)
    assert "ESBUILD_BINARY_PATH" not in calls[0][1]["env"]
    assert calls[1][0][0][1:] == ["run", "build"]
    assert "ESBUILD_BINARY_PATH" not in calls[1][1]["env"]
