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

    import hermes_constants

    def fake_find(name: str) -> str | None:
        if name == "node":
            return "/usr/bin/node"
        raise AssertionError(
            f"unexpected find_node_executable({name!r}) — "
            "bundled path must not need npm/git"
        )

    monkeypatch.setattr(hermes_constants, "find_node_executable", fake_find)

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
    monkeypatch.setattr(main_mod, "_ensure_tui_node", lambda: None)
    monkeypatch.setattr(main_mod, "_find_bundled_tui", lambda: None)

    import hermes_constants

    monkeypatch.setattr(
        hermes_constants, "find_node_executable", lambda name: f"/bin/{name}"
    )
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
    # --silent used to suppress EBADENGINE and skip managed-Node repair (#78826)
    assert "--silent" not in install_cmd


def test_make_tui_argv_preserves_install_error_preview(
    tmp_path: Path, main_mod, monkeypatch, capsys
) -> None:
    """Failed TUI install must surface the last lines of npm stderr.

    Quietness comes from capture_output + CI=1, not ``--silent`` — dropping
    ``--silent`` is what makes this preview (and engine detection) possible.
    """
    tui_dir = tmp_path / "ui-tui"
    tui_dir.mkdir()
    (tui_dir / "package.json").write_text("{}")
    (tui_dir / "package-lock.json").write_text("{}")
    _touch_tui_entry(tui_dir)
    _touch_ink(tui_dir)

    monkeypatch.delenv("TERMUX_VERSION", raising=False)
    monkeypatch.setenv("PREFIX", "/usr")
    monkeypatch.setenv("HERMES_QUIET", "1")
    monkeypatch.setattr(main_mod, "_tui_need_npm_install", lambda _root: True)
    monkeypatch.setattr(main_mod, "_ensure_tui_node", lambda: None)
    monkeypatch.setattr(main_mod, "_find_bundled_tui", lambda: None)

    import hermes_constants

    monkeypatch.setattr(
        hermes_constants, "find_node_executable", lambda name: f"/bin/{name}"
    )

    def fake_run(cmd, **kwargs):
        _assert_utf8_replace_capture(kwargs)
        if list(cmd[:2]) == ["/bin/npm", "install"]:
            return types.SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="npm error code EUSAGE\nlockfile is out of sync\n",
            )
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(main_mod.subprocess, "run", fake_run)

    with pytest.raises(SystemExit) as exited:
        main_mod._make_tui_argv(tui_dir, tui_dev=False)

    assert exited.value.code == 1
    out = capsys.readouterr().out
    assert "npm install failed." in out
    assert "lockfile is out of sync" in out


def test_make_tui_argv_repairs_opaque_npm_engine_failure(
    tmp_path: Path, main_mod, monkeypatch
) -> None:
    """TUI install failure with empty output still triggers engine repair.

    Regression for #78826: even if stderr is empty (historical --silent),
    maybe_repair_npm_engine must be consulted and a successful repair must
    retry the install with the returned npm.
    """
    tui_dir = tmp_path / "ui-tui"
    tui_dir.mkdir()
    (tui_dir / "package.json").write_text("{}")
    (tui_dir / "package-lock.json").write_text("{}")
    _touch_tui_entry(tui_dir)
    _touch_ink(tui_dir)

    monkeypatch.delenv("TERMUX_VERSION", raising=False)
    monkeypatch.setenv("PREFIX", "/usr")
    monkeypatch.setenv("HERMES_QUIET", "1")
    monkeypatch.setattr(main_mod, "_tui_need_npm_install", lambda _root: True)
    monkeypatch.setattr(main_mod, "_ensure_tui_node", lambda: None)
    monkeypatch.setattr(main_mod, "_find_bundled_tui", lambda: None)

    import hermes_constants

    monkeypatch.setattr(
        hermes_constants, "find_node_executable", lambda name: f"/bin/{name}"
    )

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if list(cmd[:2]) == ["/bin/npm", "install"]:
            return types.SimpleNamespace(returncode=1, stdout="", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(main_mod.subprocess, "run", fake_run)

    import hermes_cli.npm_engine as npm_engine

    repair_calls = []

    def fake_repair(npm, output, *, quiet=False):
        repair_calls.append((npm, output))
        return "/managed/npm"

    monkeypatch.setattr(npm_engine, "maybe_repair_npm_engine", fake_repair)

    argv, cwd = main_mod._make_tui_argv(tui_dir, tui_dev=False)

    assert repair_calls, "opaque install failure must consult maybe_repair_npm_engine"
    assert repair_calls[0][0] == "/bin/npm"
    assert any(c[:2] == ["/managed/npm", "install"] for c in calls), (
        f"repair must retry install with managed npm, got: {calls}"
    )
    assert argv == ["/bin/node", "--expose-gc", str(tui_dir / "dist" / "entry.js")]
    assert cwd == tui_dir
