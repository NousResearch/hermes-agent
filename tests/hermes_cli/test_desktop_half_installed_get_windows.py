"""A half-extracted ``node_modules/get-windows`` self-heals before the desktop npm install (#90829).

An in-place Windows update with the Desktop/gateway holding files open fails tar extraction
mid-package, leaving the dir without ``package.json``; npm never revisits an existing dir, so
the optional dep stayed broken on every later update until a manual repair.
"""

import shutil
from pathlib import Path

from hermes_cli import main_desktop


def _desktop_checkout(tmp_path: Path, monkeypatch, *, electron_dir: Path | None = None) -> Path:
    """A minimal checkout with a staged Electron package; returns the electron dir."""
    import hermes_cli.main as main_mod
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.setattr(main_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(main_desktop, "_nixos_build_env", lambda: {})
    (tmp_path / "package.json").write_text('{"workspaces": ["apps/*"]}', encoding="utf-8")
    (tmp_path / "package-lock.json").write_text("{}", encoding="utf-8")
    (tmp_path / "apps" / "desktop").mkdir(parents=True)
    (tmp_path / "apps" / "desktop" / "package.json").write_text("{}", encoding="utf-8")
    electron = electron_dir or tmp_path / "node_modules" / "electron"
    electron.mkdir(parents=True)
    (electron / "package.json").write_text("{}", encoding="utf-8")
    return electron


def test_half_installed_dir_is_removed_and_a_complete_one_is_kept(tmp_path):
    half = tmp_path / "node_modules" / "get-windows" / "lib" / "binding"
    half.mkdir(parents=True)
    complete = tmp_path / "apps" / "desktop" / "node_modules" / "get-windows"
    complete.mkdir(parents=True)
    (complete / "package.json").write_text('{"name": "get-windows"}', encoding="utf-8")

    removed = main_desktop._remove_half_installed_get_windows(tmp_path)

    assert removed == [tmp_path / "node_modules" / "get-windows"]
    assert not (tmp_path / "node_modules" / "get-windows").exists()
    assert (complete / "package.json").exists()


def test_install_removes_the_half_installed_dir_before_npm_runs(tmp_path, monkeypatch):
    half = tmp_path / "node_modules" / "get-windows"
    half.mkdir(parents=True)
    seen: list[bool] = []

    def _fake_npm(npm, cwd, **kwargs):
        seen.append(half.exists())
        return type("R", (), {"returncode": 0})()

    import hermes_cli.main as main_mod
    import hermes_cli.main_web_build as web_build
    monkeypatch.setattr(main_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(web_build, "_run_npm_install_deterministic", _fake_npm)
    monkeypatch.setattr(main_desktop, "_nixos_build_env", lambda: {})

    main_desktop._install_desktop_workspace_deps("npm", {})

    assert seen == [False], "npm must run only after the stale dir is gone"


def test_install_is_skipped_while_manifests_and_electron_are_unchanged(tmp_path, monkeypatch):
    """Root `npm ci` re-reifies the whole graph; only manifest changes or a pruned Electron warrant it. See #43837."""
    import hermes_cli.main_web_build as web_build
    from hermes_constants import get_default_hermes_root
    # Electron under the workspace-local hoist, where electronDist points (_electron_dir).
    electron = _desktop_checkout(tmp_path, monkeypatch, electron_dir=tmp_path / "apps" / "desktop" / "node_modules" / "electron")
    calls: list[Path] = []
    monkeypatch.setattr(web_build, "_run_npm_install_deterministic",
                        lambda npm, cwd, **kw: calls.append(cwd) or type("R", (), {"returncode": 0})())

    main_desktop._install_desktop_workspace_deps("npm", {})
    main_desktop._install_desktop_workspace_deps("npm", {})
    assert calls == [tmp_path], "second build with unchanged manifests must not npm ci again"

    shutil.rmtree(electron)  # pass 1 pruned Electron → the full install must run even with a matching stamp
    main_desktop._install_desktop_workspace_deps("npm", {})
    assert calls == [tmp_path, tmp_path]
    electron.mkdir(parents=True)
    (electron / "package.json").write_text("{}", encoding="utf-8")

    (tmp_path / "apps" / "desktop" / "package.json").write_text('{"name": "bumped"}', encoding="utf-8")
    main_desktop._install_desktop_workspace_deps("npm", {})
    assert calls == [tmp_path] * 3, "a changed manifest must reinstall"


def test_failed_reinstall_drops_the_stamp_so_the_next_update_repairs(tmp_path, monkeypatch):
    """The stamp is written on success only; an interrupted full `npm ci` after a code-only update
    must not leave an older matching stamp behind, or the half-installed tree is never revisited."""
    import pytest

    import hermes_cli.main_web_build as web_build
    from hermes_constants import get_default_hermes_root
    electron = _desktop_checkout(tmp_path, monkeypatch)
    codes = [0, 1, 0]
    monkeypatch.setattr(web_build, "_run_npm_install_deterministic",
                        lambda npm, cwd, **kw: type("R", (), {"returncode": codes.pop(0)})())

    main_desktop._install_desktop_workspace_deps("npm", {})  # success → stamp
    (electron / "package.json").unlink()  # pass 1 pruned Electron → full install needed
    with pytest.raises(SystemExit):
        main_desktop._install_desktop_workspace_deps("npm", {})  # interrupted mid-extract
    (electron / "package.json").write_text("{}", encoding="utf-8")  # partially reified tree

    main_desktop._install_desktop_workspace_deps("npm", {})
    assert codes == [], "the next update must run the install again, not skip on the stale stamp"
