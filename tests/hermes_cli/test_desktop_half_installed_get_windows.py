"""A half-extracted ``node_modules/get-windows`` self-heals before the desktop npm install (#90829).

An in-place Windows update with the Desktop/gateway holding files open fails tar extraction
mid-package, leaving the dir without ``package.json``; npm never revisits an existing dir, so
the optional dep stayed broken on every later update until a manual repair.
"""

from hermes_cli import main_desktop


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
