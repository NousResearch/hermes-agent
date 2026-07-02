

def test_tui_finds_bundled_entry_js(tmp_path):
    """_find_bundled_tui finds entry.js bundled in the package."""
    tui_dist = tmp_path / "hermes_cli" / "tui_dist"
    tui_dist.mkdir(parents=True)
    entry = tui_dist / "entry.js"
    entry.write_text("// bundled TUI", encoding="utf-8")

    from hermes_cli.main import _find_bundled_tui
    result = _find_bundled_tui(hermes_cli_dir=tmp_path / "hermes_cli")
    assert result is not None
    assert result.name == "entry.js"


def test_tui_returns_none_when_no_bundle(tmp_path):
    """_find_bundled_tui returns None when no bundle exists."""
    from hermes_cli.main import _find_bundled_tui
    result = _find_bundled_tui(hermes_cli_dir=tmp_path / "hermes_cli")
    assert result is None


def test_tui_uses_bundled_dist_without_source_workspace(tmp_path, monkeypatch):
    """Packaged installs may ship hermes_cli/tui_dist but not ui-tui/."""
    from hermes_cli import main

    bundled = tmp_path / "hermes_cli" / "tui_dist" / "entry.js"
    bundled.parent.mkdir(parents=True)
    bundled.write_text("// bundled TUI", encoding="utf-8")
    project_root = tmp_path / "site-packages"
    project_root.mkdir()

    def fail_if_workspace_required(_tui_dir):
        raise AssertionError("source ui-tui workspace should not be required")

    monkeypatch.delenv("HERMES_TUI_DIR", raising=False)
    monkeypatch.setattr(main, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(main, "_find_bundled_tui", lambda: bundled)
    monkeypatch.setattr(main, "_ensure_tui_workspace", fail_if_workspace_required)
    monkeypatch.setattr(main, "_ensure_tui_node", lambda: None)
    monkeypatch.setattr(main.shutil, "which", lambda name: "/usr/bin/node" if name == "node" else None)

    argv, cwd = main._make_tui_argv(project_root / "ui-tui", tui_dev=False)

    assert argv == ["/usr/bin/node", "--expose-gc", str(bundled)]
    assert cwd == bundled.parent
