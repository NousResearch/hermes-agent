"""Regression for #128974: GUI dry-run and GUI removal preserve shared state."""

import sys
from pathlib import Path

import pytest

from hermes_cli import gui_uninstall, main
from hermes_constants import get_hermes_home


@pytest.fixture
def installation(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = get_hermes_home()
    assert home.resolve().is_relative_to(tmp_path.resolve())
    root = home / "hermes-agent"
    protected = {
        root / "hermes_cli" / "__init__.py": b"# agent source\n",
        root / "venv" / "runtime": b"agent runtime",
        root / "node_modules" / "shared-package" / "index.js": b"shared workspace dependency",
        home / "config.yaml": b"model: fixture-model\n",
        home / ".env": b"FIXTURE_TOKEN=fake\n",
        home / "sessions" / "saved.json": b'{"saved": true}',
    }
    record = tmp_path / "desktop-installed-apps.json"
    bundle = tmp_path / "apps" / "Hermes.app"
    userdata = tmp_path / "desktop-data"
    gui_files = {
        root / "apps" / "desktop" / "dist" / "index.html": b"renderer",
        root / "apps" / "desktop" / "release" / "artifact": b"release",
        root / "apps" / "desktop" / "node_modules" / "private-module": b"desktop dependency",
        home / "desktop-build-stamp.json": b'{"fixture": true}',
        bundle / "app": b"desktop app",
        userdata / "connections.json": b'{"fixture": true}',
        record: b'{"fixture": true}',
    }
    for path, data in {**protected, **gui_files}.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    monkeypatch.setattr(gui_uninstall, "packaged_gui_app_paths", lambda: [bundle])
    monkeypatch.setattr(gui_uninstall, "desktop_userdata_dir", lambda: userdata)
    monkeypatch.setattr(gui_uninstall, "desktop_install_record", lambda: record)
    return protected, gui_files, record


def _run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["hermes", "uninstall", "--gui", *args])
    return main.main()


def test_gui_dry_run_never_confirms_or_removes_installed_state(installation, monkeypatch, capsys):
    protected, gui_files, record = installation

    def unexpected_input(*args):
        pytest.fail("A dry-run must not request removal confirmation")

    monkeypatch.setattr("builtins.input", unexpected_input)
    for flags in (("--yes",), ()):
        _run(monkeypatch, "--dry-run", *flags)
        for path, data in {**protected, **gui_files}.items():
            assert path.read_bytes() == data
        output = capsys.readouterr().out
        assert "Dry run" in output
        assert str(record) in output
        assert "Uninstalled!" not in output

    # The sibling full-agent preview must also work without an interactive terminal.
    monkeypatch.setattr(sys, "argv", ["hermes", "uninstall", "--dry-run"])
    main.main()
    for path, data in {**protected, **gui_files}.items():
        assert path.read_bytes() == data


def test_gui_removal_keeps_agent_data_and_shared_workspace_dependencies(installation, monkeypatch):
    protected, gui_files, _ = installation
    _run(monkeypatch, "--yes")
    for path, data in protected.items():
        assert path.read_bytes() == data
    assert all(not path.exists() for path in gui_files)
