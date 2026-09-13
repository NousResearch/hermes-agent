"""Data deletion respects declared runtime ownership, confirmation and scope."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_cli.uninstall as uninstall


@pytest.fixture
def layout(tmp_path, monkeypatch):
    user = tmp_path / "user"
    user.mkdir()
    home = user / "hermes-data"
    source = home / "workspace" / "custom-source"
    store = home / "machine" / "tool-store"
    userdata = user / "desktop-data"
    witnesses = [source / "hermes_cli" / "__init__.py", store / "python" / "python.exe",
                 home / "installs" / "other-install" / "facts.json", home / "bin" / "hermes.cmd",
                 home / "profiles" / "sibling" / "config.yaml",
                 home / "cache" / "partials" / ".locks" / "other-profile-transfer"]
    for path in witnesses:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("keep", encoding="utf-8")
    data = [home / "config.yaml", home / "sessions" / "session.json", home / "workspace" / "notes.txt"]
    for path in data:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}" if path.suffix in {".yaml", ".json"} else "user data", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: user)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(store))
    monkeypatch.setattr(uninstall, "get_project_root", lambda: source)
    monkeypatch.setattr(uninstall, "get_hermes_home", lambda: home)
    monkeypatch.setattr("hermes_cli.gui_uninstall.desktop_userdata_dir", lambda: userdata)
    return home, witnesses, data


@pytest.mark.parametrize("mode", ["confirmed", "cancel", "dry-run"])
def test_data_only_preserves_runtime_and_sibling_homes(layout, monkeypatch, mode):
    import json
    from hermes_cli.runtime_paths import install_state_dir, runtime_facts_path, selected_venv

    monkeypatch.delattr(Path, "is_junction", raising=False)
    home, witnesses, data = layout
    source = uninstall.get_project_root()
    generation = install_state_dir(source) / "environments" / "selected"
    generation.mkdir(parents=True)
    (generation / "pyvenv.cfg").write_text("include-system-site-packages = false", encoding="utf-8")
    runtime_facts_path(source).write_text(json.dumps({"packages": {"venv": {"environment": str(generation)}}}), encoding="utf-8")
    monkeypatch.setattr("builtins.input", lambda *args: "no")
    uninstall.run_data_uninstall(SimpleNamespace(yes=mode != "cancel", dry_run=mode == "dry-run"))
    assert selected_venv(source) == generation
    assert all(path.read_text(encoding="utf-8") == "keep" for path in witnesses)
    assert all(path.exists() is (mode != "confirmed") for path in data)
    assert home.is_dir()


def test_data_only_reports_a_failed_removal(layout, monkeypatch, capsys):
    home, witnesses, _ = layout
    real = uninstall.shutil.rmtree

    def refuse(path, *args, **kwargs):
        if Path(path) == home / "sessions":
            raise PermissionError("fixture holds sessions")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(uninstall.shutil, "rmtree", refuse)
    with pytest.raises(SystemExit) as failure:
        uninstall.run_data_uninstall(SimpleNamespace(yes=True, dry_run=False))
    assert failure.value.code != 0
    output = capsys.readouterr().out
    assert "Hermes data removed" not in output
    assert str(home / "sessions") in output
    assert all(path.exists() for path in witnesses)


def test_named_home_does_not_erase_siblings_or_desktop_data(layout, monkeypatch):
    home, witnesses, _ = layout
    active = home / "profiles" / "active"
    active.mkdir()
    config = active / "config.yaml"
    config.write_text("model: selected", encoding="utf-8")
    userdata = home.parent / "desktop-data"
    userdata.mkdir()
    (userdata / "preferences.json").write_text("keep", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(active))
    monkeypatch.setattr(uninstall, "get_hermes_home", lambda: active)
    uninstall.run_data_uninstall(SimpleNamespace(yes=True))
    assert not config.exists()
    assert all(path.exists() for path in witnesses)
    assert (userdata / "preferences.json").read_text(encoding="utf-8") == "keep"


def test_directory_replaced_with_a_link_does_not_expand_removal(layout, tmp_path):
    from hermes_cli.data_cleanup import plan_data_removal, remove_data

    home, _, _ = layout
    original = home / "workspace"
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    witness = foreign / "notes.txt"
    witness.write_text("not Hermes data", encoding="utf-8")
    probe = tmp_path / "symlink-probe"
    try:
        probe.symlink_to(foreign, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"native symlink creation unavailable: {exc}")
    probe.unlink()
    plan = plan_data_removal(home, uninstall.get_project_root())
    original.rename(home / "workspace-moved")
    original.symlink_to(foreign, target_is_directory=True)
    _, failures = remove_data(plan)
    assert failures
    assert witness.read_text(encoding="utf-8") == "not Hermes data"


def test_data_only_preserves_the_containing_bundled_application(layout, monkeypatch):
    import json
    from hermes_cli.bundled_app import PAYLOAD_DIR_NAME

    home, _, data = layout
    app = home / "installed-app"
    payload = app / "resources" / PAYLOAD_DIR_NAME
    project = payload / "repo"
    project.mkdir(parents=True)
    (project / "install-stamp.json").write_text(json.dumps({"payload": "bundled"}), encoding="utf-8")
    shell = app / "Hermes.exe"
    shell.write_bytes(b"retained app bytes")
    (payload / "venv").mkdir()
    manifest = payload / "manifest.json"
    manifest.write_text(json.dumps({"repo": "repo", "venv": "venv"}), encoding="utf-8")
    monkeypatch.setattr(uninstall, "get_project_root", lambda: project)
    uninstall.run_data_uninstall(SimpleNamespace(yes=True))
    assert shell.read_bytes() == b"retained app bytes"
    assert manifest.is_file()
    assert all(not path.exists() for path in data)
