"""Tests for hermes_cli.gui_uninstall — GUI-only uninstall + install discovery.

Covers the cross-platform artifact discovery, the agent/GUI detection the
desktop UI gates options on, and that ``uninstall_gui`` removes only GUI
artifacts (built renderer/release/node_modules, packaged bundle, Electron
userData) while leaving the Python agent + config/sessions/.env intact.
"""

import sys
from pathlib import Path

import pytest

import hermes_cli.gui_uninstall as gu


def _make_agent(hermes_home: Path) -> Path:
    """Create a fake agent install: source package + venv."""
    agent_root = hermes_home / "hermes-agent"
    (agent_root / "hermes_cli").mkdir(parents=True)
    (agent_root / "hermes_cli" / "__init__.py").write_text("")
    (agent_root / "venv" / "bin").mkdir(parents=True)
    return agent_root


def _make_gui_build(hermes_home: Path) -> None:
    """Create the source-built GUI artifacts a `hermes desktop` run produces."""
    desktop = hermes_home / "hermes-agent" / "apps" / "desktop"
    (desktop / "dist").mkdir(parents=True)
    (desktop / "dist" / "index.html").write_text("<html>")
    (desktop / "release" / "linux-unpacked").mkdir(parents=True)
    (desktop / "node_modules").mkdir(parents=True)
    (hermes_home / "hermes-agent" / "node_modules").mkdir(parents=True)
    (hermes_home / "desktop-build-stamp.json").write_text("{}")


def _make_user_data(hermes_home: Path) -> None:
    (hermes_home / "config.yaml").write_text("x: 1\n")
    (hermes_home / ".env").write_text("KEY=secret\n")
    (hermes_home / "sessions").mkdir()


def test_agent_is_installed_detects_source_and_venv(tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    assert gu.agent_is_installed(hermes_home) is False
    _make_agent(hermes_home)
    assert gu.agent_is_installed(hermes_home) is True


def test_agent_is_installed_venv_only(tmp_path):
    """A checkout with only a venv (no package dir yet) still counts."""
    hermes_home = tmp_path / ".hermes"
    (hermes_home / "hermes-agent" / "venv").mkdir(parents=True)
    assert gu.agent_is_installed(hermes_home) is True


def test_source_built_artifacts_lists_known_paths(tmp_path):
    hermes_home = tmp_path / ".hermes"
    _make_gui_build(hermes_home)
    artifacts = gu.source_built_gui_artifacts(hermes_home)
    names = {p.name for p in artifacts}
    assert "dist" in names
    assert "release" in names
    assert "node_modules" in names
    assert "desktop-build-stamp.json" in names


def test_gui_is_installed_true_when_built(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    _make_gui_build(hermes_home)
    # Make sure packaged-app + userdata probes don't false-positive on the box
    # running the test.
    monkeypatch.setattr(gu, "packaged_gui_app_paths", lambda: [])
    monkeypatch.setattr(gu, "desktop_userdata_dir", lambda: tmp_path / "nope")
    assert gu.gui_is_installed(hermes_home) is True


def test_gui_is_installed_false_when_nothing(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setattr(gu, "packaged_gui_app_paths", lambda: [])
    monkeypatch.setattr(gu, "desktop_userdata_dir", lambda: tmp_path / "nope")
    assert gu.gui_is_installed(hermes_home) is False


def test_uninstall_gui_removes_only_gui_artifacts(tmp_path, monkeypatch):
    """The core invariant: GUI gone, agent + user data untouched."""
    hermes_home = tmp_path / ".hermes"
    agent_root = _make_agent(hermes_home)
    _make_gui_build(hermes_home)
    _make_user_data(hermes_home)

    # Isolate the packaged-app + userdata probes from the test machine.
    monkeypatch.setattr(gu, "packaged_gui_app_paths", lambda: [])
    monkeypatch.setattr(gu, "desktop_userdata_dir", lambda: tmp_path / "userdata-none")

    removed = gu.uninstall_gui(hermes_home)
    removed_names = {p.name for p in removed}

    # GUI artifacts removed.
    desktop = agent_root / "apps" / "desktop"
    assert not (desktop / "dist").exists()
    assert not (desktop / "release").exists()
    assert not (desktop / "node_modules").exists()
    assert not (agent_root / "node_modules").exists()
    assert not (hermes_home / "desktop-build-stamp.json").exists()
    assert "dist" in removed_names

    # Agent + user data preserved.
    assert (agent_root / "hermes_cli" / "__init__.py").exists()
    assert (agent_root / "venv").exists()
    assert (hermes_home / "config.yaml").exists()
    assert (hermes_home / ".env").exists()
    assert (hermes_home / "sessions").exists()
    # The desktop source dir itself survives (only its build output is gone).
    assert desktop.exists()


def test_uninstall_gui_removes_userdata(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    _make_agent(hermes_home)
    userdata = tmp_path / "Hermes-userdata"
    userdata.mkdir()
    (userdata / "connection.json").write_text("{}")

    monkeypatch.setattr(gu, "packaged_gui_app_paths", lambda: [])
    monkeypatch.setattr(gu, "desktop_userdata_dir", lambda: userdata)

    gu.uninstall_gui(hermes_home)
    assert not userdata.exists()


def test_uninstall_gui_keeps_userdata_when_requested(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    _make_agent(hermes_home)
    userdata = tmp_path / "Hermes-userdata"
    userdata.mkdir()

    monkeypatch.setattr(gu, "packaged_gui_app_paths", lambda: [])
    monkeypatch.setattr(gu, "desktop_userdata_dir", lambda: userdata)

    gu.uninstall_gui(hermes_home, remove_userdata=False)
    assert userdata.exists()


def test_uninstall_gui_removes_packaged_bundle(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    _make_agent(hermes_home)
    bundle = tmp_path / "Hermes.app"
    (bundle / "Contents").mkdir(parents=True)

    monkeypatch.setattr(gu, "packaged_gui_app_paths", lambda: [bundle])
    monkeypatch.setattr(gu, "desktop_userdata_dir", lambda: tmp_path / "none")

    removed = gu.uninstall_gui(hermes_home)
    assert not bundle.exists()
    assert bundle in removed


def test_gui_install_summary_shape(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    _make_agent(hermes_home)
    _make_gui_build(hermes_home)
    monkeypatch.setattr(gu, "packaged_gui_app_paths", lambda: [])
    monkeypatch.setattr(gu, "desktop_userdata_dir", lambda: tmp_path / "none")

    summary = gu.gui_install_summary(hermes_home)
    # JSON-serializable primitives the desktop UI gates on.
    assert summary["agent_installed"] is True
    assert summary["gui_installed"] is True
    assert isinstance(summary["source_built_artifacts"], list)
    assert all(isinstance(p, str) for p in summary["source_built_artifacts"])
    assert summary["hermes_home"] == str(hermes_home)
    assert summary["platform"] == sys.platform


def test_userdata_dir_per_platform(monkeypatch):
    """userData path matches Electron's app.getPath('userData') for "Hermes"."""
    home = Path("/home/tester")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    monkeypatch.setattr(gu.sys, "platform", "darwin")
    assert gu.desktop_userdata_dir() == home / "Library" / "Application Support" / "Hermes"

    monkeypatch.setattr(gu.sys, "platform", "linux")
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    assert gu.desktop_userdata_dir() == home / ".config" / "Hermes"


def test_userdata_dir_windows(monkeypatch):
    home = Path("/home/tester")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(gu.sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", r"C:\Users\tester\AppData\Roaming")
    assert gu.desktop_userdata_dir() == Path(r"C:\Users\tester\AppData\Roaming") / "Hermes"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlink semantics")
def test_remove_path_handles_symlink(tmp_path):
    target = tmp_path / "real"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target)
    assert gu._remove_path(link) is True
    assert not link.exists()
    # The symlink is gone but its target is untouched.
    assert target.exists()


class _Args:
    """Minimal argparse-Namespace stand-in for run_uninstall."""

    def __init__(self, *, yes=False, full=False, gui=False, gui_summary=False):
        self.yes = yes
        self.full = full
        self.gui = gui
        self.gui_summary = gui_summary


def test_run_uninstall_yes_keep_data_is_non_interactive(tmp_path, monkeypatch):
    """``--yes`` (no ``--full``) runs with no prompt, sweeps the GUI, keeps data.

    We DO NOT spawn the real CLI here (its project_root removal would delete the
    test checkout) — we call run_uninstall in-process against a throwaway
    HERMES_HOME with all the destructive externals stubbed out.
    """
    import hermes_cli.uninstall as uninstall

    hermes_home = tmp_path / ".hermes"
    agent_root = hermes_home / "hermes-agent"
    (agent_root / "hermes_cli").mkdir(parents=True)
    (hermes_home / "config.yaml").write_text("x: 1\n")
    desktop = agent_root / "apps" / "desktop"
    (desktop / "release").mkdir(parents=True)
    (hermes_home / "desktop-build-stamp.json").write_text("{}")
    fake_code = tmp_path / "checkout"
    fake_code.mkdir()

    # Stub every destructive external so the test only exercises the control
    # flow + the real GUI sweep (which is safe inside tmp_path).
    monkeypatch.setattr(uninstall, "get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr(uninstall, "get_project_root", lambda: fake_code)
    monkeypatch.setattr(uninstall, "uninstall_gateway_service", lambda: False)
    monkeypatch.setattr(uninstall, "remove_path_from_shell_configs", lambda: [])
    monkeypatch.setattr(uninstall, "remove_wrapper_script", lambda: [])
    monkeypatch.setattr(uninstall, "remove_node_symlinks", lambda h: [])
    monkeypatch.setattr(uninstall, "_discover_named_profiles", lambda: [])
    # Make input() blow up so a regression that reaches a prompt fails loudly.
    monkeypatch.setattr("builtins.input", lambda *a, **k: pytest.fail("prompted in --yes mode"))

    from hermes_cli import gui_uninstall as gu_mod
    monkeypatch.setattr(gu_mod, "packaged_gui_app_paths", lambda: [])
    monkeypatch.setattr(gu_mod, "desktop_userdata_dir", lambda: tmp_path / "none")

    uninstall.run_uninstall(_Args(yes=True, full=False))

    # Code checkout removed, GUI artifacts swept, but user data preserved.
    assert not fake_code.exists()
    assert not (hermes_home / "desktop-build-stamp.json").exists()
    assert not (desktop / "release").exists()
    assert (hermes_home / "config.yaml").exists()
    assert hermes_home.exists()


def test_run_uninstall_yes_full_wipes_home(tmp_path, monkeypatch):
    """``--yes --full`` removes the whole HERMES_HOME non-interactively."""
    import hermes_cli.uninstall as uninstall

    hermes_home = tmp_path / ".hermes"
    (hermes_home / "hermes-agent" / "hermes_cli").mkdir(parents=True)
    (hermes_home / "config.yaml").write_text("x: 1\n")
    fake_code = tmp_path / "checkout"
    fake_code.mkdir()

    monkeypatch.setattr(uninstall, "get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr(uninstall, "get_project_root", lambda: fake_code)
    monkeypatch.setattr(uninstall, "uninstall_gateway_service", lambda: False)
    monkeypatch.setattr(uninstall, "remove_path_from_shell_configs", lambda: [])
    monkeypatch.setattr(uninstall, "remove_wrapper_script", lambda: [])
    monkeypatch.setattr(uninstall, "remove_node_symlinks", lambda h: [])
    monkeypatch.setattr(uninstall, "_discover_named_profiles", lambda: [])
    monkeypatch.setattr("builtins.input", lambda *a, **k: pytest.fail("prompted in --yes mode"))

    from hermes_cli import gui_uninstall as gu_mod
    monkeypatch.setattr(gu_mod, "packaged_gui_app_paths", lambda: [])
    monkeypatch.setattr(gu_mod, "desktop_userdata_dir", lambda: tmp_path / "none")

    uninstall.run_uninstall(_Args(yes=True, full=True))

    assert not hermes_home.exists()


def test_uninstall_module_main_gui_mode(tmp_path, monkeypatch):
    """`python -m hermes_cli.uninstall --mode gui` runs the GUI-only path.

    This is the lightweight, venv-independent entrypoint the desktop launches
    with a system Python (so lite/full don't rmtree their own running venv on
    Windows). Verify it dispatches by mode without prompting.
    """
    import hermes_cli.uninstall as uninstall

    hermes_home = tmp_path / ".hermes"
    agent_root = hermes_home / "hermes-agent"
    (agent_root / "hermes_cli").mkdir(parents=True)
    desktop = agent_root / "apps" / "desktop"
    (desktop / "release").mkdir(parents=True)
    (hermes_home / "desktop-build-stamp.json").write_text("{}")
    (hermes_home / "config.yaml").write_text("x: 1\n")

    monkeypatch.setattr(uninstall, "get_hermes_home", lambda: hermes_home)
    from hermes_cli import gui_uninstall as gu_mod
    monkeypatch.setattr(gu_mod, "packaged_gui_app_paths", lambda: [])
    monkeypatch.setattr(gu_mod, "desktop_userdata_dir", lambda: tmp_path / "none")
    monkeypatch.setattr(gu_mod, "get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr("builtins.input", lambda *a, **k: pytest.fail("prompted in module main"))

    rc = uninstall.main(["--mode", "gui"])
    assert rc == 0
    # GUI swept, agent + config kept (gui-only contract).
    assert not (desktop / "release").exists()
    assert not (hermes_home / "desktop-build-stamp.json").exists()
    assert (agent_root / "hermes_cli").exists()
    assert (hermes_home / "config.yaml").exists()


def test_uninstall_module_main_rejects_bad_mode():
    """An invalid --mode exits non-zero (argparse), never silently full-wipes."""
    import hermes_cli.uninstall as uninstall

    with pytest.raises(SystemExit) as exc:
        uninstall.main(["--mode", "nuke"])
    assert exc.value.code != 0


def test_uninstall_args_namespace_mode_mapping():
    """_UninstallArgs maps mode → the gui/full flags run_uninstall reads."""
    import hermes_cli.uninstall as uninstall

    gui = uninstall._UninstallArgs(mode="gui")
    assert gui.gui is True and gui.full is False and gui.yes is True

    lite = uninstall._UninstallArgs(mode="lite")
    assert lite.gui is False and lite.full is False and lite.yes is True

    full = uninstall._UninstallArgs(mode="full")
    assert full.gui is False and full.full is True and full.yes is True
