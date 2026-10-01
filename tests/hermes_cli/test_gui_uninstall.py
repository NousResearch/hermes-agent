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


@pytest.mark.platforms("linux")
def test_uninstall_removes_launcher_entry_and_refreshes_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))

    from hermes_cli import linux_desktop_entry as lde

    entry = lde.desktop_entry_path()
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text("x", encoding="utf-8")
    # The pre-rename entry lives on as a hidden alias of the app-id entry (#124492);
    # a GUI uninstall must take it with the real one.
    legacy_alias = entry.with_name(lde.LEGACY_DESKTOP_ENTRY_NAME)
    legacy_alias.write_text("x", encoding="utf-8")

    refreshed: list[Path] = []
    monkeypatch.setattr(
        lde, "refresh_desktop_databases", lambda d: refreshed.append(d) or ["kbuildsycoca6"]
    )

    hermes_home = tmp_path / ".hermes"
    _make_agent(hermes_home)
    icon = lde.icon_path(hermes_home / "hermes-agent")
    icon.parent.mkdir(parents=True, exist_ok=True)
    icon.write_bytes(b"\x89PNG")
    monkeypatch.setattr(gu, "desktop_userdata_dir", lambda: tmp_path / "none")

    removed = gu.uninstall_gui(hermes_home)

    assert entry in removed and not entry.exists()
    assert legacy_alias in removed and not legacy_alias.exists()
    assert refreshed == [entry.parent]
    # The icon lives in the checkout. A GUI uninstall must not delete it.
    assert lde.icon_path(hermes_home / "hermes-agent").exists()
    # The agent itself survives a GUI uninstall.
    assert (hermes_home / "hermes-agent" / "hermes_cli").is_dir()


@pytest.mark.platforms("posix")  # POSIX symlink semantics
def test_remove_path_handles_symlink(tmp_path):
    target = tmp_path / "real"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target)
    assert gu._remove_path(link) is True
    assert not link.exists()
    # The symlink is gone but its target is untouched.
    assert target.exists()


def test_uninstall_args_namespace_mode_mapping():
    """_UninstallArgs maps mode → the gui/full flags run_uninstall reads."""
    import hermes_cli.uninstall as uninstall

    gui = uninstall._UninstallArgs(mode="gui")
    assert gui.gui is True and gui.full is False and gui.yes is True

    lite = uninstall._UninstallArgs(mode="lite")
    assert lite.gui is False and lite.full is False and lite.yes is True

    full = uninstall._UninstallArgs(mode="full")
    assert full.gui is False and full.full is True and full.yes is True


def _make_desktop_build(checkout: Path) -> Path:
    """Create the `hermes desktop` build output inside *checkout* (a source install root)."""
    desktop = checkout / "apps" / "desktop"
    (desktop / "dist").mkdir(parents=True)
    (desktop / "dist" / "index.html").write_text("<html>")
    (desktop / "release" / "linux-unpacked").mkdir(parents=True)
    (desktop / "node_modules").mkdir(parents=True)
    (checkout / "node_modules").mkdir(parents=True)
    return desktop


def test_custom_path_checkout_artifacts_are_discovered_and_removed(tmp_path, monkeypatch):
    """A git checkout outside $HERMES_HOME owns the artifacts its desktop build wrote.

    Reporting only the install.sh layout let `hermes uninstall --gui` claim success while the
    build output the next `hermes update` keys off stayed on disk — so the app was rebuilt.
    """
    hermes_home = tmp_path / ".hermes"
    _make_agent(hermes_home)
    checkout = tmp_path / "Hermes-Agent"  # `hermes --version`'s install directory
    desktop = _make_desktop_build(checkout)
    monkeypatch.setattr(gu, "packaged_gui_app_paths", lambda: [])
    monkeypatch.setattr(gu, "desktop_userdata_dir", lambda: tmp_path / "none")

    summary = gu.gui_install_summary(hermes_home, project_root=checkout)
    assert summary["gui_installed"] is True
    assert str(desktop / "dist") in summary["source_built_artifacts"]
    assert str(desktop / "release") in summary["source_built_artifacts"]

    removed = gu.uninstall_gui(hermes_home, project_root=checkout, remove_userdata=False)

    assert not (desktop / "dist").exists()
    assert not (desktop / "release").exists()
    assert not (desktop / "node_modules").exists()
    # The workspace-root node_modules is shared with the TUI/dashboard workspaces, so it is kept
    # (matching `uninstall --gui`'s behaviour on the install.sh layout).
    assert (checkout / "node_modules").exists()
    assert desktop / "dist" in removed and desktop / "release" in removed
    # The agent beside the checkout is never part of a GUI uninstall.
    assert (hermes_home / "hermes-agent" / "hermes_cli").is_dir()


def test_standard_layout_root_is_not_listed_twice(tmp_path):
    """`$HERMES_HOME/hermes-agent` is the standard layout; both roots resolving there is one set."""
    hermes_home = tmp_path / ".hermes"

    artifacts = gu.source_built_gui_artifacts(hermes_home, project_root=hermes_home / "hermes-agent")

    assert len(artifacts) == len(set(artifacts))


def test_default_project_root_is_the_running_checkout(tmp_path):
    """With no caller-supplied root, discovery keys off the checkout this code runs from."""
    artifacts = gu.source_built_gui_artifacts(tmp_path / ".hermes")

    assert gu._running_checkout_root() / "apps" / "desktop" / "release" in artifacts

