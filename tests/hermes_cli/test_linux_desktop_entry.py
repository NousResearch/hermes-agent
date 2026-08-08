"""Tests for the Linux XDG desktop entry installed by ``hermes desktop``."""

from __future__ import annotations

import errno
import stat
from pathlib import Path

import pytest

import utils
from hermes_cli import linux_desktop_entry as lde


@pytest.fixture
def xdg_home(tmp_path, monkeypatch) -> Path:
    data_home = tmp_path / "xdg-data"
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    monkeypatch.setattr(lde.sys, "platform", "linux")
    return data_home


def _make_project(tmp_path: Path) -> Path:
    root = tmp_path / "hermes-agent"
    icon = root / "apps" / "desktop" / "assets" / "icon.png"
    icon.parent.mkdir(parents=True)
    icon.write_bytes(b"\x89PNG fake")
    return root


def _make_executable(path: Path) -> Path:
    """Create an executable stand-in for a hermes entry point."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _fake_which(monkeypatch, hermes: "str | None") -> None:
    """Pin ``shutil.which("hermes")``.

    ``relaunch`` and this module share the one ``shutil`` module, so a
    single patch covers both lookups.
    """
    monkeypatch.setattr(
        lde.shutil, "which", lambda name: hermes if name == "hermes" else None
    )


def _parse(entry_text: str) -> dict:
    values = {}
    for line in entry_text.splitlines():
        if "=" in line and not line.startswith("["):
            key, val = line.split("=", 1)
            values[key] = val
    return values


def test_install_writes_entry_with_absolute_exec_and_icon(tmp_path, xdg_home, monkeypatch):
    root = _make_project(tmp_path)
    hermes_bin = tmp_path / "bin" / "hermes"
    hermes_bin.parent.mkdir()
    hermes_bin.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "hermes_cli.relaunch.resolve_hermes_bin", lambda: str(hermes_bin)
    )
    monkeypatch.setattr(lde, "refresh_desktop_databases", lambda _dir: [])

    entry = lde.install_desktop_entry(root)

    assert entry == xdg_home / "applications" / "hermes.desktop"
    values = _parse(entry.read_text(encoding="utf-8"))

    # Exec must be the absolute path of the resolved binary. The launcher
    # runs with a minimal PATH, so a bare `hermes` would not resolve.
    assert values["Exec"] == f"{hermes_bin} desktop"
    assert Path(values["Exec"].split(" ")[0]).is_absolute()

    # Icon must be an absolute path to the real icon in the checkout.
    icon_path = Path(values["Icon"])
    assert icon_path.is_absolute()
    assert icon_path == lde.icon_path(root)
    assert icon_path.read_bytes() == b"\x89PNG fake"

    assert values["Type"] == "Application"
    assert values["Name"] == "Hermes"
    assert values["Terminal"] == "false"


def test_installed_entry_is_executable(tmp_path, xdg_home, monkeypatch):
    root = _make_project(tmp_path)
    monkeypatch.setattr("hermes_cli.relaunch.resolve_hermes_bin", lambda: "/usr/bin/hermes")
    monkeypatch.setattr(lde, "refresh_desktop_databases", lambda _dir: [])

    entry = lde.install_desktop_entry(root)

    assert entry.stat().st_mode & stat.S_IXUSR


def test_exec_falls_back_to_interpreter_module(tmp_path, xdg_home, monkeypatch):
    root = _make_project(tmp_path)
    monkeypatch.setattr("hermes_cli.relaunch.resolve_hermes_bin", lambda: None)
    monkeypatch.setattr(lde, "refresh_desktop_databases", lambda _dir: [])

    entry = lde.install_desktop_entry(root)
    exec_line = _parse(entry.read_text(encoding="utf-8"))["Exec"]

    assert exec_line.endswith("-m hermes_cli.main desktop")
    assert Path(exec_line.split(" ")[0]).is_absolute()


# ---------------------------------------------------------------------------
# Exec must not depend on what launched this process (#80439)
# ---------------------------------------------------------------------------


def test_exec_prefers_path_wrapper_over_checkout_argv0(tmp_path, xdg_home, monkeypatch):
    root = _make_project(tmp_path)
    checkout_script = _make_executable(root / "hermes")
    wrapper = _make_executable(tmp_path / "local" / "bin" / "hermes")
    # The desktop entry launched us, so argv[0] is the checkout script.
    monkeypatch.setattr(lde.sys, "argv", [str(checkout_script), "desktop"])
    _fake_which(monkeypatch, str(wrapper))
    monkeypatch.setattr(lde, "refresh_desktop_databases", lambda _dir: [])

    entry = lde.install_desktop_entry(root)
    exec_line = _parse(entry.read_text(encoding="utf-8"))["Exec"]

    expected = lde._quote_exec_arg(str(wrapper.resolve()))
    assert exec_line == f"{expected} desktop"
    assert str(checkout_script) not in exec_line


def test_exec_rejects_checkout_argv0_when_no_wrapper_on_path(tmp_path, xdg_home, monkeypatch):
    root = _make_project(tmp_path)
    checkout_script = _make_executable(root / "hermes")
    monkeypatch.setattr(lde.sys, "argv", [str(checkout_script), "desktop"])
    _fake_which(monkeypatch, None)
    monkeypatch.setattr(lde, "refresh_desktop_databases", lambda _dir: [])

    entry = lde.install_desktop_entry(root)
    exec_line = _parse(entry.read_text(encoding="utf-8"))["Exec"]

    # The checkout's bare launcher runs under `/usr/bin/env python3` with
    # no venv shim, so a cold menu launch cannot import hermes_cli through
    # it. Persisting it strands the entry permanently.
    assert str(checkout_script) not in exec_line
    assert exec_line.endswith("-m hermes_cli.main desktop")
    assert Path(exec_line.split(" ")[0]).is_absolute()


def test_exec_accepts_argv0_outside_the_checkout(tmp_path, xdg_home, monkeypatch):
    root = _make_project(tmp_path)
    wrapper = _make_executable(tmp_path / "opt" / "bin" / "hermes")
    monkeypatch.setattr(lde.sys, "argv", [str(wrapper), "desktop"])
    _fake_which(monkeypatch, None)
    monkeypatch.setattr(lde, "refresh_desktop_databases", lambda _dir: [])

    entry = lde.install_desktop_entry(root)
    exec_line = _parse(entry.read_text(encoding="utf-8"))["Exec"]

    # Only checkout-internal entry points are rejected. An installed
    # wrapper reached through argv[0] is still the best value to persist.
    expected = lde._quote_exec_arg(str(wrapper.resolve()))
    assert exec_line == f"{expected} desktop"


def test_entry_is_stable_across_a_relaunch_through_itself(tmp_path, xdg_home, monkeypatch):
    root = _make_project(tmp_path)
    checkout_script = _make_executable(root / "hermes")
    wrapper = _make_executable(tmp_path / "local" / "bin" / "hermes")
    _fake_which(monkeypatch, str(wrapper))
    refreshes: list[Path] = []
    monkeypatch.setattr(lde, "refresh_desktop_databases", lambda d: refreshes.append(d) or [])

    # Launch one: from a terminal, through the installed wrapper.
    monkeypatch.setattr(lde.sys, "argv", [str(wrapper), "desktop"])
    entry = lde.install_desktop_entry(root)
    first = entry.read_text(encoding="utf-8")
    assert len(refreshes) == 1

    # Launch two: the menu runs the entry, so argv[0] is now whatever the
    # entry pointed at. Feed back the worst case.
    monkeypatch.setattr(lde.sys, "argv", [str(checkout_script), "desktop"])
    lde.install_desktop_entry(root)

    assert entry.read_text(encoding="utf-8") == first
    # No rewrite means no menu-cache churn, so Plasma keeps the taskbar
    # pin associated with this entry instead of spawning a second window.
    assert len(refreshes) == 1


def test_install_is_idempotent_and_skips_cache_refresh(tmp_path, xdg_home, monkeypatch):
    root = _make_project(tmp_path)
    monkeypatch.setattr("hermes_cli.relaunch.resolve_hermes_bin", lambda: "/usr/bin/hermes")
    calls: list[Path] = []
    monkeypatch.setattr(lde, "refresh_desktop_databases", lambda d: calls.append(d) or [])

    lde.install_desktop_entry(root)
    assert len(calls) == 1

    # Unchanged content → no rewrite, no menu-cache churn on every launch.
    lde.install_desktop_entry(root)
    assert len(calls) == 1


def test_install_publishes_atomically_and_leaves_no_temp(tmp_path, xdg_home, monkeypatch):
    root = _make_project(tmp_path)
    monkeypatch.setattr(lde, "refresh_desktop_databases", lambda _dir: [])

    monkeypatch.setattr("hermes_cli.relaunch.resolve_hermes_bin", lambda: "/usr/bin/hermes")
    entry = lde.install_desktop_entry(root)
    # Change the rendered contents so this is a real overwrite, not the
    # unchanged-contents fast path.
    monkeypatch.setattr("hermes_cli.relaunch.resolve_hermes_bin", lambda: "/opt/hermes/bin/hermes")
    assert lde.install_desktop_entry(root) == entry

    assert list(entry.parent.iterdir()) == [entry]
    assert _parse(entry.read_text(encoding="utf-8"))["Exec"] == "/opt/hermes/bin/hermes desktop"
    assert stat.S_IMODE(entry.stat().st_mode) == 0o755


def test_failed_publish_leaves_the_existing_entry_intact(tmp_path, xdg_home, monkeypatch):
    root = _make_project(tmp_path)
    monkeypatch.setattr(lde, "refresh_desktop_databases", lambda _dir: [])
    monkeypatch.setattr("hermes_cli.relaunch.resolve_hermes_bin", lambda: "/usr/bin/hermes")

    entry = lde.install_desktop_entry(root)
    published = entry.read_text(encoding="utf-8")

    def boom(_tmp, _target):
        raise OSError(errno.EIO, "the disk went away mid-publish")

    monkeypatch.setattr(utils, "atomic_replace", boom)
    monkeypatch.setattr("hermes_cli.relaunch.resolve_hermes_bin", lambda: "/opt/hermes/bin/hermes")

    assert lde.install_desktop_entry(root) is None
    # A truncate-then-write would have left a partial or zero-length
    # entry here, dropping Hermes out of the menu and killing the pin.
    assert entry.read_text(encoding="utf-8") == published
    assert list(entry.parent.iterdir()) == [entry]


def test_install_without_source_icon_uses_themed_name(tmp_path, xdg_home, monkeypatch):
    root = tmp_path / "hermes-agent"
    root.mkdir()
    monkeypatch.setattr("hermes_cli.relaunch.resolve_hermes_bin", lambda: "/usr/bin/hermes")
    monkeypatch.setattr(lde, "refresh_desktop_databases", lambda _dir: [])

    entry = lde.install_desktop_entry(root)

    # A broken absolute path renders as no icon. The themed name resolves
    # when Hermes is installed some other way.
    assert _parse(entry.read_text(encoding="utf-8"))["Icon"] == "hermes"


@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_install_is_a_noop_off_linux(tmp_path, monkeypatch, platform):
    monkeypatch.setattr(lde.sys, "platform", platform)
    assert lde.install_desktop_entry(_make_project(tmp_path)) is None


# ---------------------------------------------------------------------------
# Cache refresh tool gating
# ---------------------------------------------------------------------------


def _stub_tools(monkeypatch, available: "set[str]") -> "list[list[str]]":
    ran: list[list[str]] = []
    monkeypatch.setattr(
        lde.shutil, "which", lambda name: f"/usr/bin/{name}" if name in available else None
    )
    monkeypatch.setattr(lde, "_run_quiet", lambda cmd: ran.append(cmd) or True)
    return ran


def test_refresh_runs_kbuildsycoca6_when_present(monkeypatch, tmp_path):
    ran = _stub_tools(monkeypatch, {"update-desktop-database", "kbuildsycoca6"})

    tools = lde.refresh_desktop_databases(tmp_path)

    assert tools == ["update-desktop-database", "kbuildsycoca6"]
    assert ran == [
        ["/usr/bin/update-desktop-database", str(tmp_path)],
        ["/usr/bin/kbuildsycoca6", "--noincremental"],
    ]


def test_refresh_falls_back_to_kbuildsycoca5(monkeypatch, tmp_path):
    ran = _stub_tools(monkeypatch, {"kbuildsycoca5"})

    tools = lde.refresh_desktop_databases(tmp_path)

    assert tools == ["kbuildsycoca5"]
    assert ran == [["/usr/bin/kbuildsycoca5", "--noincremental"]]


def test_refresh_prefers_kbuildsycoca6_over_5(monkeypatch, tmp_path):
    ran = _stub_tools(monkeypatch, {"kbuildsycoca6", "kbuildsycoca5"})

    lde.refresh_desktop_databases(tmp_path)

    assert [cmd[0] for cmd in ran] == ["/usr/bin/kbuildsycoca6"]


def test_refresh_skips_missing_tools(monkeypatch, tmp_path):
    ran = _stub_tools(monkeypatch, set())

    assert lde.refresh_desktop_databases(tmp_path) == []
    assert ran == []


def test_refresh_reports_only_tools_that_succeeded(monkeypatch, tmp_path):
    monkeypatch.setattr(lde.shutil, "which", lambda name: f"/usr/bin/{name}")
    # update-desktop-database fails (exit != 0). kbuildsycoca6 succeeds.
    monkeypatch.setattr(lde, "_run_quiet", lambda cmd: "kbuildsycoca" in cmd[0])

    assert lde.refresh_desktop_databases(tmp_path) == ["kbuildsycoca6"]


def test_run_quiet_swallows_missing_binary(tmp_path):
    assert lde._run_quiet([str(tmp_path / "definitely-not-a-binary")]) is False


def test_exec_arg_quoting_handles_spaces(tmp_path, xdg_home, monkeypatch):
    root = _make_project(tmp_path)
    spaced = tmp_path / "my apps" / "hermes"
    spaced.parent.mkdir()
    spaced.write_text("", encoding="utf-8")
    monkeypatch.setattr("hermes_cli.relaunch.resolve_hermes_bin", lambda: str(spaced))
    monkeypatch.setattr(lde, "refresh_desktop_databases", lambda _dir: [])

    entry = lde.install_desktop_entry(root)
    exec_line = _parse(entry.read_text(encoding="utf-8"))["Exec"]

    assert exec_line == f'"{spaced}" desktop'
