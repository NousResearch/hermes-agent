"""Tests for the Linux XDG desktop entry installed by ``hermes desktop``."""

from __future__ import annotations

import errno
import os
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


#: What an installed console script's shebang looks like: its own
#: interpreter, hardcoded absolute, with no ``PATH`` lookup at exec time.
INSTALLED_SHEBANG = "#!/opt/hermes/venv/bin/python3\n"
#: What the checkout's bare ``hermes`` launcher looks like: the
#: interpreter is whatever ``PATH`` yields when the script runs.
ENV_PYTHON_SHEBANG = "#!/usr/bin/env python3\n"


def _make_executable(path: Path, shebang: str = INSTALLED_SHEBANG) -> Path:
    """Create an executable stand-in for a hermes entry point."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(shebang, encoding="utf-8")
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
    checkout_script = _make_executable(root / "hermes", ENV_PYTHON_SHEBANG)
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
    checkout_script = _make_executable(root / "hermes", ENV_PYTHON_SHEBANG)
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


@pytest.mark.skipif(os.name == "nt", reason="symlink creation needs privileges on Windows")
def test_exec_keeps_the_lexical_venv_interpreter_path(tmp_path, xdg_home, monkeypatch):
    """The interpreter fallback persists the path Python was invoked
    through, not the symlink target behind it.

    A venv's ``bin/python`` is a symlink to a base interpreter, and CPython
    decides it is inside a venv by finding ``pyvenv.cfg`` beside the
    *invocation* path. A uv-created venv points that symlink at
    ``~/.local/share/uv/python/...``, outside the venv, so a dereferenced
    Exec starts an interpreter that has never heard of the venv's
    ``site-packages`` and cannot import ``hermes_cli`` at all. Absolute is
    not enough — the entry has to stay inside the venv.
    """
    root = _make_project(tmp_path)
    base = tmp_path / "uv" / "python" / "cpython-3.11" / "bin" / "python3.11"
    base.parent.mkdir(parents=True)
    base.write_bytes(b"")
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(base)
    (venv_python.parent.parent / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")

    monkeypatch.setattr(lde.sys, "executable", str(venv_python))
    monkeypatch.setattr("hermes_cli.relaunch.resolve_hermes_bin", lambda: None)
    monkeypatch.setattr(lde, "refresh_desktop_databases", lambda _dir: [])

    entry = lde.install_desktop_entry(root)
    exec_line = _parse(entry.read_text(encoding="utf-8"))["Exec"]

    # Guard the fixture: this asserts nothing unless the interpreter really
    # is a symlink pointing somewhere else.
    assert venv_python.is_symlink()
    assert os.path.realpath(venv_python) != str(venv_python)

    assert exec_line == f"{lde._quote_exec_arg(str(venv_python))} -m hermes_cli.main desktop"
    assert str(base) not in exec_line
    # The reason the lexical path matters: pyvenv.cfg is beside it, so
    # CPython still finds the venv. Beside the resolved target it is not.
    persisted = Path(exec_line.split(" ", 1)[0])
    assert (persisted.parent.parent / "pyvenv.cfg").is_file()


def test_exec_rejects_env_python_wrapper_outside_the_checkout(tmp_path, xdg_home, monkeypatch):
    """A ``#!/usr/bin/env python3`` wrapper is not durable wherever it lives.

    A hand-rolled ``~/bin/hermes``, or a console script from a non-venv
    editable install, sits outside the checkout and so survives
    ``_is_inside_checkout`` — yet it still looks up ``python3`` on ``PATH``
    when it runs. A cold menu launch supplies the desktop session's
    ``PATH``, not the shell's, and the interpreter it lands on need not
    have Hermes on ``sys.path``. Rejecting by location alone is not enough.
    """
    root = _make_project(tmp_path)
    wrapper = _make_executable(tmp_path / "home" / "bin" / "hermes", ENV_PYTHON_SHEBANG)
    monkeypatch.setattr(lde.sys, "argv", [str(wrapper), "desktop"])
    _fake_which(monkeypatch, str(wrapper))
    monkeypatch.setattr(lde, "refresh_desktop_databases", lambda _dir: [])

    entry = lde.install_desktop_entry(root)
    exec_line = _parse(entry.read_text(encoding="utf-8"))["Exec"]

    assert str(wrapper) not in exec_line
    expected = lde._quote_exec_arg(os.path.abspath(lde.sys.executable))
    assert exec_line == f"{expected} -m hermes_cli.main desktop"


def test_exec_keeps_an_interpreter_under_a_directory_named_envs(tmp_path, xdg_home, monkeypatch):
    """The shebang check keys off the program's basename, not a substring.

    ``#!/home/u/envs/hermes/bin/python3`` hardcodes its interpreter — that
    is what an installed console script looks like. Matching ``env``
    loosely would discard it and drop back to the ambient interpreter for
    no reason.
    """
    root = _make_project(tmp_path)
    wrapper = _make_executable(
        tmp_path / "envs" / "hermes" / "bin" / "hermes",
        "#!/home/u/envs/hermes/bin/python3\n",
    )
    monkeypatch.setattr(lde.sys, "argv", [str(wrapper), "desktop"])
    _fake_which(monkeypatch, None)
    monkeypatch.setattr(lde, "refresh_desktop_databases", lambda _dir: [])

    entry = lde.install_desktop_entry(root)
    exec_line = _parse(entry.read_text(encoding="utf-8"))["Exec"]

    assert exec_line == f"{lde._quote_exec_arg(str(wrapper.resolve()))} desktop"


@pytest.mark.parametrize(
    "first_bytes,expected",
    [
        (b"#!/usr/bin/env python3\n", True),
        (b"#!/usr/bin/env python\n", True),
        (b"#!/usr/bin/env -S python3 -u\n", True),
        (b"#!/opt/hermes/venv/bin/python3\n", False),
        (b"#!/home/u/envs/hermes/bin/python3\n", False),
        (b"#!/usr/bin/env node\n", False),
        (b"#!/bin/sh\n", False),
        (b"#!/usr/bin/env\n", False),
        (b"\x7fELF\x02\x01\x01\x00", False),
        (b"", False),
    ],
)
def test_env_python_wrapper_detection(tmp_path, first_bytes, expected):
    candidate = tmp_path / "hermes"
    candidate.write_bytes(first_bytes)
    assert lde._is_env_python_wrapper(str(candidate)) is expected


def test_env_python_wrapper_detection_tolerates_an_unreadable_candidate(tmp_path):
    # An unreadable candidate is not evidence of a PATH dependency, and it
    # must not raise on the launch path either.
    assert lde._is_env_python_wrapper(str(tmp_path / "does-not-exist")) is False


def test_exec_accepts_argv0_outside_the_checkout(tmp_path, xdg_home, monkeypatch):
    root = _make_project(tmp_path)
    wrapper = _make_executable(tmp_path / "opt" / "bin" / "hermes")
    monkeypatch.setattr(lde.sys, "argv", [str(wrapper), "desktop"])
    _fake_which(monkeypatch, None)
    monkeypatch.setattr(lde, "refresh_desktop_databases", lambda _dir: [])

    entry = lde.install_desktop_entry(root)
    exec_line = _parse(entry.read_text(encoding="utf-8"))["Exec"]

    # Only non-durable entry points are rejected. An installed wrapper
    # that names its own interpreter is still the best value to persist,
    # even when argv[0] is how we found it.
    expected = lde._quote_exec_arg(str(wrapper.resolve()))
    assert exec_line == f"{expected} desktop"


def test_entry_is_stable_across_a_relaunch_through_itself(tmp_path, xdg_home, monkeypatch):
    root = _make_project(tmp_path)
    checkout_script = _make_executable(root / "hermes", ENV_PYTHON_SHEBANG)
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


@pytest.mark.macos_only
def test_install_is_a_noop_on_macos(tmp_path):
    """Faking darwin only renamed the host — the real macOS runner is the
    only place the `sys.platform` guard is exercised against a real host."""
    assert lde.install_desktop_entry(_make_project(tmp_path)) is None


@pytest.mark.windows_only
def test_install_is_a_noop_on_windows(tmp_path):
    """As above for Windows: a fake left POSIX paths and a POSIX XDG layout
    in place, so the no-op was never proven against a real one."""
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
