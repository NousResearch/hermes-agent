"""Tests for the Linux XDG desktop entry installed by ``hermes desktop``."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest
import sys

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


# #90292: the shell installer's bash wrapper makes argv[0] the repo `hermes`
# python script whose `#!/usr/bin/env python3` shebang resolves to the SYSTEM
# interpreter when the DE spawns the .desktop entry → ModuleNotFoundError,
# silent (Terminal=false). The Exec line must prefix sys.executable for any
# resolved bin that is a python script escaping the running venv.
def test_exec_prefixes_interpreter_for_env_shebang_python_script(tmp_path, xdg_home, monkeypatch):
    import os
    import sys

    root = _make_project(tmp_path)
    hermes_bin = tmp_path / "bin" / "hermes"
    hermes_bin.parent.mkdir()
    hermes_bin.write_text("#!/usr/bin/env python3\nimport hermes_cli\n", encoding="utf-8")
    hermes_bin.chmod(0o755)
    monkeypatch.setattr("hermes_cli.relaunch.resolve_hermes_bin", lambda: str(hermes_bin))
    monkeypatch.setattr(lde, "refresh_desktop_databases", lambda _dir: [])

    entry = lde.install_desktop_entry(root)
    exec_line = _parse(entry.read_text(encoding="utf-8"))["Exec"]

    # Deliberately NOT ``Path(sys.executable).resolve()``. That was this
    # assertion before #92086, and it is the bug: on POSIX a venv's
    # ``bin/python`` is a symlink to the base interpreter, so resolving it
    # named an interpreter without ``hermes_cli`` and the test agreed with
    # the code. Written independently of the implementation, from what the
    # Exec line is FOR -- an absolute path to an interpreter that can import
    # Hermes -- rather than by calling the helper back.
    interpreter = os.path.abspath(sys.executable)
    assert exec_line.split(" ")[0].strip('"') == interpreter
    assert str(hermes_bin) in exec_line
    assert exec_line.endswith("desktop")


def test_exec_leaves_shell_wrapper_launchers_alone(tmp_path, xdg_home, monkeypatch):
    root = _make_project(tmp_path)
    hermes_bin = tmp_path / "bin" / "hermes"
    hermes_bin.parent.mkdir()
    hermes_bin.write_text('#!/bin/bash\nexec /opt/hermes/venv/bin/python "$@"\n', encoding="utf-8")
    hermes_bin.chmod(0o755)
    monkeypatch.setattr("hermes_cli.relaunch.resolve_hermes_bin", lambda: str(hermes_bin))
    monkeypatch.setattr(lde, "refresh_desktop_databases", lambda _dir: [])

    entry = lde.install_desktop_entry(root)
    exec_line = _parse(entry.read_text(encoding="utf-8"))["Exec"]

    # A bash wrapper execs the venv python itself — no interpreter prefix.
    assert exec_line == f"{hermes_bin} desktop"


def test_exec_leaves_venv_shebang_scripts_alone(tmp_path, xdg_home, monkeypatch):
    import sys

    root = _make_project(tmp_path)
    hermes_bin = tmp_path / "bin" / "hermes"
    hermes_bin.parent.mkdir()
    # Despite this test's name, `.resolve()` is the BASE interpreter on any
    # POSIX venv -- `bin/python` is a symlink to it (#92086), which is how
    # this test managed to pass against the bug it looks like it covers.
    # Kept as-is, because a console script installed against the base
    # interpreter is a real case that must keep working. The venv spelling
    # the name promises is covered by
    # `test_needs_interpreter_accepts_a_shebang_naming_the_venv_python`.
    interpreter = str(Path(sys.executable).resolve())
    hermes_bin.write_text(f"#!{interpreter}\nimport hermes_cli\n", encoding="utf-8")
    hermes_bin.chmod(0o755)
    monkeypatch.setattr("hermes_cli.relaunch.resolve_hermes_bin", lambda: str(hermes_bin))
    monkeypatch.setattr(lde, "refresh_desktop_databases", lambda _dir: [])

    entry = lde.install_desktop_entry(root)
    exec_line = _parse(entry.read_text(encoding="utf-8"))["Exec"]

    # Console-script with the venv's own interpreter in the shebang: correct
    # as-is, prefixing would only add noise.
    assert exec_line == f"{hermes_bin} desktop"


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

# ── #92086: sys.executable is a symlink inside a venv ───────────────────────
#
# `python -m venv` on POSIX and `uv venv` both create `bin/python` as a
# SYMLINK to the base interpreter. Resolving it leaves the venv, and the base
# interpreter has none of the venv's site-packages -- so a `.desktop` entry
# built from the resolved path dies on `import hermes_cli`, silently, because
# the entry sets Terminal=false.
#
# These tests drive `resolve_exec_command` / `_needs_interpreter` directly
# rather than through `install_desktop_entry`, so they assert on the decision
# instead of on a rendered Exec line with platform-specific quoting.


def _symlinked_venv(tmp_path):
    """A base interpreter plus a venv whose `python` is a symlink to it."""
    import os

    # Deliberately mixed-case: `_needs_interpreter` lowercases the shebang it
    # reads, so an install path with any uppercase in it is where a
    # case-blind comparison stops matching.
    base_dir = tmp_path / "Uv" / "cpython-3.11" / "bin"
    base_dir.mkdir(parents=True)
    base = base_dir / "python3.11"
    base.write_text("", encoding="utf-8")

    venv_dir = tmp_path / "Projects" / "venv" / "bin"
    venv_dir.mkdir(parents=True)
    venv = venv_dir / "python"
    try:
        os.symlink(base, venv)
    except (OSError, NotImplementedError) as exc:  # unprivileged Windows
        pytest.skip(f"symlinks unavailable: {exc}")
    return base, venv


def test_running_interpreter_keeps_the_venv_python(tmp_path, monkeypatch):
    """The whole bug in one assertion.

    `Path(sys.executable).resolve()` answers with the base interpreter, which
    cannot import `hermes_cli`. The venv's own `bin/python` can, and it is
    what belongs in `Exec=`.
    """
    base, venv = _symlinked_venv(tmp_path)
    monkeypatch.setattr(sys, "executable", str(venv))

    assert lde._running_interpreter() == Path(venv)
    assert Path(sys.executable).resolve() == Path(base).resolve(), (
        "precondition: this venv python really does resolve out of the venv"
    )


def test_needs_interpreter_accepts_a_shebang_naming_the_venv_python(
    tmp_path, monkeypatch
):
    """A console script installed INTO the venv is already correct.

    Before #92086 this answered True: the shebang named the venv's `bin`, the
    comparison used the resolved base `bin`, they did not match, and the entry
    was prefixed with an interpreter that cannot import Hermes.
    """
    _base, venv = _symlinked_venv(tmp_path)
    monkeypatch.setattr(sys, "executable", str(venv))

    script = tmp_path / "local-bin" / "hermes"
    script.parent.mkdir()
    script.write_text(f"#!{venv}\nfrom hermes_cli.main import main\n", encoding="utf-8")

    assert lde._needs_interpreter(script) is False


def test_needs_interpreter_accepts_a_shebang_naming_the_resolved_interpreter(
    tmp_path, monkeypatch
):
    """Both spellings of "inside this environment" still count.

    A script installed against the base interpreter carries the resolved path,
    and that was always accepted. Widening the check must not narrow it.
    """
    base, venv = _symlinked_venv(tmp_path)
    monkeypatch.setattr(sys, "executable", str(venv))

    script = tmp_path / "local-bin" / "hermes"
    script.parent.mkdir()
    script.write_text(f"#!{base}\nfrom hermes_cli.main import main\n", encoding="utf-8")

    assert lde._needs_interpreter(script) is False


def test_needs_interpreter_still_flags_env_python3(tmp_path, monkeypatch):
    """#90292 must keep working: a foreign shebang still gets the prefix."""
    _base, venv = _symlinked_venv(tmp_path)
    monkeypatch.setattr(sys, "executable", str(venv))

    script = tmp_path / "checkout" / "hermes"
    script.parent.mkdir()
    script.write_text(
        "#!/usr/bin/env python3\nfrom hermes_cli.main import main\n", encoding="utf-8"
    )

    assert lde._needs_interpreter(script) is True


def test_exec_prefix_names_the_venv_python_not_its_target(tmp_path, monkeypatch):
    """The prefixed form must carry the interpreter that has hermes_cli."""
    base, venv = _symlinked_venv(tmp_path)
    monkeypatch.setattr(sys, "executable", str(venv))

    script = tmp_path / "checkout" / "hermes"
    script.parent.mkdir()
    script.write_text(
        "#!/usr/bin/env python3\nfrom hermes_cli.main import main\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        "hermes_cli.relaunch.resolve_hermes_bin", lambda: str(script)
    )

    exec_command = lde.resolve_exec_command()

    assert lde._quote_exec_arg(str(venv)) in exec_command
    assert str(base) not in exec_command


def test_module_fallback_names_the_venv_python_too(tmp_path, monkeypatch):
    """The no-launcher branch builds from sys.executable as well."""
    base, venv = _symlinked_venv(tmp_path)
    monkeypatch.setattr(sys, "executable", str(venv))
    monkeypatch.setattr("hermes_cli.relaunch.resolve_hermes_bin", lambda: None)

    exec_command = lde.resolve_exec_command()

    assert lde._quote_exec_arg(str(venv)) in exec_command
    assert str(base) not in exec_command
    assert "-m hermes_cli.main" in exec_command
