"""Tests for the Linux XDG desktop entry installed by ``hermes desktop``."""

from __future__ import annotations

import os
import stat
import subprocess
import venv
from pathlib import Path

import pytest

from hermes_cli import linux_desktop_entry as lde

# Original subprocess.run, captured so tests can selectively stub the
# hermes_cli import-probe without disturbing other subprocess calls.
_orig_run = subprocess.run


class _FakeReturn:
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode

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
    interpreter = str(os.path.abspath(sys.executable))
    hermes_bin.write_text(f"#!{interpreter}\nimport hermes_cli\n", encoding="utf-8")
    hermes_bin.chmod(0o755)
    monkeypatch.setattr("hermes_cli.relaunch.resolve_hermes_bin", lambda: str(hermes_bin))
    monkeypatch.setattr(lde, "refresh_desktop_databases", lambda _dir: [])

    entry = lde.install_desktop_entry(root)
    exec_line = _parse(entry.read_text(encoding="utf-8"))["Exec"]

    # Console-script with the venv's own interpreter in the shebang: correct
    # as-is, prefixing would only add noise.
    assert exec_line == f"{hermes_bin} desktop"


@pytest.mark.skipif(os.name == "nt", reason="requires unprivileged symlink creation")
def test_needs_interpreter_preserves_venv_python_symlink(tmp_path, monkeypatch):
    base_python = tmp_path / "uv" / "python3.11"
    base_python.parent.mkdir()
    base_python.write_text("", encoding="utf-8")
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(base_python)

    hermes_bin = tmp_path / "hermes"
    hermes_bin.write_text(f"#!{venv_python}\nimport hermes_cli\n", encoding="utf-8")
    monkeypatch.setattr(lde.sys, "executable", str(venv_python))

    assert lde._needs_interpreter(hermes_bin) is False


def test_needs_interpreter_rejects_sibling_of_venv_bin(tmp_path, monkeypatch):
    venv_python = tmp_path / "venv" / "bin" / "python"
    sibling_python = tmp_path / "venv" / "bin-extra" / "python"
    hermes_bin = tmp_path / "hermes"
    hermes_bin.write_text(f"#!{sibling_python}\nimport hermes_cli\n", encoding="utf-8")
    monkeypatch.setattr(lde, "_running_interpreter", lambda: venv_python)

    # A path-component match (not substring) must reject a sibling dir:
    # ``venv/bin-extra/python`` is NOT inside ``venv/bin``.
    assert lde._needs_interpreter(hermes_bin) is True


def test_can_import_rejects_checkout_false_positive(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout_package = checkout / "hermes_cli"
    checkout_package.mkdir(parents=True)
    (checkout_package / "__init__.py").write_text("", encoding="utf-8")
    (checkout_package / "main.py").write_text("", encoding="utf-8")

    env_dir = tmp_path / "clean-venv"
    venv.EnvBuilder(with_pip=False).create(env_dir)
    interpreter = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    purelib = subprocess.check_output(
        [str(interpreter), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        text=True,
    ).strip()
    installed_package = Path(purelib) / "hermes_cli"
    installed_package.mkdir()
    (installed_package / "__init__.py").write_text("", encoding="utf-8")
    (installed_package / "main.py").write_text(
        "raise ModuleNotFoundError('missing runtime dependency')\n", encoding="utf-8"
    )
    monkeypatch.chdir(checkout)
    monkeypatch.setenv("PYTHONPATH", str(checkout))

    assert lde._can_import_hermes_cli(interpreter) is False

    (installed_package / "main.py").write_text("", encoding="utf-8")
    assert lde._can_import_hermes_cli(interpreter) is True


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


def test_exec_prefers_venv_wrapper_over_resolved_bin(tmp_path, xdg_home, monkeypatch):
    root = _make_project(tmp_path)
    # The bare launcher from the checkout: a python shebang whose interpreter
    # is not the one that can import hermes_cli. resolve_hermes_bin() may still
    # return it (e.g. when Hermes was launched via a uv/system python shim),
    # but the generated entry must NOT exec it directly.
    bare = tmp_path / "checkout" / "hermes"
    bare.parent.mkdir(parents=True)
    bare.write_text("#!/usr/bin/env python3\nfrom hermes_cli.main import main\n", encoding="utf-8")
    # Installed venv-backed wrapper on PATH (e.g. ~/.local/bin/hermes).
    wrapper = tmp_path / "bin" / "hermes"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text("#!/usr/bin/env bash\nexec venv/bin/python hermes \"$@\"\n", encoding="utf-8")
    monkeypatch.setattr("hermes_cli.relaunch.resolve_hermes_bin", lambda: str(bare))
    monkeypatch.setattr(lde.shutil, "which", lambda name: str(wrapper) if name == "hermes" else None)
    # Simulate sys.executable not being able to import hermes_cli (uv/system
    # shim), so the resolver must fall through to the PATH wrapper.
    monkeypatch.setattr(
        lde.subprocess, "run",
        lambda *a, **k: _FakeReturn(1) if "import hermes_cli" in str(a) else _orig_run(*a, **k),
    )
    monkeypatch.setattr(lde, "refresh_desktop_databases", lambda _dir: [])

    entry = lde.install_desktop_entry(root)
    exec_line = _parse(entry.read_text(encoding="utf-8"))["Exec"]

    # The venv wrapper must win over the bare launcher, otherwise the entry
    # fails silently (ModuleNotFoundError: hermes_cli) under Terminal=false.
    assert exec_line == f"{wrapper} desktop"


def test_exec_falls_back_to_module_when_no_wrapper(tmp_path, xdg_home, monkeypatch):
    root = _make_project(tmp_path)
    bare = tmp_path / "checkout" / "hermes"
    bare.parent.mkdir(parents=True)
    bare.write_text("#!/usr/bin/env python3\nfrom hermes_cli.main import main\n", encoding="utf-8")
    monkeypatch.setattr("hermes_cli.relaunch.resolve_hermes_bin", lambda: str(bare))
    monkeypatch.setattr(lde.shutil, "which", lambda name: None)
    # No working interpreter on the system path: fall back to -m.
    monkeypatch.setattr(
        lde.subprocess, "run",
        lambda *a, **k: _FakeReturn(1) if "import hermes_cli" in str(a) else _orig_run(*a, **k),
    )
    monkeypatch.setattr(lde, "refresh_desktop_databases", lambda _dir: [])

    entry = lde.install_desktop_entry(root)
    exec_line = _parse(entry.read_text(encoding="utf-8"))["Exec"]

    # Run via sys.executable -m as the last resort.
    assert exec_line.endswith("-m hermes_cli.main desktop")
    assert Path(exec_line.split(" ")[0]).is_absolute()


def test_exec_does_not_reuse_foreign_python_shebang_wrapper(tmp_path, xdg_home, monkeypatch):
    import sys

    root = _make_project(tmp_path)
    # A `hermes` on PATH whose shebang is /usr/bin/env python3 — the same
    # foreign script resolve_hermes_bin() hands back when argv[0] is unusable,
    # and the very file _needs_interpreter() just rejected one frame earlier.
    wrapper = tmp_path / "bin" / "hermes"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text("#!/usr/bin/env python3\nfrom hermes_cli.main import main\n", encoding="utf-8")
    wrapper.chmod(0o755)
    monkeypatch.setattr("hermes_cli.relaunch.resolve_hermes_bin", lambda: str(wrapper))
    monkeypatch.setattr(lde.shutil, "which", lambda name: str(wrapper) if name == "hermes" else None)
    # sys.executable cannot import hermes_cli (uv/system shim), so rung 1
    # fails; rung 2 must be skipped because the wrapper is a foreign python
    # script, and the entry must fall to `-m hermes_cli.main` rather than
    # silently reverting to the broken wrapper.
    monkeypatch.setattr(
        lde.subprocess, "run",
        lambda *a, **k: _FakeReturn(1) if "import hermes_cli" in str(a) else _orig_run(*a, **k),
    )
    monkeypatch.setattr(lde, "refresh_desktop_databases", lambda _dir: [])

    entry = lde.install_desktop_entry(root)
    exec_line = _parse(entry.read_text(encoding="utf-8"))["Exec"]

    assert exec_line.endswith(f"{os.path.abspath(sys.executable)} -m hermes_cli.main desktop")
    # The foreign wrapper must not appear anywhere in Exec=.
    assert str(wrapper) not in exec_line


def test_rendered_entry_disables_startup_notify(tmp_path, monkeypatch):
    # StartupNotify must be false: the Electron startup token never reaches
    # the desktop environment through the wrapper, so a true value leaves a
    # pinned launcher spinning forever even when the app starts.
    monkeypatch.setattr(lde.sys, "platform", "linux")
    text = lde.render_desktop_entry("/opt/hermes desktop", "/opt/icon.png")
    values = _parse(text)
    assert values["StartupNotify"] == "false"
    assert values["StartupWMClass"] == "Hermes"
