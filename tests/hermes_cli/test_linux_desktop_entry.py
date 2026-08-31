"""Tests for the Linux XDG desktop entry installed by ``hermes desktop``."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

from hermes_cli import linux_desktop_entry as lde


@pytest.fixture
def xdg_home(tmp_path, monkeypatch) -> Path:
    data_home = tmp_path / "xdg-data"
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    # Isolate the known-wrapper probe too: tests must never see the real
    # ~/.local/bin/hermes on the dev machine.
    monkeypatch.setenv("HOME", str(tmp_path))
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


def test_install_writes_entry_with_absolute_exec_and_icon(
    tmp_path, xdg_home, monkeypatch
):
    root = _make_project(tmp_path)
    hermes_bin = tmp_path / "bin" / "hermes"
    hermes_bin.parent.mkdir()
    hermes_bin.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "hermes_cli.relaunch.resolve_hermes_bin", lambda: str(hermes_bin)
    )
    monkeypatch.setattr(lde, "refresh_desktop_databases", lambda _dir: [])
    # Keep the icon install out of the way: this test pins the
    # absolute-path FALLBACK (copy impossible / not attempted here).
    monkeypatch.setattr(lde, "_install_icon_to_hicolor", lambda _icon: False)

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


def test_install_prefers_themed_icon_from_hicolor(tmp_path, xdg_home, monkeypatch):
    """When the icon installs into hicolor, the entry uses the themed name.

    The themed name survives a moved/archived checkout; an absolute
    Icon= path does not (the same durability class the Exec line was
    fixed for).
    """
    root = _make_project(tmp_path)
    hermes_bin = tmp_path / "bin" / "hermes"
    hermes_bin.parent.mkdir()
    hermes_bin.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "hermes_cli.relaunch.resolve_hermes_bin", lambda: str(hermes_bin)
    )
    monkeypatch.setattr(lde, "refresh_desktop_databases", lambda _dir: [])

    entry = lde.install_desktop_entry(root)

    values = _parse(entry.read_text(encoding="utf-8"))
    assert values["Icon"] == "hermes"

    # And the icon really landed in the hicolor tree: the fixture icon is
    # a fake PNG (no valid IHDR), so the size is unknown and the icon
    # lands under scalable/.
    dest = xdg_home / "icons" / "hicolor" / "scalable" / "apps" / "hermes.png"
    assert dest.is_file()
    assert dest.read_bytes() == lde.icon_path(root).read_bytes()


def test_install_icon_copy_failure_falls_back_to_absolute(
    tmp_path, xdg_home, monkeypatch
):
    """An impossible icon copy keeps the absolute path (never breaks)."""
    root = _make_project(tmp_path)
    hermes_bin = tmp_path / "bin" / "hermes"
    hermes_bin.parent.mkdir()
    hermes_bin.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "hermes_cli.relaunch.resolve_hermes_bin", lambda: str(hermes_bin)
    )
    monkeypatch.setattr(lde, "refresh_desktop_databases", lambda _dir: [])

    def _boom(src, dst):
        raise OSError("read-only tree")

    monkeypatch.setattr(lde.shutil, "copyfile", _boom)

    entry = lde.install_desktop_entry(root)
    values = _parse(entry.read_text(encoding="utf-8"))
    # The real helper catches the copy OSError and returns False, so the
    # caller falls back to the absolute path without raising.
    assert values["Icon"] == str(lde.icon_path(root))

    assert values["Type"] == "Application"
    assert values["Name"] == "Hermes"
    assert values["Terminal"] == "false"


def test_installed_entry_is_executable(tmp_path, xdg_home, monkeypatch):
    root = _make_project(tmp_path)
    monkeypatch.setattr(
        "hermes_cli.relaunch.resolve_hermes_bin", lambda: "/usr/bin/hermes"
    )
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
def test_exec_prefixes_interpreter_for_env_shebang_python_script(
    tmp_path, xdg_home, monkeypatch
):
    import sys

    root = _make_project(tmp_path)
    hermes_bin = tmp_path / "bin" / "hermes"
    hermes_bin.parent.mkdir()
    hermes_bin.write_text(
        "#!/usr/bin/env python3\nimport hermes_cli\n", encoding="utf-8"
    )
    hermes_bin.chmod(0o755)
    monkeypatch.setattr(
        "hermes_cli.relaunch.resolve_hermes_bin", lambda: str(hermes_bin)
    )
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
    hermes_bin.write_text(
        '#!/bin/bash\nexec /opt/hermes/venv/bin/python "$@"\n', encoding="utf-8"
    )
    hermes_bin.chmod(0o755)
    monkeypatch.setattr(
        "hermes_cli.relaunch.resolve_hermes_bin", lambda: str(hermes_bin)
    )
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
    interpreter = os.path.abspath(sys.executable)
    hermes_bin.write_text(f"#!{interpreter}\nimport hermes_cli\n", encoding="utf-8")
    hermes_bin.chmod(0o755)
    monkeypatch.setattr(
        "hermes_cli.relaunch.resolve_hermes_bin", lambda: str(hermes_bin)
    )
    monkeypatch.setattr(lde, "refresh_desktop_databases", lambda _dir: [])

    entry = lde.install_desktop_entry(root)
    exec_line = _parse(entry.read_text(encoding="utf-8"))["Exec"]

    # Console-script with the venv's own interpreter in the shebang: correct
    # as-is, prefixing would only add noise.
    assert exec_line == f"{hermes_bin} desktop"


# The persisted entry must be launch-context independent: whatever process
# writes it, the next launch reads and rewrites the same bytes. argv[0]
# differs per launch path (wrapper / repo script / python -m), so a
# checkout-internal argv[0] must not be persisted — the resolver falls
# through to PATH, where the installer's durable wrapper lives.
def _argv0_context(monkeypatch, argv0: str) -> None:
    import sys

    monkeypatch.setattr(sys, "argv", [argv0, "desktop"])


def test_exec_converges_from_repo_script_argv0_to_installed_wrapper(
    tmp_path, xdg_home, monkeypatch
):
    """A broken interpreter-form entry must self-heal to the wrapper form.

    Launching with argv[0] = <checkout>/hermes (what the broken entry
    itself spawns) previously re-persisted the same broken form forever —
    the bootstrap loop that kept #90492 from repairing existing installs.
    """
    import sys

    root = _make_project(tmp_path)
    repo_script = root / "hermes"  # checkout-internal launcher candidate
    repo_script.write_text(
        "#!/usr/bin/env python3\nimport hermes_cli\n", encoding="utf-8"
    )
    repo_script.chmod(0o755)
    wrapper = tmp_path / "installed" / "bin" / "hermes"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(f'#!/bin/bash\nexec {sys.executable} "$@"\n', encoding="utf-8")
    wrapper.chmod(0o755)

    # argv[0] = repo script; PATH lookup finds the installed wrapper.
    _argv0_context(monkeypatch, str(repo_script))
    monkeypatch.setattr(
        "shutil.which", lambda name: str(wrapper) if name == "hermes" else None
    )
    monkeypatch.setattr(lde, "refresh_desktop_databases", lambda _dir: [])

    entry = lde.install_desktop_entry(root)
    exec_line = _parse(entry.read_text(encoding="utf-8"))["Exec"]

    # Converged on the durable wrapper — NOT the repo script, and NOT an
    # interpreter-prefixed form pinning sys.executable.
    assert exec_line == f"{wrapper} desktop"


def test_exec_never_persists_a_bare_interpreter_command(
    tmp_path, xdg_home, monkeypatch
):
    """The `python -m hermes_cli.main` relaunch context must not write
    `Exec=<python> desktop` — a command line no DE can run."""
    import sys

    root = _make_project(tmp_path)
    wrapper = tmp_path / "installed" / "bin" / "hermes"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    wrapper.chmod(0o755)

    interpreter = tmp_path / "uv" / "cpython-3.11.15" / "bin" / "python3.11"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b"\x7fELF fake")
    interpreter.chmod(0o755)

    # argv[0] IS the interpreter (python -m context); PATH has the wrapper.
    _argv0_context(monkeypatch, str(interpreter))
    monkeypatch.setattr(
        "shutil.which", lambda name: str(wrapper) if name == "hermes" else None
    )
    monkeypatch.setattr(lde, "refresh_desktop_databases", lambda _dir: [])

    entry = lde.install_desktop_entry(root)
    exec_line = _parse(entry.read_text(encoding="utf-8"))["Exec"]

    first_token = exec_line.split(" ")[0].strip('"')
    assert Path(first_token) != interpreter
    assert not (
        Path(first_token).name.startswith("python")
        and "desktop" in exec_line.split(" ", 1)[1]
    ), f"persisted an unrunnable bare-interpreter Exec: {exec_line}"
    assert exec_line == f"{wrapper} desktop"


def test_exec_keeps_resolver_fallback_when_no_wrapper_on_path(
    tmp_path, xdg_home, monkeypatch
):
    """No wrapper anywhere → #90492's runnable fallback, never a dead Exec.

    With argv[0] checkout-internal and PATH + known locations both empty,
    the resolver returns None and resolve_exec_command emits the runnable
    `sys.executable -m hermes_cli.main desktop` fallback. Persisting the
    interpreter itself (`<python> desktop`) would be unrunnable by any DE;
    persisting the repo script alone dies on its env shebang.
    """
    import sys

    root = _make_project(tmp_path)
    repo_script = root / "hermes"
    repo_script.write_text(
        "#!/usr/bin/env python3\nimport hermes_cli\n", encoding="utf-8"
    )
    repo_script.chmod(0o755)

    _argv0_context(monkeypatch, str(repo_script))
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr(lde, "refresh_desktop_databases", lambda _dir: [])

    def fake_resolve():
        # Mirror resolve_hermes_bin's chain: argv[0] → relative → PATH → None.
        return sys.argv[0] if sys.argv[0] else None

    monkeypatch.setattr("hermes_cli.relaunch.resolve_hermes_bin", fake_resolve)

    entry = lde.install_desktop_entry(root)
    exec_line = _parse(entry.read_text(encoding="utf-8"))["Exec"]

    # The runnable module fallback — NOT the bare repo script (its env
    # shebang would escape the venv under a DE) and NOT `<python> desktop`.
    assert exec_line.endswith("-m hermes_cli.main desktop")
    assert Path(exec_line.split(" ")[0].strip('"')).is_absolute()
    assert str(repo_script) not in exec_line


def test_exec_uses_known_wrapper_when_path_lookup_misses(
    tmp_path, xdg_home, monkeypatch
):
    """Stripped-PATH session + wrapper at the known installer location.

    systemd user sessions and autostart relaunches often run without
    ~/.local/bin on PATH. When shutil.which finds nothing, the resolver
    must probe the known durable locations directly instead of silently
    persisting a checkout-internal Exec line.
    """
    import sys

    root = _make_project(tmp_path)
    repo_script = root / "hermes"
    repo_script.write_text(
        "#!/usr/bin/env python3\nimport hermes_cli\n", encoding="utf-8"
    )
    repo_script.chmod(0o755)

    # The wrapper exists at the known location but is NOT on PATH.
    # Realistic installer shim: execs this checkout's venv python on the
    # checkout's hermes script (the aidiyet check requires it to target
    # the writing checkout).
    known_wrapper = tmp_path / "known-home" / ".local" / "bin" / "hermes"
    known_wrapper.parent.mkdir(parents=True)
    known_wrapper.write_text(
        f'#!/bin/bash\nexec {root / "venv" / "bin" / "python"} {root / "hermes"} "$@"\n',
        encoding="utf-8",
    )
    known_wrapper.chmod(0o755)
    monkeypatch.setenv("HOME", str(tmp_path / "known-home"))

    _argv0_context(monkeypatch, str(repo_script))
    monkeypatch.setattr("shutil.which", lambda name: None)

    def fake_resolve():
        return sys.argv[0] if sys.argv[0] else None

    monkeypatch.setattr("hermes_cli.relaunch.resolve_hermes_bin", fake_resolve)
    monkeypatch.setattr(lde, "refresh_desktop_databases", lambda _dir: [])

    entry = lde.install_desktop_entry(root)
    exec_line = _parse(entry.read_text(encoding="utf-8"))["Exec"]

    # The probe found the wrapper despite the PATH miss.
    assert exec_line == f"{known_wrapper} desktop"


def test_exec_rejects_known_wrapper_from_another_checkout(
    tmp_path, xdg_home, monkeypatch
):
    """A known-location wrapper that targets a DIFFERENT checkout is skipped.

    On machines with multiple installs over time, ~/.local/bin/hermes may
    belong to another checkout. Persisting it would make the entry stable
    but silently point at that other installation — the failure class the
    aidiyet check exists to prevent. The runnable module fallback must win
    instead.
    """
    import sys

    root = _make_project(tmp_path)
    repo_script = root / "hermes"
    repo_script.write_text(
        "#!/usr/bin/env python3\nimport hermes_cli\n", encoding="utf-8"
    )
    repo_script.chmod(0o755)

    # A shim belonging to a DIFFERENT checkout.
    other_root = tmp_path / "other-install"
    other_root.mkdir()
    foreign_wrapper = tmp_path / "known-home" / ".local" / "bin" / "hermes"
    foreign_wrapper.parent.mkdir(parents=True)
    foreign_wrapper.write_text(
        f"#!/bin/bash\nexec {other_root / 'venv' / 'bin' / 'python'} "
        f'{other_root / "hermes"} "$@"\n',
        encoding="utf-8",
    )
    foreign_wrapper.chmod(0o755)
    monkeypatch.setenv("HOME", str(tmp_path / "known-home"))

    _argv0_context(monkeypatch, str(repo_script))
    monkeypatch.setattr("shutil.which", lambda name: None)

    def fake_resolve():
        return sys.argv[0] if sys.argv[0] else None

    monkeypatch.setattr("hermes_cli.relaunch.resolve_hermes_bin", fake_resolve)
    monkeypatch.setattr(lde, "refresh_desktop_databases", lambda _dir: [])

    entry = lde.install_desktop_entry(root)
    exec_line = _parse(entry.read_text(encoding="utf-8"))["Exec"]

    # The foreign wrapper was rejected; the runnable module fallback won.
    assert str(foreign_wrapper) not in exec_line
    assert exec_line.endswith("-m hermes_cli.main desktop")


@pytest.mark.parametrize(
    ("layout", "env_overrides", "expected"),
    [
        pytest.param(
            "user",
            {},
            "HOME-SET-BY-TEST/.local/bin/hermes",
            id="user-layout",
        ),
        pytest.param(
            "termux",
            {"PREFIX": "PREFIX-SET-BY-TEST"},
            "PREFIX-SET-BY-TEST/bin/hermes",
            id="termux-prefix-first",
        ),
        pytest.param(
            "root-fhs",
            {"__EUID0__": "1"},
            "/usr/local/bin/hermes",
            id="root-fhs",
        ),
        pytest.param(
            "non-root-no-fhs",
            {"__EUID0__": "0"},
            "HOME-SET-BY-TEST/.local/bin/hermes",
            id="non-root-excludes-fhs",
        ),
    ],
)
def test_known_wrapper_candidates_cover_installer_layouts(
    layout, env_overrides, expected, monkeypatch
):
    """_known_wrapper_candidates mirrors get_command_link_dir() layouts.

    Termux ($PREFIX/bin) outranks everything; root FHS (/usr/local/bin)
    applies only to euid 0; the user layout (~/.local/bin) is always a
    candidate. Locking these in protects against silent regressions in
    the stripped-PATH probe path.
    """
    import os

    sentinel_home = "/home/__sentinel_home__"
    monkeypatch.setenv("HOME", sentinel_home)
    for key, value in env_overrides.items():
        if key == "__EUID0__":
            monkeypatch.setattr(lde.os, "geteuid", lambda: 0 if value == "1" else 1000)
        else:
            monkeypatch.setenv(key, value)

    candidates = [str(c) for c in lde._known_wrapper_candidates()]

    expected_resolved = expected.replace("HOME-SET-BY-TEST", sentinel_home)
    assert expected_resolved in candidates
    if layout == "termux":
        # PREFIX outranks the user layout.
        assert candidates[0] == expected_resolved
    if layout == "root-fhs":
        # Root FHS outranks the user layout.
        assert candidates.index("/usr/local/bin/hermes") < candidates.index(
            f"{sentinel_home}/.local/bin/hermes"
        )
    if layout == "non-root-no-fhs":
        # Non-root euid: /usr/local/bin must be excluded outright.
        assert "/usr/local/bin/hermes" not in candidates


def test_install_is_idempotent_and_skips_cache_refresh(tmp_path, xdg_home, monkeypatch):
    root = _make_project(tmp_path)
    monkeypatch.setattr(
        "hermes_cli.relaunch.resolve_hermes_bin", lambda: "/usr/bin/hermes"
    )
    calls: list[Path] = []
    monkeypatch.setattr(
        lde, "refresh_desktop_databases", lambda d: calls.append(d) or []
    )

    lde.install_desktop_entry(root)
    assert len(calls) == 1

    # Unchanged content → no rewrite, no menu-cache churn on every launch.
    lde.install_desktop_entry(root)
    assert len(calls) == 1


def test_install_without_source_icon_uses_themed_name(tmp_path, xdg_home, monkeypatch):
    root = tmp_path / "hermes-agent"
    root.mkdir()
    monkeypatch.setattr(
        "hermes_cli.relaunch.resolve_hermes_bin", lambda: "/usr/bin/hermes"
    )
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
        lde.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in available else None,
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


@pytest.mark.skipif(
    sys.platform == "win32", reason="Symlinks require elevated privileges on Windows"
)
def test_running_interpreter_keeps_venv_semantic_path(tmp_path, monkeypatch):
    """Lexical preserved only when pyvenv.cfg marks the path as a venv."""
    # venv layout: bin/python symlink -> base, pyvenv.cfg at venv root
    base = tmp_path / "base" / "python3.11"
    base.parent.mkdir(parents=True)
    base.write_text("", encoding="utf-8")
    venv_root = tmp_path / "venv"
    venv_bin = venv_root / "bin"
    venv_bin.mkdir(parents=True)
    (venv_root / "pyvenv.cfg").write_text("home = /base\n", encoding="utf-8")
    venv_python = venv_bin / "python"
    venv_python.symlink_to(base)

    monkeypatch.setattr(lde.sys, "executable", str(venv_python))
    assert lde._running_interpreter() == str(venv_python)

    # non-venv symlink: resolve instead (durability over lexical)
    plain_root = tmp_path / "plain" / "bin"
    plain_root.mkdir(parents=True)
    plain_link = plain_root / "python3"
    plain_link.symlink_to(base)
    monkeypatch.setattr(lde.sys, "executable", str(plain_link))
    assert lde._running_interpreter() == str(base)


def test_running_interpreter_resolves_plain_interpreter(monkeypatch):
    """A non-symlinked, non-venv executable resolves to itself."""
    monkeypatch.setattr(lde.sys, "executable", "/usr/bin/python3")
    out = lde._running_interpreter()
    assert Path(out).is_absolute()


def test_can_import_probe_runs_and_caches(tmp_path):
    """The probe executes the real interpreter and memoizes the answer.

    Asserts the two things that hold on ANY host: the probe returns a
    definite boolean for a real interpreter (not None, not an exception
    path), and the per-path cache is populated so the second call pays
    no subprocess. Host-dependent capability itself (True vs False) is
    deliberately NOT asserted - a CI host with hermes pip-installed
    system-wide would legitimately answer True.
    """
    import time

    real = Path("/usr/bin/python3")
    if not real.exists():
        pytest.skip("no system python to probe")
    lde._probe_cache.pop(str(real), None)
    try:
        first = lde._can_import_hermes_cli(real)
        assert isinstance(first, bool)
        assert str(real) in lde._probe_cache
        t0 = time.monotonic()
        second = lde._can_import_hermes_cli(real)
        assert second is first
        assert time.monotonic() - t0 < 0.05  # cache hit: no subprocess
    finally:
        lde._probe_cache.pop(str(real), None)


def test_exec_falls_back_to_running_interpreter_when_probe_fails(
    tmp_path, xdg_home, monkeypatch
):
    """A candidate interpreter that fails the import probe is not persisted."""
    import sys as _s

    root = _make_project(tmp_path)
    interpreter = tmp_path / "uv" / "bin" / "python3.11"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b"\x7fELF fake")
    interpreter.chmod(0o755)

    _argv0_context(monkeypatch, str(interpreter))
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr(lde, "refresh_desktop_databases", lambda _dir: [])

    def fake_resolve():
        return _s.argv[0] if _s.argv[0] else None

    monkeypatch.setattr("hermes_cli.relaunch.resolve_hermes_bin", fake_resolve)
    # Force the probe to fail for whatever interpreter gets chosen first.
    monkeypatch.setattr(lde, "_can_import_hermes_cli", lambda p: False)
    # And the fallback interpreter must itself pass (it always should).
    monkeypatch.setattr(
        lde,
        "_running_interpreter_fallback",
        lambda: os.path.abspath(sys.executable),
    )

    entry = lde.install_desktop_entry(root)
    exec_line = _parse(entry.read_text(encoding="utf-8"))["Exec"]

    # Runnable module form under the RUNNING interpreter - never the
    # unprobeable ELF fake, never a bare "<python> desktop".
    assert exec_line.endswith("-m hermes_cli.main desktop")
    first = exec_line.split(" ")[0].strip('"')
    assert first == os.path.abspath(sys.executable)
    assert str(interpreter) not in exec_line


@pytest.mark.parametrize(
    "suffix",
    ["-old", ".bak", "-copy"],
)
def test_wrapper_ownership_rejects_sibling_extensions(suffix, tmp_path):
    """A shim execing `<checkout><suffix>/...` must NOT pass ownership.

    Bare substring matching accepted these; the boundary-aware matcher
    must reject them (stable-but-wrong entry pointing at the renamed
    old install).
    """
    checkout = tmp_path / "hermes-agent"
    checkout.mkdir()
    evil = tmp_path / "evil-shim"
    evil.write_text(
        f"#!/bin/bash\n"
        f"exec {checkout}{suffix}/venv/bin/python "
        f'{checkout}{suffix}/hermes "$@"\n',
        encoding="utf-8",
    )
    assert lde._wrapper_targets_checkout(evil, checkout) is False


@pytest.mark.skipif(
    sys.platform == "win32", reason="Symlinks require elevated privileges on Windows"
)
def test_wrapper_ownership_accepts_shim_via_symlinked_home(tmp_path, monkeypatch):
    """Installer writes $INSTALL_DIR lexically; the root stays lexical too.

    With /home -> /real-home, a shim that references the lexical
    checkout path must match the lexical checkout root (the resolved
    root alone would never match the shim's text).
    """
    home_link = tmp_path / "home-link"
    home_real = tmp_path / "home-real"
    home_real.mkdir()
    home_link.symlink_to(home_real)
    lexical_checkout = home_link / "hermes-agent"
    (home_real / "hermes-agent").mkdir()

    shim = home_link / ".local" / "bin" / "hermes"
    shim.parent.mkdir(parents=True)
    shim.write_text(
        f"#!/bin/bash\n"
        f"exec {lexical_checkout}/venv/bin/python "
        f'{lexical_checkout}/hermes "$@"\n',
        encoding="utf-8",
    )
    assert lde._wrapper_targets_checkout(shim, lexical_checkout) is True
    # And the end-to-end probe finds it via the lexical root: make the
    # shim executable, point HOME at the symlinked home, and give the
    # resolver a checkout-internal primary (the repo script) so the
    # probe leg actually engages.
    shim.chmod(0o755)
    monkeypatch.setenv("HOME", str(home_link))
    monkeypatch.setattr("shutil.which", lambda name: None)
    repo_script = lexical_checkout / "hermes"
    repo_script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [str(repo_script), "desktop"])
    assert lde._resolve_hermes_bin_for_desktop_entry(
        resolve_fn=lambda: sys.argv[0] if sys.argv[0] else None,
        checkout_root=lexical_checkout,
    ) == str(shim)


def test_needs_interpreter_case_insensitive_match(tmp_path, monkeypatch):
    """Interpreter paths with uppercase must not flag own venv scripts.

    The shebang is lowercased for comparison; the interpreter dir must
    be too (conda env names, usernames, uv's ephemeral .tmpXXX dirs all
    carry uppercase - an asymmetric compare would prefix the venv's own
    console script spuriously).
    """
    venv_bin = tmp_path / "MyEnv" / "bin"
    venv_bin.mkdir(parents=True)
    interpreter = venv_bin / "python"
    interpreter.write_text("", encoding="utf-8")

    console_script = venv_bin / "hermes"
    console_script.write_text(f"#!{interpreter}\nimport hermes_cli\n", encoding="utf-8")
    monkeypatch.setattr(lde.sys, "executable", str(interpreter))

    assert lde._needs_interpreter(console_script) is False


def test_needs_interpreter_rejects_sibling_directory(tmp_path, monkeypatch):
    """``<venv>/bin-extra/python`` is NOT inside ``<venv>/bin``.

    Substring matching accepted it (the parent dir appears verbatim inside
    the sibling path), skipping the interpreter prefix for a script whose
    shebang actually points OUTSIDE the venv. Path-component comparison
    rejects it.
    """
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    interp = venv_bin / "python"
    interp.write_text("", encoding="utf-8")
    monkeypatch.setattr(lde.sys, "executable", str(interp))

    sibling_script = tmp_path / "sibling"
    sibling_script.write_text(
        f"#!{tmp_path}/venv/bin-extra/python\nimport hermes_cli\n",
        encoding="utf-8",
    )
    assert lde._needs_interpreter(sibling_script) is True


def test_needs_interpreter_strips_flags_before_comparing(tmp_path, monkeypatch):
    """A flagged own-venv shebang is not misclassified by the flag token."""
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    interp = venv_bin / "python"
    interp.write_text("", encoding="utf-8")
    monkeypatch.setattr(lde.sys, "executable", str(interp))

    flagged = tmp_path / "flagged"
    flagged.write_text(f"#!{interp} -S\nimport hermes_cli\n", encoding="utf-8")
    assert lde._needs_interpreter(flagged) is False


def test_needs_interpreter_env_shebang_always_escapes(tmp_path, monkeypatch):
    """``env`` resolves through the DE's PATH - not the installer's."""
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    interp = venv_bin / "python"
    interp.write_text("", encoding="utf-8")
    monkeypatch.setattr(lde.sys, "executable", str(interp))

    # Even when env itself sits in the venv bin (parent equality would
    # pass), the PATH resolution semantics mean the shebang escapes.
    env_script = tmp_path / "envscript"
    env_script.write_text(
        f"#!{venv_bin}/env python3\nimport hermes_cli\n", encoding="utf-8"
    )
    assert lde._needs_interpreter(env_script) is True

    # ...unless env carries an absolute venv interpreter after -S.
    env_abs = tmp_path / "envabs"
    env_abs.write_text(
        f"#!/usr/bin/env -S {interp}\nimport hermes_cli\n", encoding="utf-8"
    )
    assert lde._needs_interpreter(env_abs) is False


def test_probe_skips_wrapper_with_escaping_python_shebang(
    tmp_path, xdg_home, monkeypatch
):
    """A checkout-referencing wrapper with an env shebang is skipped.

    Ownership alone would accept it (the body references this checkout),
    but its `#!/usr/bin/env python3` shebang dies in the DE context.
    The shebang-safety gate skips it; the module fallback wins. Idea
    credited to autumn8's #92122 rung-2 check.
    """
    import sys as _s

    root = _make_project(tmp_path)
    repo_script = root / "hermes"
    repo_script.write_text(
        "#!/usr/bin/env python3\nimport hermes_cli\n", encoding="utf-8"
    )
    repo_script.chmod(0o755)

    # A wrapper that targets this checkout but cannot run itself.
    broken_wrapper = xdg_home / ".local" / "bin" / "hermes"
    broken_wrapper.parent.mkdir(parents=True)
    broken_wrapper.write_text(
        f"#!/usr/bin/env python3\n# launcher for {root}\nimport hermes_cli\n",
        encoding="utf-8",
    )
    broken_wrapper.chmod(0o755)
    monkeypatch.setenv("HOME", str(xdg_home))
    _argv0_context(monkeypatch, str(repo_script))
    monkeypatch.setattr("shutil.which", lambda name: None)

    def fake_resolve():
        return sys.argv[0] if sys.argv[0] else None

    monkeypatch.setattr("hermes_cli.relaunch.resolve_hermes_bin", fake_resolve)
    monkeypatch.setattr(lde, "refresh_desktop_databases", lambda _dir: [])

    entry = lde.install_desktop_entry(root)
    exec_line = _parse(entry.read_text(encoding="utf-8"))["Exec"]

    assert str(broken_wrapper) not in exec_line
    assert exec_line.endswith("-m hermes_cli.main desktop")


def test_probe_accepts_shell_launcher_wrapper(tmp_path, xdg_home, monkeypatch):
    """A bash launcher is safe by construction and still wins the probe."""
    root = _make_project(tmp_path)
    repo_script = root / "hermes"
    repo_script.write_text(
        "#!/usr/bin/env python3\nimport hermes_cli\n", encoding="utf-8"
    )
    repo_script.chmod(0o755)

    good_wrapper = xdg_home / ".local" / "bin" / "hermes"
    good_wrapper.parent.mkdir(parents=True)
    good_wrapper.write_text(
        f"#!/bin/bash\nexec {root / 'venv' / 'bin' / 'python'} "
        f'{root / "hermes"} "$@"\n',
        encoding="utf-8",
    )
    good_wrapper.chmod(0o755)
    monkeypatch.setenv("HOME", str(xdg_home))
    _argv0_context(monkeypatch, str(repo_script))
    monkeypatch.setattr("shutil.which", lambda name: None)

    def fake_resolve():
        return sys.argv[0] if sys.argv[0] else None

    monkeypatch.setattr("hermes_cli.relaunch.resolve_hermes_bin", fake_resolve)
    monkeypatch.setattr(lde, "refresh_desktop_databases", lambda _dir: [])

    entry = lde.install_desktop_entry(root)
    exec_line = _parse(entry.read_text(encoding="utf-8"))["Exec"]
    assert exec_line == f"{good_wrapper} desktop"


def test_install_icon_handles_truncated_png_header(tmp_path, xdg_home, monkeypatch):
    """A truncated PNG (valid signature + IHDR tag, <24 bytes) must not
    raise struct.error out of the fail-safe: it lands in scalable/ like
    any other unknown-size image."""
    root = _make_project(tmp_path)
    icon = lde.icon_path(root)
    icon.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR\x00\x00"  # 22 bytes
    )
    hermes_bin = tmp_path / "bin" / "hermes"
    hermes_bin.parent.mkdir()
    hermes_bin.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "hermes_cli.relaunch.resolve_hermes_bin", lambda: str(hermes_bin)
    )
    monkeypatch.setattr(lde, "refresh_desktop_databases", lambda _dir: [])

    entry = lde.install_desktop_entry(root)

    values = _parse(entry.read_text(encoding="utf-8"))
    assert values["Icon"] == "hermes"
    dest = xdg_home / "icons" / "hicolor" / "scalable" / "apps" / "hermes.png"
    assert dest.is_file()
