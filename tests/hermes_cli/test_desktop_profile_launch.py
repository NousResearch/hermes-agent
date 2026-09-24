"""hermes desktop forwards an explicit profile into the packaged Electron launch.

Electron boots from active-profile.json. ``hermes -p <name> desktop`` used to
set only the CLI HERMES_HOME, so the packaged app still started the stored
profile. A launch with no ``-p`` / ``--profile`` must not invent a flag.
"""

from __future__ import annotations

import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from hermes_cli import main as cli_main
from hermes_cli import main_desktop


def _ns(**kwargs):
    defaults = dict(
        skip_build=True,
        build_only=False,
        force_build=False,
        source=False,
        fake_boot=False,
        ignore_existing=False,
        hermes_root=None,
        cwd=None,
        setup_tcc_identity=False,
        identity=None,
        local=False,
    )
    defaults.update(kwargs)
    return Namespace(**defaults)


def _named_profiles(argv: list[str]) -> set[str]:
    names: set[str] = set()
    for index, arg in enumerate(argv):
        if arg in {"-p", "--profile"} and index + 1 < len(argv):
            names.add(argv[index + 1].strip().casefold())
        elif arg.startswith("--profile="):
            names.add(arg.split("=", 1)[1].strip().casefold())
    names.discard("default")
    return names


def _packaged_exe(root: Path) -> Path:
    desktop_dir = root / "apps" / "desktop"
    if sys.platform == "darwin":
        exe = desktop_dir / "release" / "mac-arm64" / "Hermes.app" / "Contents" / "MacOS" / "Hermes"
    elif sys.platform == "win32":
        exe = desktop_dir / "release" / "win-unpacked" / "Hermes.exe"
    else:
        exe = desktop_dir / "release" / "linux-unpacked" / "hermes"
        exe.parent.mkdir(parents=True, exist_ok=True)
        (exe.parent / "chrome-sandbox").write_text("", encoding="utf-8")
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_text("", encoding="utf-8")
    return exe


def _live_profile(root: Path, name: str) -> None:
    profile = root / "profiles" / name
    profile.mkdir(parents=True, exist_ok=True)
    (profile / "config.yaml").write_text("{}\n", encoding="utf-8")


def _launch_command(tmp_path, monkeypatch, argv: list[str], *, active_profile: str | None = None) -> list[str]:
    hermes_root = tmp_path / ".hermes"
    hermes_root.mkdir()
    if active_profile and active_profile != "default":
        _live_profile(hermes_root, active_profile)
        (hermes_root / "active_profile").write_text(active_profile + "\n", encoding="utf-8")
    for name in _named_profiles(argv):
        _live_profile(hermes_root, name)

    project = tmp_path / "hermes-agent"
    (project / "apps" / "desktop").mkdir(parents=True)
    (project / "apps" / "desktop" / "package.json").write_text("{}", encoding="utf-8")
    exe = _packaged_exe(project)

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("GNOME_KEYRING_CONTROL", "/run/user/1000/keyring")
    for var in (
        "HERMES_SUPERVISED_CHILD",
        "HERMES_S6_SUPERVISED_CHILD",
        "INVOCATION_ID",
        "HERMES_GATEWAY_EXTERNAL_SUPERVISOR",
        "KDE_SESSION_VERSION",
        "KDE_FULL_SESSION",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", project)
    monkeypatch.setattr(main_desktop, "_desktop_exe_integrity_error", lambda _path: None)

    cli_main._apply_profile_override()

    calls: list[list[str]] = []

    def _run(cmd, **_kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", _run)
    with pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns())
    assert exc.value.code == 0
    launches = [cmd for cmd in calls if cmd and cmd[0] == str(exe)]
    assert len(launches) == 1
    return launches[0]


def test_dash_p_desktop_appends_profile_to_packaged_launch(tmp_path, monkeypatch):
    launched = _launch_command(tmp_path, monkeypatch, ["hermes", "-p", "desktop", "desktop"])

    assert launched[-2:] == ["--profile", "desktop"]


def test_profile_equals_spelling_is_forwarded(tmp_path, monkeypatch):
    launched = _launch_command(tmp_path, monkeypatch, ["hermes", "desktop", "--profile=work"])

    assert launched[-2:] == ["--profile", "work"]


def test_missing_flag_does_not_forward_sticky_profile(tmp_path, monkeypatch):
    launched = _launch_command(
        tmp_path, monkeypatch, ["hermes", "desktop"], active_profile="coder"
    )

    assert "--profile" not in launched
    assert os.environ.get("HERMES_HOME", "").endswith(f"{os.sep}profiles{os.sep}coder")
