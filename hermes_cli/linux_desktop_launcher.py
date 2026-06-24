"""Install and run the Linux Hermes Desktop launcher (.desktop + wrapper script).

Unlike Windows shortcuts that can point directly at the packaged ``Hermes.exe``,
Linux Electron builds require a one-time ``chrome-sandbox`` setuid setup after
every unpack/rebuild.  The wrapper ensures that invariant before ``exec``-ing
the packaged binary — the correct Linux analogue of a Start Menu shortcut.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

from hermes_constants import get_hermes_home

from hermes_cli.desktop_linux import ensure_electron_sandbox_fixup

LAUNCHER_NAME = "hermes-desktop-launch"
DESKTOP_FILENAME = "hermes-desktop.desktop"
STARTUP_WM_CLASS = "Hermes"


def resolve_packaged_executable(agent_root: Path) -> Path | None:
    """Return the newest packaged Linux desktop executable under *agent_root*."""
    release_dir = agent_root / "apps" / "desktop" / "release"
    candidates = [
        release_dir / "linux-unpacked" / "Hermes",
        release_dir / "linux-unpacked" / "hermes",
        release_dir / "linux-arm64-unpacked" / "Hermes",
        release_dir / "linux-arm64-unpacked" / "hermes",
    ]
    existing = [p for p in candidates if p.is_file()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


def _agent_root_from_env() -> Path:
    hermes_home = Path(os.environ.get("HERMES_HOME", get_hermes_home())).expanduser()
    return Path(os.environ.get("HERMES_AGENT", hermes_home / "hermes-agent")).expanduser().resolve()


def _local_bin_dir() -> Path:
    return Path.home() / ".local" / "bin"


def _applications_dir() -> Path:
    xdg_data = os.environ.get("XDG_DATA_HOME")
    data_base = Path(xdg_data).expanduser() if xdg_data else (Path.home() / ".local" / "share")
    return data_base / "applications"


def _desktop_icon_path(agent_root: Path) -> Path:
    return agent_root / "apps" / "desktop" / "assets" / "icon.png"


def render_launcher_script(agent_root: Path) -> str:
    agent_root = agent_root.resolve()
    return f"""#!/usr/bin/env bash
set -euo pipefail
export HERMES_HOME="${{HERMES_HOME:-$HOME/.hermes}}"
export HERMES_AGENT="${{HERMES_AGENT:-$HERMES_HOME/hermes-agent}}"
export PATH="$HERMES_AGENT/venv/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
exec "$HERMES_AGENT/venv/bin/python" -m hermes_cli.linux_desktop_launcher launch "$@"
"""


def render_desktop_entry(*, launcher_path: Path, icon_path: Path) -> str:
    return (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=Hermes\n"
        "Comment=Hermes Agent desktop app\n"
        f"Exec={launcher_path}\n"
        f"Icon={icon_path}\n"
        "Terminal=false\n"
        "Categories=Development;\n"
        f"StartupWMClass={STARTUP_WM_CLASS}\n"
    )


def install_linux_desktop_shortcuts(agent_root: Path) -> list[Path]:
    """Install the launcher script and .desktop entry. Best-effort; returns paths written."""
    if sys.platform != "linux":
        return []

    agent_root = agent_root.resolve()
    icon_path = _desktop_icon_path(agent_root)
    if not icon_path.is_file():
        raise FileNotFoundError(f"Desktop icon not found: {icon_path}")

    launcher_dir = _local_bin_dir()
    launcher_dir.mkdir(parents=True, exist_ok=True)
    launcher_path = launcher_dir / LAUNCHER_NAME
    launcher_path.write_text(render_launcher_script(agent_root), encoding="utf-8")
    launcher_path.chmod(launcher_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    apps_dir = _applications_dir()
    apps_dir.mkdir(parents=True, exist_ok=True)
    desktop_path = apps_dir / DESKTOP_FILENAME
    desktop_path.write_text(
        render_desktop_entry(launcher_path=launcher_path.resolve(), icon_path=icon_path.resolve()),
        encoding="utf-8",
    )
    return [launcher_path, desktop_path]


def linux_desktop_shortcut_paths() -> list[Path]:
    """Standard install locations for the Linux desktop launcher artifacts."""
    return [
        _local_bin_dir() / LAUNCHER_NAME,
        _applications_dir() / DESKTOP_FILENAME,
        _applications_dir() / "hermes.desktop",
        _applications_dir() / "Hermes.desktop",
    ]


def finalize_packaged_linux_desktop(agent_root: Path, packaged_executable: Path) -> bool:
    """Post-build invariant: sandbox helper configured and launcher installed."""
    if sys.platform != "linux":
        return True

    if not ensure_electron_sandbox_fixup(packaged_executable):
        return False

    try:
        written = install_linux_desktop_shortcuts(agent_root)
    except OSError as exc:
        print(f"  (warning: could not install Linux desktop launcher: {exc})")
        return True

    for path in written:
        print(f"✓ Linux desktop launcher installed: {path}")
    return True


def launch_packaged_desktop(extra_args: list[str]) -> None:
    """Ensure sandbox is configured, then replace this process with Hermes Desktop."""
    agent_root = _agent_root_from_env()
    packaged_executable = resolve_packaged_executable(agent_root)
    if packaged_executable is None:
        print(
            f"✗ No packaged Hermes Desktop executable found under {agent_root / 'apps/desktop/release'}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if not ensure_electron_sandbox_fixup(packaged_executable):
        raise SystemExit(1)

    argv = [str(packaged_executable), *extra_args]
    os.execve(str(packaged_executable), argv, os.environ)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] == "launch":
        launch_args = args[1:] if args and args[0] == "launch" else args
        launch_packaged_desktop(launch_args)
        return 1

    if args[0] == "finalize-build":
        if len(args) != 2:
            print("usage: python -m hermes_cli.linux_desktop_launcher finalize-build <agent-root>", file=sys.stderr)
            return 2
        agent_root = Path(args[1]).expanduser().resolve()
        packaged_executable = resolve_packaged_executable(agent_root)
        if packaged_executable is None:
            print(f"✗ No packaged Linux desktop executable found under {agent_root}", file=sys.stderr)
            return 1
        return 0 if finalize_packaged_linux_desktop(agent_root, packaged_executable) else 1

    print(
        "usage: python -m hermes_cli.linux_desktop_launcher [launch [args...] | finalize-build <agent-root>]",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
