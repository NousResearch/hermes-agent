"""Linux desktop integration for locally built Hermes Desktop apps.

``electron-builder --dir`` produces a runnable unpacked tree, but unlike its
deb/rpm targets it does not install a ``.desktop`` file.  Electron therefore
has no desktop application id to hand to ``xdg-mime`` when the unpacked binary
calls ``app.setAsDefaultProtocolClient()``.  Register the local build explicitly
so ``hermes://`` links have a stable handler too.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


_DESKTOP_FILE_NAME = "hermes.desktop"
_PROTOCOL_MIME = "x-scheme-handler/hermes"


def _desktop_exec_arg(path: Path) -> str:
    """Quote one executable for the Desktop Entry ``Exec`` grammar."""
    value = str(path)
    if "\n" in value or "\r" in value:
        raise ValueError("Desktop executable path contains a newline")
    # Exec values are not shell commands. Inside a quoted argument the Desktop
    # Entry spec requires these characters to be backslash-escaped; ``%%`` is
    # the literal-percent field code.
    value = value.replace("\\", "\\\\")
    value = value.replace('"', '\\"').replace("`", "\\`").replace("$", "\\$")
    value = value.replace("%", "%%")
    return f'"{value}"'


def _desktop_string(value: Path) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def _desktop_entry(executable: Path, icon: Optional[Path]) -> str:
    lines = [
        "[Desktop Entry]",
        "Type=Application",
        "Name=Hermes",
        "Comment=Native desktop shell for Hermes Agent",
        f"Exec={_desktop_exec_arg(executable)} %U",
    ]
    if icon is not None and icon.is_file():
        lines.append(f"Icon={_desktop_string(icon)}")
    lines.extend(
        [
            "Terminal=false",
            "Categories=Development;",
            "StartupWMClass=Hermes",
            f"MimeType={_PROTOCOL_MIME};",
            "",
        ]
    )
    return "\n".join(lines)


def _applications_dir() -> Path:
    data_home = os.environ.get("XDG_DATA_HOME")
    if data_home:
        candidate = Path(data_home).expanduser()
        # The XDG Base Directory spec requires absolute paths and says relative
        # values must be ignored.
        if candidate.is_absolute():
            return candidate / "applications"
    return Path.home() / ".local" / "share" / "applications"


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )


def register_linux_deep_link_protocol(
    executable: Path, *, icon: Optional[Path] = None
) -> bool:
    """Install and register a user-scoped handler for ``hermes://``.

    This is deliberately best-effort: a missing desktop utility must not make a
    successful Desktop build unusable.  The caller gets ``False`` and the user
    gets one actionable warning instead of Electron silently discarding its
    failed boolean registration result.
    """
    if sys.platform != "linux":
        return True

    executable = executable.expanduser().resolve()
    icon = icon.expanduser().resolve() if icon is not None else None
    applications_dir = _applications_dir()
    desktop_file = applications_dir / _DESKTOP_FILE_NAME

    try:
        if not executable.is_file():
            raise FileNotFoundError(f"Desktop executable not found: {executable}")
        applications_dir.mkdir(parents=True, exist_ok=True)
        content = _desktop_entry(executable, icon)
        entry_changed = (
            not desktop_file.is_file()
            or desktop_file.read_text(encoding="utf-8") != content
        )
        if entry_changed:
            temporary = desktop_file.with_suffix(".desktop.tmp")
            temporary.write_text(content, encoding="utf-8")
            temporary.chmod(0o644)
            temporary.replace(desktop_file)

        update_database = shutil.which("update-desktop-database")
        if update_database and entry_changed:
            try:
                _run([update_database, str(applications_dir)])
            except (OSError, subprocess.SubprocessError):
                # xdg-mime can register a handler without the optional cache
                # refresh utility; keep going and verify the authoritative MIME
                # association below.
                pass

        xdg_mime = shutil.which("xdg-mime")
        if not xdg_mime:
            raise RuntimeError("xdg-mime is not installed")

        verified = _run([xdg_mime, "query", "default", _PROTOCOL_MIME])
        if (
            verified.returncode == 0
            and (verified.stdout or "").strip() == _DESKTOP_FILE_NAME
        ):
            return True

        registered = _run([xdg_mime, "default", _DESKTOP_FILE_NAME, _PROTOCOL_MIME])
        if registered.returncode != 0:
            detail = (registered.stderr or registered.stdout).strip()
            raise RuntimeError(
                detail or f"xdg-mime exited with status {registered.returncode}"
            )

        verified = _run([xdg_mime, "query", "default", _PROTOCOL_MIME])
        if (
            verified.returncode != 0
            or (verified.stdout or "").strip() != _DESKTOP_FILE_NAME
        ):
            raise RuntimeError("xdg-mime did not retain the Hermes protocol handler")
        return True
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(
            "⚠ Could not register hermes:// links for this local Linux Desktop build: "
            f"{exc}\n"
            f"  Desktop entry: {desktop_file}\n"
            f"  Retry: xdg-mime default {_DESKTOP_FILE_NAME} {_PROTOCOL_MIME}"
        )
        return False


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Register a local Hermes Desktop build on Linux"
    )
    parser.add_argument("executable", type=Path)
    parser.add_argument("--icon", type=Path)
    args = parser.parse_args(argv)
    return (
        0 if register_linux_deep_link_protocol(args.executable, icon=args.icon) else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
