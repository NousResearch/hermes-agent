"""Install and remove the Linux desktop entry (``hermes.desktop``).

``hermes desktop`` builds and launches the Electron app. On Linux, a
freshly-built app has no launcher presence: no menu item, no icon. This
module writes the XDG desktop entry that gives it one.
``hermes uninstall --gui`` removes the entry again.

Two values must be absolute for the entry to work:

  - ``Exec`` — the launcher runs without shell ``PATH`` customizations, so
    a bare ``hermes desktop`` fails when hermes lives in ``~/.local/bin``
    or a venv. Resolve the real binary and write its full path.
  - ``Icon`` — an unqualified icon name needs an indexed icon theme. The
    spec allows an absolute path instead, so point at the app icon in the
    checkout. Do not copy the icon: ``Exec`` already depends on that tree.

Cache refresh is best-effort and tool-gated: ``update-desktop-database``
for the freedesktop menu cache, and ``kbuildsycoca6``/``kbuildsycoca5``
for Plasma. Run each tool only when it exists. A missing tool is not an
error.

Import-light and side-effect-free at import time: the uninstaller and the
Electron main process both use this without loading the full CLI.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

DESKTOP_ENTRY_NAME = "hermes.desktop"


def is_supported() -> bool:
    """XDG desktop entries exist only on Linux and BSD."""
    return sys.platform.startswith(("linux", "freebsd", "openbsd", "netbsd"))


def _xdg_data_home() -> Path:
    raw = os.environ.get("XDG_DATA_HOME")
    if raw and raw.strip():
        return Path(raw).expanduser()
    return Path.home() / ".local" / "share"


def desktop_entry_path() -> Path:
    """Where the ``hermes.desktop`` entry lives."""
    return _xdg_data_home() / "applications" / DESKTOP_ENTRY_NAME


def icon_path(project_root: Path) -> Path:
    """The app icon shipped in the desktop workspace."""
    return project_root / "apps" / "desktop" / "assets" / "icon.png"


def resolve_exec_command() -> str:
    """Build the absolute ``Exec=`` command line for ``hermes desktop``.

    Prefer the real ``hermes`` executable (argv[0] or PATH). When Hermes
    runs as a module with no launcher installed, use the current
    interpreter, also absolute.
    """
    from hermes_cli.relaunch import resolve_hermes_bin

    bin_path = resolve_hermes_bin()
    interpreter = _running_interpreter()
    if bin_path:
        resolved = Path(bin_path).resolve()
        if _needs_interpreter(resolved):
            # The resolved launcher is a Python script whose shebang points at
            # a NON-venv interpreter (e.g. the repo's `hermes` script with
            # `#!/usr/bin/env python3` when argv[0] came from the shell
            # installer's bash wrapper). Launched from the .desktop entry that
            # shebang resolves to the SYSTEM python and dies on the first
            # third-party import (#90292) — silently, since Terminal=false.
            # Prefix the interpreter that is actually running Hermes (the
            # venv one). Do not follow a venv/bin/python → /usr/bin/python
            # symlink or the prefix is the system interpreter again.
            argv = [str(interpreter), str(resolved), "desktop"]
        else:
            argv = [str(resolved), "desktop"]
    else:
        argv = [str(interpreter), "-m", "hermes_cli.main", "desktop"]
    return " ".join(_quote_exec_arg(a) for a in argv)


def _running_interpreter() -> Path:
    """Interpreter that can see Hermes' site-packages.

    Do not ``Path.resolve()`` this. A venv's ``bin/python`` is often a
    symlink to the system interpreter (``/usr/bin/python3.11``). Following
    that symlink writes an ``Exec=`` that imports from the system
    site-packages and dies with ``ModuleNotFoundError: hermes_cli`` —
    silent under ``Terminal=false``.
    """
    return Path(sys.executable)


def _needs_interpreter(bin_path: Path) -> bool:
    """Whether ``bin_path`` is a Python script that must run under
    ``sys.executable`` to see Hermes' venv (rather than its own shebang)."""
    try:
        with open(bin_path, "rb") as fh:
            head = fh.readline(256)
    except OSError:
        return False
    if not head.startswith(b"#!"):
        # Native binary (uv tool shim, PyInstaller, distro package) — its own
        # loader is self-sufficient.
        return False
    shebang = head.decode("utf-8", errors="replace").strip().lower()
    if "python" not in shebang:
        # A shell wrapper (e.g. the installer's bash launcher) execs the venv
        # python itself — leave it alone.
        return False
    shebang_interp = shebang[2:].strip().split()[0] if shebang.startswith("#!") else ""
    # ``#!/usr/bin/env python3`` always escapes to the DE's PATH python.
    # Do not treat this as "inside the venv" just because exe_dir is
    # ``/usr/bin`` (a substring of ``/usr/bin/env``).
    if shebang_interp in {"/usr/bin/env", "env"}:
        return True
    # Console script next to its own interpreter (venv/bin/hermes shebang
    # venv/bin/python3) is already self-sufficient, even when
    # sys.executable.resolve() would point at /usr/bin/python3.11.
    if shebang_interp:
        interp_path = Path(shebang_interp)
        try:
            if interp_path.is_file() and interp_path.parent == bin_path.parent:
                return False
        except OSError:
            pass
    # A python shebang pointing INSIDE the running interpreter's environment
    # already resolves correctly; anything else (a system path) would
    # escape the venv when spawned by the DE. Compare against the
    # unresolved interpreter so a venv symlink does not look like a
    # foreign /usr/bin python.
    exe_dir = str(_running_interpreter().parent)
    return exe_dir not in shebang


def _quote_exec_arg(arg: str) -> str:
    """Quote one ``Exec`` argument per the desktop entry spec.

    Reserved characters require double quotes. Inside the quotes, escape
    a backslash and a double quote with a backslash.
    """
    if not any(c in arg for c in ' \t\n"\'\\><~|&;$*?#()`'):
        return arg
    escaped = arg.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def render_desktop_entry(exec_command: str, icon: str) -> str:
    return (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=Hermes\n"
        "GenericName=Hermes Desktop\n"
        "Comment=Launch Hermes Desktop\n"
        f"Exec={exec_command}\n"
        f"Icon={icon}\n"
        "Terminal=false\n"
        "Categories=Utility;\n"
        "StartupNotify=true\n"
        "StartupWMClass=Hermes\n"
    )


def refresh_desktop_databases(applications_dir: Path) -> "list[str]":
    """Reindex the menu caches. Run each tool only when it exists.

    Return the names of the tools that ran (for logging and tests).
    """
    ran: list[str] = []

    update_db = shutil.which("update-desktop-database")
    if update_db:
        if _run_quiet([update_db, str(applications_dir)]):
            ran.append("update-desktop-database")

    # Plasma 6 first, then Plasma 5. Only one of them is ever installed.
    for tool in ("kbuildsycoca6", "kbuildsycoca5"):
        resolved = shutil.which(tool)
        if not resolved:
            continue
        if _run_quiet([resolved, "--noincremental"]):
            ran.append(tool)
        break

    return ran


def _run_quiet(cmd: "list[str]") -> bool:
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def install_desktop_entry(project_root: Path) -> Optional[Path]:
    """Write (or refresh) the Hermes desktop entry. Return its path.

    Return ``None`` on non-Linux platforms or when the write fails. This
    is a convenience, never a reason to fail a launch.
    """
    if not is_supported():
        return None

    entry_path = desktop_entry_path()
    icon = icon_path(project_root)
    # Use the themed name when the checkout has no icon (a lite or
    # packaged install). A broken absolute path renders as no icon.
    icon_value = str(icon) if icon.is_file() else "hermes"
    contents = render_desktop_entry(resolve_exec_command(), icon_value)

    try:
        entry_path.parent.mkdir(parents=True, exist_ok=True)
        # When nothing changed, skip the rewrite. Then a launch does not
        # churn the menu caches.
        if entry_path.is_file() and entry_path.read_text(encoding="utf-8") == contents:
            return entry_path
        entry_path.write_text(contents, encoding="utf-8")
        # Some launchers (and older Plasma) offer the entry only when it
        # is executable.
        entry_path.chmod(0o755)
    except OSError:
        return None

    refresh_desktop_databases(entry_path.parent)
    return entry_path
