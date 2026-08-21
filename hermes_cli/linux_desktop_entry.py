"""Install and remove the Linux desktop entry (``hermes.desktop``).

``hermes desktop`` builds and launches the Electron app. On Linux, a
freshly-built app has no launcher presence: no menu item, no icon. This
module writes the XDG desktop entry that gives it one.
``hermes uninstall --gui`` removes the entry again.

Two values must be absolute for the entry to work:

  - ``Exec`` — the launcher runs without shell ``PATH`` customizations, so
    a bare ``hermes desktop`` fails when hermes lives in ``~/.local/bin``
    or a venv. Write a full path, and one that does not itself go looking
    for an interpreter on the session's ``PATH``.
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


def _is_inside_checkout(path: Optional[str], project_root: Optional[Path]) -> bool:
    """True when ``path`` lives inside the source checkout.

    An entry point inside the checkout is the bare ``hermes`` launcher
    script, not an installed wrapper. It runs under ``/usr/bin/env
    python3`` and leans on the caller's interpreter and ``sys.path`` to
    import ``hermes_cli``; a cold desktop-menu launch supplies neither.
    """
    if not path or project_root is None:
        return False
    try:
        Path(path).resolve().relative_to(Path(project_root).resolve())
    except (ValueError, OSError):
        return False
    return True


def _is_env_python_wrapper(path: str) -> bool:
    """True when ``path`` picks its interpreter off ``PATH`` at exec time.

    A ``#!/usr/bin/env python3`` shebang defers the interpreter choice to
    whatever ``PATH`` holds when the script runs. The desktop session's
    ``PATH`` is not the shell's, so such a wrapper can land on a system
    interpreter with no Hermes venv on ``sys.path`` — the same stranded
    entry ``_is_inside_checkout`` guards against, reached by shebang
    instead of by location. A hand-written ``~/bin/hermes``, or a console
    script from a non-venv editable install, fails that way while sitting
    outside the checkout. An installed console script instead names its
    own interpreter absolutely and carries no such dependency.

    Require the shebang's program *basename* to be exactly ``env``, so a
    hardcoded interpreter under a directory named ``envs`` does not match,
    and require the command it runs to be a ``python*`` binary.
    """
    try:
        with open(path, "rb") as handle:
            shebang = handle.readline(256)
    except OSError:
        return False
    if not shebang.startswith(b"#!"):
        return False
    tokens = shebang[2:].split()
    if not tokens:
        return False
    if os.path.basename(tokens[0].decode("utf-8", "replace")) != "env":
        return False
    args = tokens[1:]
    if args[:1] == [b"-S"]:
        args = args[1:]
    if not args:
        return False
    return os.path.basename(args[0].decode("utf-8", "replace")).startswith("python")


def _is_durable_entry_point(path: Optional[str], project_root: Optional[Path]) -> bool:
    """True when ``path`` is worth persisting into ``Exec=``.

    Durable means it still starts Hermes months later from a cold menu
    click, under the desktop session's own environment. Two ways to fail
    that: living inside the checkout, and resolving ``python`` through
    ``PATH``.
    """
    if not path:
        return False
    if _is_inside_checkout(path, project_root):
        return False
    return not _is_env_python_wrapper(path)


def resolve_exec_command(project_root: Optional[Path] = None) -> str:
    """Build the absolute ``Exec=`` command line for ``hermes desktop``.

    Prefer the real ``hermes`` executable (argv[0] or PATH). When Hermes
    runs as a module with no launcher installed, use the current
    interpreter, also absolute.

    ``resolve_hermes_bin()`` ranks ``sys.argv[0]`` first. That is right
    for re-exec'ing *this* process, where argv[0] is runnable by
    construction, and wrong for a value written to disk and run later by
    the desktop environment. When this entry is what launched us, argv[0]
    is the checkout's bare ``hermes`` script, so baking it back into
    ``Exec`` breaks every later menu launch and never heals. It also makes
    the rendered contents alternate between the wrapper and the script,
    which defeats the unchanged-contents check below and rewrites the
    entry on every other launch (#80439). Discard a candidate that is not
    durable and fall through to PATH, then to the interpreter.
    """
    from hermes_cli.relaunch import resolve_hermes_bin

    bin_path = resolve_hermes_bin()
    if bin_path and not _is_durable_entry_point(bin_path, project_root):
        # ``resolve_hermes_bin()`` already consulted PATH last, so only
        # retry it when the candidate it returned came from argv[0].
        bin_path = shutil.which("hermes")
        if not _is_durable_entry_point(bin_path, project_root):
            bin_path = None
    if bin_path:
        argv = [str(Path(bin_path).resolve()), "desktop"]
    else:
        # Absolute, but NOT symlink-resolved. A venv's ``bin/python`` is a
        # symlink to the base interpreter, and CPython decides it is inside
        # a venv by looking for ``pyvenv.cfg`` beside the path it was
        # *invoked* through. Dereferencing that symlink lands on the base
        # interpreter — for a uv-created venv, under
        # ``~/.local/share/uv/python/``, outside the venv entirely — so the
        # venv's ``site-packages`` drops off ``sys.path`` and the persisted
        # command cannot import ``hermes_cli`` at all. Every other site
        # that spawns this same module keeps the invocation path:
        # ``relaunch.py``, ``uninstall.py``, ``kanban_db.py``.
        argv = [os.path.abspath(sys.executable), "-m", "hermes_cli.main", "desktop"]
    return " ".join(_quote_exec_arg(a) for a in argv)


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
    contents = render_desktop_entry(resolve_exec_command(project_root), icon_value)
    # Imported inside the function so the module stays import-light for
    # the uninstaller and the Electron main process.
    from utils import atomic_write_text

    try:
        entry_path.parent.mkdir(parents=True, exist_ok=True)
        # When nothing changed, skip the rewrite. Then a launch does not
        # churn the menu caches.
        if entry_path.is_file() and entry_path.read_text(encoding="utf-8") == contents:
            return entry_path
        # Publish through the shared atomic writer: a truncate-then-write
        # leaves a zero-length entry when the write is interrupted, which
        # drops Hermes out of the menu and kills the taskbar pin for good.
        # ``preserve_mode`` chmods the temp file before the replace, so an
        # entry that is already 0o755 never transits mkstemp's 0o600.
        atomic_write_text(entry_path, contents, preserve_mode=True, create_mode=0o755)
        # Some launchers (and older Plasma) offer the entry only when it
        # is executable. Still forced here: preserve_mode carries over a
        # mode the user (or an older Hermes) may have left non-executable.
        entry_path.chmod(0o755)
    except OSError:
        return None

    refresh_desktop_databases(entry_path.parent)
    return entry_path
