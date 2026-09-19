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

import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Mapping, Optional

# XDG startup notification: set by an app-grid / menu launch, absent for terminal and detached
# (updater relaunch) launches. See launched_from_shell().
SHELL_LAUNCH_ENV_VAR = "DESKTOP_STARTUP_ID"

DESKTOP_ENTRY_NAME = "hermes.desktop"

logger = logging.getLogger(__name__)


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
    if bin_path:
        resolved = Path(bin_path).resolve()
        if _needs_interpreter(resolved):
            # The resolved launcher is a Python script whose shebang points
            # at a NON-venv interpreter (e.g. the repo's `hermes` script with
            # `#!/usr/bin/env python3` when argv[0] came from the shell
            # installer's bash wrapper). Launched from the .desktop entry that
            # shebang resolves to an interpreter that may NOT have hermes_cli
            # importable — silently failing under Terminal=false. Prefix it
            # with an interpreter that can actually run Hermes: prefer
            # sys.executable only when it can import hermes_cli, otherwise
            # the installed venv-backed wrapper on PATH, otherwise run as a
            # module under sys.executable.
            argv = _interpreter_for(resolved)
        else:
            argv = [str(resolved), "desktop"]
    else:
        argv = [str(_running_interpreter()), "-m", "hermes_cli.main", "desktop"]
    return " ".join(_quote_exec_arg(a) for a in argv)


def _running_interpreter() -> Path:
    """The interpreter actually running Hermes, as an absolute path.

    Use ``os.path.abspath`` rather than ``Path(...).resolve()``. On POSIX a
    venv's ``bin/python`` is a symlink to the base interpreter (the default
    for both ``python -m venv`` and ``uv venv``). ``.resolve()`` follows that
    symlink out of the venv to an interpreter whose site-packages lack
    Hermes' dependencies, so the capability probe in ``_interpreter_for``
    would wrongly answer "cannot import hermes_cli" and discard the one
    interpreter that works. ``abspath`` normalises the path without
    dereferencing the symlink, keeping us inside the venv.
    """
    return Path(os.path.abspath(sys.executable))


def _interpreter_for(script: Path) -> "list[str]":
    """Build an argv prefix that runs *script* under a Hermes-capable python.

    Resolution order, first match wins:
      1. ``sys.executable`` — but only when it can import ``hermes_cli``.
         Depending on how Hermes was launched (uv shim, system python, a
         non-venv interpreter), ``sys.executable`` itself may lack
         hermes_cli, in which case prefixing it would reproduce the same
         silent failure the .desktop is trying to avoid.
      2. The installed ``hermes`` wrapper on PATH (e.g. ~/.local/bin/hermes),
         but ONLY when that wrapper is itself safe to exec — a native binary
         or a bash launcher that exec's the venv python. A wrapper carrying a
         ``#!/usr/bin/env python3`` shebang is the same broken script
         ``resolve_hermes_bin()`` already handed us (when argv[0] is unusable
         it returns ``shutil.which("hermes")`` verbatim), so reusing it would
         silently restore the pre-fix ``Exec=``. Such a wrapper is skipped and
         we fall through to rung 3.
      3. Fall back to ``sys.executable -m hermes_cli.main``.
    """
    if _can_import_hermes_cli(_running_interpreter()):
        argv = [str(_running_interpreter()), str(script), "desktop"]
        logger.debug("desktop entry: using sys.executable (hermes_cli importable)")
        return argv
    wrapper = shutil.which("hermes")
    if wrapper and not _needs_interpreter(Path(wrapper).resolve()):
        # Only trust a wrapper that is genuinely runnable as-is (native
        # binary or a bash launcher that exec's the venv python). A python
        # shebang wrapper is the same foreign script we just rejected, so it
        # would reproduce the silent failure — skip it and let rung 3 answer.
        argv = [str(Path(wrapper).resolve()), "desktop"]
        logger.debug("desktop entry: using PATH wrapper %s", wrapper)
        return argv
    argv = [str(_running_interpreter()), "-m", "hermes_cli.main", "desktop"]
    logger.debug("desktop entry: falling back to %s -m hermes_cli.main", _running_interpreter())
    return argv


def _can_import_hermes_cli(interpreter: Path) -> bool:
    """Whether *interpreter* can import Hermes' CLI entry module (the venv gate).

    Bound the probe with a timeout so a hung interpreter (e.g. one that
    stalls on site initialization) can't stall desktop-entry generation for
    longer than a few seconds. Isolated mode (-I) ignores inherited Python
    path overrides (a PYTHONPATH pointing at a checkout with a stub
    hermes_cli would otherwise fake a successful import), and the root cwd
    keeps the implicit import path neutral.
    """
    try:
        result = subprocess.run(
            [str(interpreter), "-I", "-c", "import hermes_cli.main"],
            cwd=os.path.abspath(os.sep),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


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
    shebang = head.decode("utf-8", errors="replace").strip()
    if "python" not in shebang.lower():
        # A shell wrapper (e.g. the installer's bash launcher) execs the venv
        # python itself — leave it alone.
        return False
    # Match path components, not substrings: ``venv/bin-extra/python`` is NOT
    # inside ``venv/bin`` even though the former starts with the latter. Compare
    # the resolved parent of the shebang's interpreter against the running
    # interpreter's parent.
    interpreter = Path(shebang[2:].split(maxsplit=1)[0])
    return interpreter.parent != _running_interpreter().parent


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
        "StartupNotify=false\n"
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


def launched_from_shell(environ: Optional[Mapping[str, str]] = None) -> bool:
    """True when this process was started from the app grid / menu (XDG startup notification).

    A grid launch has a gnome-shell ShellApp in STARTING until our window maps; unpatched
    shells (before GNOME MR !4428) drop that app's last reference when its own ``.desktop``
    entry changes, and the next idle GC kills the whole Wayland session (#111906). A false
    negative degrades to the pre-#111906 behaviour; a false positive only delays a heal.
    """
    env = os.environ if environ is None else environ
    return bool(env.get(SHELL_LAUNCH_ENV_VAR))


def _ready_fd_env_var() -> str:
    return "HERMES_DESKTOP_READY_FD"


def _reveal_byte() -> bytes:
    return b"\x00"


def _desktop_ready_fd() -> int | None:
    raw = os.environ.get(_ready_fd_env_var())
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


READY_FD_ENV_VAR = _ready_fd_env_var()
REVEAL_BYTE = _reveal_byte()


class DeferredDesktopEntryInstall:
    """Install the entry once the desktop window is on screen — never while the shell's
    ShellApp is STARTING.

    The launcher hands Electron the write end of a pipe (``HERMES_DESKTOP_READY_FD``); Electron
    writes one byte when the main window is revealed and a worker thread then installs the entry.
    An exit without a reveal (boot crash, early quit) does NOT heal: gnome-shell keeps the ShellApp
    in STARTING until the startup-notification sequence completes or times out (mutter, ~15 s),
    not until the process dies, so a write right after such an exit still lands in the arming
    window. The next terminal/updater launch or revealed grid launch installs the entry instead.
    Terminal and detached launches never build one of these: they install immediately, as before.
    """

    def __init__(
        self,
        project_root: Path,
        install: Optional[Callable[[Path], Optional[Path]]] = None,
        settle_seconds: float = 2.0,
    ) -> None:
        self._project_root = project_root
        self._install = install or install_desktop_entry
        # Electron reports the reveal before the compositor has necessarily mapped the surface;
        # a short settle after the signal keeps the write on the RUNNING side. It is a margin
        # after the condition, not a substitute for it.
        self._settle_seconds = settle_seconds
        self._read_fd, self.write_fd = os.pipe()
        self._thread = threading.Thread(target=self._wait_for_reveal, name="desktop-entry-install", daemon=True)

    def child_env(self, env: dict) -> dict:
        env[READY_FD_ENV_VAR] = str(self.write_fd)
        return env

    @property
    def pass_fds(self) -> tuple[int, ...]:
        return (self.write_fd,)

    def start(self) -> None:
        self._thread.start()

    def _wait_for_reveal(self) -> None:
        try:
            data = os.read(self._read_fd, 1)
        except OSError:
            return
        if data != REVEAL_BYTE:  # woken by finish(): the app exited without a reveal, skip the heal
            return
        if self._settle_seconds:
            time.sleep(self._settle_seconds)
        try:
            entry = self._install(self._project_root)
            if entry:
                print(f"Desktop launcher entry installed: {entry}")
        except Exception as exc:  # never fail a launch on launcher plumbing
            print(f"Could not install the desktop launcher entry: {exc}")

    def finish(self) -> None:
        try:
            os.write(self.write_fd, REVEAL_BYTE)
        except OSError:
            return
        self._thread.join(timeout=self._settle_seconds + 1.0)
