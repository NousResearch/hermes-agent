"""Layout-aware self-heal for the primary ``hermes`` launcher (issue #76421).

One check/repair implementation shared by ``hermes update`` (which calls it
before the "Already up to date!" return and again post-update) and
``hermes doctor --fix`` (Command Installation section).

Repairs, in the ONE command dir the installer would use:
- missing launchers,
- dangling/wrong legacy symlinks (older installs created the launcher path
  as a symlink into the venv),
- recognized stale Hermes-managed wrappers (regular files matching the
  managed shim from ``scripts/install.sh`` whose exec targets are gone or
  no longer point at the current project).

Preserves arbitrary user-managed regular wrappers — warns, never overwrites,
when ownership cannot be established.  Never writes through a symlink: the
launcher path is unlinked first and a NEW regular-file shim is written (the
#21454 safety contract — ``scripts/install.sh`` does ``rm -f`` before
``cat >``).  Never touches the ``hermes-acp``/ACP path.

The shim is the EXACT installer form (``scripts/install.sh`` setup_path):
the venv INTERPRETER plus the checked-in ``hermes`` entrypoint — NOT the
uv console script, which resolves itself through ``realpath`` and stock
macOS does not ship ``realpath``.
"""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path
from typing import Iterable, List, Optional

# Status values returned by ``managed_launcher_status``.
HEALTHY = "healthy"
MISSING = "missing"
WRONG_SYMLINK = "wrong_symlink"
STALE_MANAGED = "stale_managed"
USER_WRAPPER = "user_wrapper"


def managed_shim_text(venv_python: Path, entrypoint: Path) -> str:
    """The managed shim, byte-mirroring ``scripts/install.sh``'s venv launcher.

    ``venv_python`` is the venv interpreter (``<root>/venv/bin/python`` or
    ``.venv/bin/python``); ``entrypoint`` is the checked-in ``hermes``
    script at the project root.  Deliberately NOT the uv console script:
    those resolve through ``realpath``, which stock macOS lacks.
    """
    return (
        "#!/usr/bin/env bash\n"
        "unset PYTHONPATH\n"
        "unset PYTHONHOME\n"
        f'exec "{venv_python}" "{entrypoint}" "$@"\n'
    )


def _command_link_layout() -> str:
    """Which install layout governs the command dir: termux / fhs / user.

    Mirrors install.sh's ``get_command_link_dir`` (and its ``is_termux``):
    Termux when TERMUX_VERSION is set or PREFIX points at the Termux prefix;
    FHS when running as root on Linux; otherwise the user-local layout.
    """
    prefix = os.environ.get("PREFIX", "")
    is_termux = bool(os.environ.get("TERMUX_VERSION")) or (
        "com.termux/files/usr" in prefix
    )
    if is_termux and prefix:
        return "termux"
    try:
        if sys.platform == "linux" and os.geteuid() == 0:
            return "fhs"
    except AttributeError:
        pass
    return "user"


def select_command_dir() -> Path:
    """The ONE dir the installer places the ``hermes`` launcher in.

    Termux → ``$PREFIX/bin``; root on Linux (FHS) → ``/usr/local/bin``;
    otherwise ``~/.local/bin``.  Shared by the updater and doctor so both
    agree on a single location instead of sweeping every candidate dir
    (which could create a second launcher on FHS installs).
    """
    layout = _command_link_layout()
    if layout == "termux":
        return Path(os.environ["PREFIX"]) / "bin"
    if layout == "fhs":
        return Path("/usr/local/bin")
    return Path.home() / ".local" / "bin"


def command_dir_display() -> str:
    """User-facing display form of ``select_command_dir()`` (for doctor)."""
    return {
        "termux": "$PREFIX/bin",
        "fhs": "/usr/local/bin",
        "user": "~/.local/bin",
    }[_command_link_layout()]


def _exec_line_targets(text: str) -> List[str]:
    """Absolute paths referenced by the first ``exec`` line of a shim."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("exec ") or stripped == "exec":
            try:
                tokens = shlex.split(stripped, posix=True)
            except ValueError:
                return []
            return [t for t in tokens[1:] if t.startswith("/")]
    return []


def _is_managed_wrapper(text: str) -> bool:
    """Recognize a Hermes-managed wrapper conservatively.

    Managed markers: bash shebang, ``unset PYTHONPATH``, and an ``exec``
    line referencing hermes (the checked-in entrypoint is always named
    ``hermes``).  Anything else is treated as a user wrapper: when in
    doubt, warn instead of overwrite.
    """
    lines = text.splitlines()
    if not lines or not lines[0].startswith("#!") or "bash" not in lines[0]:
        return False
    if "unset PYTHONPATH" not in text:
        return False
    exec_line = next(
        (l.strip() for l in lines if l.strip().startswith("exec")), None
    )
    if exec_line is None:
        return False
    return "hermes" in exec_line


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def managed_launcher_status(path: Path, project_root: Path) -> str:
    """Classify the launcher at ``path`` against the current ``project_root``.

    Returns one of ``HEALTHY``, ``MISSING``, ``WRONG_SYMLINK``,
    ``STALE_MANAGED`` or ``USER_WRAPPER``.

    A recognized managed wrapper is HEALTHY when every absolute path its
    exec line references exists AND is under the current project root; it
    is STALE only when a referenced path is gone or points outside the
    current project (e.g. a removed or relocated venv).  The current
    installer form (venv python + checked-in entrypoint, both inside the
    project) therefore always classifies healthy.
    """
    root = Path(project_root).resolve()
    if path.is_symlink():
        # is_symlink() catches dangling links that exists() would miss.
        if path.exists() and _is_under(path.resolve(), root):
            return HEALTHY
        return WRONG_SYMLINK
    if not path.exists():
        return MISSING
    if not path.is_file():
        # Directory, fifo, ... — never touch something we don't understand.
        return USER_WRAPPER
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return USER_WRAPPER
    if not _is_managed_wrapper(text):
        return USER_WRAPPER
    targets = _exec_line_targets(text)
    if (
        targets
        and all(Path(t).exists() for t in targets)
        and all(_is_under(Path(t).resolve(), root) for t in targets)
    ):
        return HEALTHY
    # Recognized managed wrapper whose exec targets are gone or no longer
    # point at the current project.
    return STALE_MANAGED


def repair_launcher(
    path: Path, venv_python: Path, entrypoint: Path, project_root: Path
) -> List[str]:
    """Repair the launcher at ``path``; returns human-readable actions taken.

    Replaces the launcher path itself: unlinks the path (never writes
    through a symlink, #21454), then writes a NEW regular-file shim in the
    exact installer form.  User-managed wrappers are preserved with a
    warning.  Raises ``OSError`` on unwritable locations so the caller can
    skip the directory silently.
    """
    status = managed_launcher_status(path, project_root)
    if status == HEALTHY:
        return []
    if status == USER_WRAPPER:
        return [
            f"⚠ {path} exists but is not a recognized Hermes-managed wrapper "
            "— leaving it untouched"
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    path.write_text(managed_shim_text(venv_python, entrypoint), encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o755)
    verbs = {
        MISSING: "Installed hermes launcher",
        WRONG_SYMLINK: "Replaced legacy symlink with managed shim",
        STALE_MANAGED: "Repaired stale Hermes-managed wrapper",
    }
    return [f"✓ {verbs[status]}: {path} → {venv_python} {entrypoint}"]


def ensure_hermes_launcher(
    venv_python: Path,
    entrypoint: Path,
    project_root: Path,
    bin_dirs: Optional[Iterable[Path]] = None,
) -> List[str]:
    """Check/repair the ``hermes`` launcher in the ONE command dir.

    Defaults to ``[select_command_dir()]`` — the same single layout the
    installer and doctor use, never a sweep that could create a second
    launcher.  ``bin_dirs`` is injectable for tests.  Unwritable
    directories are skipped silently.
    """
    actions: List[str] = []
    dirs = [select_command_dir()] if bin_dirs is None else list(bin_dirs)
    for bin_dir in dirs:
        try:
            actions.extend(
                repair_launcher(
                    Path(bin_dir) / "hermes", venv_python, entrypoint, project_root
                )
            )
        except OSError:
            continue
    return actions
