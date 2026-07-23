"""Managed uv — one path, no guessing.

Hermes owns its own uv binary at ``$HERMES_HOME/bin/uv`` (or ``uv.exe`` on
Windows).  Every code path that needs uv resolves it from that single location.
If the binary is missing, ``ensure_uv()`` bootstraps it via the official
standalone installer with ``UV_UNMANAGED_INSTALL`` / ``UV_INSTALL_DIR`` pointed
at ``$HERMES_HOME/bin`` so the installer writes directly there — no PATH
probing, no conda guards, no multi-location resolution chains.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def managed_uv_path() -> Path:
    """Return the path where Hermes keeps *its* uv binary.

    ``$HERMES_HOME/bin/uv`` on POSIX, ``$HERMES_HOME\\bin\\uv.exe`` on
    Windows.  The directory may not exist yet — callers should use
    ``ensure_uv()`` to bootstrap it.
    """
    home = get_hermes_home()
    if platform.system() == "Windows":
        return home / "bin" / "uv.exe"
    return home / "bin" / "uv"


def resolve_uv() -> Optional[str]:
    """Return the managed uv path if it exists, else ``None``.

    No side effects — pure lookup.
    """
    p = managed_uv_path()
    if p.is_file() and os.access(p, os.X_OK):
        return str(p)
    return None


class _UvResult(str):
    """``ensure_uv()`` return value that survives an update boundary.

    ``ensure_uv()``'s arity has flipped between a single path string and a
    ``(path, fresh_bootstrap)`` tuple across releases. ``hermes update`` runs
    the call site from the *old*, already-imported ``hermes_cli.main`` against
    this *freshly pulled* module, so the two can disagree on how many values
    ``ensure_uv()`` returns. An install parked on a 2-tuple release runs
    ``uv_bin, fresh_bootstrap = ensure_uv()`` against the single-value module
    and crashes the first update: the returned path is a plain ``str``, which is
    itself iterable, so the 2-target unpack walks its characters and raises
    ``ValueError: too many values to unpack (expected 2)`` (and on the failure
    path the ``None`` return raises ``TypeError: cannot unpack non-iterable
    NoneType``). This wrapper answers to both conventions:

        uv_bin = ensure_uv()         # behaves as the path str ("" when absent)
        uv_bin, fresh = ensure_uv()  # unpacks as (path|None, fresh_bootstrap)

    Missing uv is the empty string (falsy) instead of ``None`` so legacy
    2-target call sites can still unpack a failure without raising, while
    ``if not uv_bin`` keeps working for single-value callers.

    POSIX only. This wrapper is **never** returned on Windows — see
    ``ensure_uv()`` for why the ``__iter__`` override is unsafe there.
    """

    fresh_bootstrap: bool

    def __new__(cls, path: Optional[str], fresh: bool = False) -> "_UvResult":
        self = super().__new__(cls, path or "")
        self.fresh_bootstrap = fresh
        return self

    def __iter__(self):
        # Tuple-unpacking hook for legacy ``uv_bin, fresh = ensure_uv()`` sites.
        # First element mirrors the historical contract: the path string, or
        # ``None`` when uv is unavailable.
        return iter(((str(self) or None), self.fresh_bootstrap))


def _ensure_uv_path() -> Optional[str]:
    """Resolve the managed uv path, installing it if necessary (plain ``str``/``None``)."""
    existing = resolve_uv()
    if existing:
        return existing

    target = managed_uv_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    print(f"  → Installing managed uv into {target.parent} ...")

    try:
        _install_uv(target)
    except Exception as exc:
        logger.warning("Managed uv install failed: %s", exc)
        print(f"  ✗ Failed to install managed uv: {exc}")
        return None

    # Verify
    result = resolve_uv()
    if result:
        version = subprocess.run(
            [result, "--version"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        print(f"  ✓ Managed uv installed ({version})")
    else:
        print("  ✗ Managed uv install appeared to succeed but binary not found")
    return result


def ensure_uv():
    """Return the managed uv path, installing it first if necessary.

    On **POSIX** the result is a :class:`_UvResult` (a ``str`` subclass) that is
    both usable directly as the path *and* unpackable as
    ``(path, fresh_bootstrap)`` for older call sites parked on a 2-tuple
    release — see :class:`_UvResult` for the update-boundary rationale.

    On **Windows** we deliberately return a plain ``str``/``None`` instead.
    ``subprocess`` there serializes the argv via ``subprocess.list2cmdline``,
    which iterates every entry *as a string* (``for c in arg``). The dependency
    installer passes uv straight into the command list (``[uv_bin, "pip", ...]``),
    so a ``_UvResult`` — whose ``__iter__`` yields ``(path, fresh_bootstrap)``
    rather than characters — would inject the bool into the command line and
    crash the install with ``TypeError: sequence item 1: expected str instance,
    bool found``. A plain ``str`` matches the historical Windows contract and is
    subprocess-safe. (A single value cannot satisfy both 2-target unpacking and
    Windows char-iteration: both use the iterator protocol, with contradictory
    results.)

    On failure the result is falsy — never raises — so callers can fall back to
    pip gracefully.
    """
    result = _ensure_uv_path()
    if platform.system() == "Windows":
        # See docstring: a str subclass with an overridden __iter__ is unsafe as
        # a Windows subprocess argument. Hand back the plain path (or None).
        return result
    return _UvResult(result)


def update_managed_uv() -> Optional[str]:
    """Update managed uv and replace Python builds with vulnerable SQLite.

    Call this during ``hermes update`` so the managed copy stays current.
    Returns the managed path when uv is available, or ``None`` when it is not.
    Self-update and runtime-migration failures are non-fatal.

    Updating uv matters independently of the Python patch version: uv freezes
    its python-build-standalone download catalog per release.  CPython 3.11.15
    appears in both the old and fixed catalogs, so ``uv python upgrade`` sees
    no patch-version change and leaves SQLite 3.50.4 in place.  A reinstall
    through current uv replaces that build with one linked against fixed
    SQLite.  Keep this migration here (rather than only in ``main.py``) because
    ``hermes update`` imports this module *after* pulling new code; installs
    updating across the fix boundary therefore migrate on their first update.
    """
    existing = resolve_uv()
    if not existing:
        # Not installed yet — ensure_uv() will handle that elsewhere.
        return None

    result = subprocess.run(
        [existing, "self", "update"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        version = subprocess.run(
            [existing, "--version"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        print(f"  ✓ Managed uv updated ({version})")
    else:
        # Non-fatal — old uv still works fine.
        logger.debug("uv self update failed (rc=%d): %s", result.returncode, result.stderr)

    # Best-effort for the same reason as uv's self-update: the WAL safety gate
    # in hermes_state keeps vulnerable installs usable in DELETE mode when the
    # network is unavailable or a running Windows interpreter prevents an
    # in-place runtime replacement.  A successful refresh takes effect for the
    # next Python process; this already-running updater may retain its old
    # sqlite3 module until it exits.
    upgrade_vulnerable_sqlite_runtime(existing)
    return existing


_SQLITE_WAL_RESET_VULNERABLE_EXIT = 42
_SQLITE_WAL_RESET_PROBE = (
    "import sqlite3\n"
    "v = sqlite3.sqlite_version_info\n"
    "vulnerable = (\n"
    "    v >= (3, 7, 0)\n"
    "    and v < (3, 51, 3)\n"
    "    and not ((3, 50, 7) <= v < (3, 51, 0))\n"
    "    and not ((3, 44, 6) <= v < (3, 45, 0))\n"
    ")\n"
    "print(sqlite3.sqlite_version)\n"
    f"raise SystemExit({_SQLITE_WAL_RESET_VULNERABLE_EXIT} if vulnerable else 0)\n"
)


def _probe_sqlite_runtime(python_executable: str) -> tuple[Optional[bool], str]:
    """Return ``(is_vulnerable, version)`` for an interpreter subprocess.

    ``None`` means the interpreter could not be probed.  A subprocess is
    required after a managed-Python reinstall because the current updater has
    already loaded its old sqlite3 extension and cannot observe the replacement
    in-process.
    """
    try:
        result = subprocess.run(
            [python_executable, "-c", _SQLITE_WAL_RESET_PROBE],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("SQLite runtime probe failed for %s: %s", python_executable, exc)
        return None, ""

    version = (result.stdout or "").strip().splitlines()
    version_text = version[-1] if version else ""
    if result.returncode == 0:
        return False, version_text
    if result.returncode == _SQLITE_WAL_RESET_VULNERABLE_EXIT:
        return True, version_text

    logger.debug(
        "SQLite runtime probe failed for %s (rc=%d): %s",
        python_executable,
        result.returncode,
        (result.stderr or "").strip(),
    )
    return None, version_text


def upgrade_vulnerable_sqlite_runtime(
    uv_bin: str,
    *,
    python_version: str = "3.11",
    python_executable: Optional[str] = None,
) -> bool:
    """Reinstall managed Python when its linked SQLite has the WAL-reset bug.

    Returns ``True`` when the active interpreter was already safe or a new
    process through the same interpreter now sees a fixed SQLite build.
    Returns ``False`` on a non-fatal migration failure.  On Windows a running
    venv may lock the base runtime; in that case the normal installer retries
    after stopping Hermes processes and rebuilding the venv.
    """
    active_python = python_executable or sys.executable
    vulnerable, sqlite_version = _probe_sqlite_runtime(active_python)
    if vulnerable is False:
        return True
    if vulnerable is None:
        return False

    version_label = sqlite_version or "unknown"
    print(
        f"  ⚠ SQLite {version_label} has the WAL-reset bug; "
        "refreshing managed Python..."
    )
    try:
        result = subprocess.run(
            [str(uv_bin), "python", "install", python_version, "--reinstall"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        logger.debug("Managed Python refresh could not start: %s", exc)
        print(
            "  ⚠ Could not replace the vulnerable Python runtime automatically. "
            "Close Hermes processes and re-run the installer."
        )
        return False
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        if detail:
            logger.debug("Managed Python refresh failed: %s", detail[-1])
        print(
            "  ⚠ Could not replace the vulnerable Python runtime automatically. "
            "Close Hermes processes and re-run the installer."
        )
        return False

    vulnerable_after, sqlite_after = _probe_sqlite_runtime(active_python)
    if vulnerable_after is False:
        print(f"  ✓ Managed Python now links fixed SQLite {sqlite_after}")
        return True

    after_label = sqlite_after or "unknown"
    print(
        f"  ⚠ This venv still links vulnerable SQLite {after_label}. "
        "Close Hermes processes and re-run the installer to rebuild it."
    )
    return False


# ---------------------------------------------------------------------------
# Installer internals
# ---------------------------------------------------------------------------

def _install_uv(target: Path) -> None:
    """Bootstrap uv into *target* using the official standalone installer.

    Uses ``UV_UNMANAGED_INSTALL`` (POSIX) or ``UV_INSTALL_DIR`` (Windows)
    so the astral installer writes the binary directly into
    ``$HERMES_HOME/bin/`` instead of ``~/.local/bin/``.
    """
    system = platform.system()
    env = {
        **os.environ,
        # Tell the astral installer to drop the binary in our dir, not
        # ~/.local/bin.  UV_UNMANAGED_INSTALL is the POSIX env var; Windows
        # uses UV_INSTALL_DIR.
        "UV_UNMANAGED_INSTALL": str(target.parent),
        "UV_INSTALL_DIR": str(target.parent),
    }

    if system == "Windows":
        _install_uv_windows(env)
    else:
        _install_uv_posix(env)


def _install_uv_posix(env: dict[str, str]) -> None:
    """Download + sh the POSIX installer (two-stage to avoid curl|sh pitfalls)."""
    with tempfile.NamedTemporaryFile(suffix=".sh", delete=False) as f:
        installer_path = f.name

    try:
        subprocess.run(
            ["curl", "-LsSf", "https://astral.sh/uv/install.sh", "-o", installer_path],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["sh", installer_path],
            env=env,
            check=True,
            capture_output=True,
        )
    finally:
        try:
            os.unlink(installer_path)
        except OSError:
            pass


def _install_uv_windows(env: dict[str, str]) -> None:
    """Invoke the PowerShell installer."""
    cmd = (
        'irm https://astral.sh/uv/install.ps1 | iex'
    )
    subprocess.run(
        ["powershell", "-ExecutionPolicy", "Bypass", "-c", cmd],
        env=env,
        check=True,
        capture_output=True,
    )

def rebuild_venv(uv_bin: str, venv_dir: Path, python_version: str = "3.11") -> bool:
    True # dont remove me. ask ethernet
