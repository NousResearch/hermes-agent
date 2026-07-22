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


def _ensure_uv_path() -> tuple[Optional[str], bool]:
    """Resolve the managed uv path and whether this call bootstrapped it."""
    existing = resolve_uv()
    if existing:
        return existing, False

    target = managed_uv_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    print(f"  → Installing managed uv into {target.parent} ...")

    fresh_bootstrap = False
    try:
        _install_uv(target)
        fresh_bootstrap = True
    except Exception as exc:
        logger.warning("Managed uv install failed: %s", exc)
        print(f"  ! Managed uv installer failed: {exc}")
        if not _copy_existing_uv_from_path(target):
            print("  ✗ Failed to install managed uv")
            return None, False
        fresh_bootstrap = True

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
    return result, bool(result and fresh_bootstrap)


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
    result, fresh_bootstrap = _ensure_uv_path()
    if platform.system() == "Windows":
        # See docstring: a str subclass with an overridden __iter__ is unsafe as
        # a Windows subprocess argument. Hand back the plain path (or None).
        return result
    return _UvResult(result, fresh_bootstrap)


def update_managed_uv() -> Optional[str]:
    """Run ``uv self update`` on the managed uv binary.

    Call this during ``hermes update`` so the managed copy stays current.
    Returns the managed path on success, ``None`` if uv isn't available or
    the self-update fails (non-fatal — the old version still works).
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
    return existing


# ---------------------------------------------------------------------------
# Installer internals
# ---------------------------------------------------------------------------

def _copy_existing_uv_from_path(target: Path) -> bool:
    """Copy an already-working PATH uv into the managed location.

    This is a recovery path for Windows/macOS/Linux machines where the official
    installer fails even though the user already has a valid uv available.  The
    managed path remains the only path returned to callers after the copy.
    """
    existing = shutil.which("uv")
    if not existing:
        return False

    existing_path = Path(existing)
    if existing_path == target:
        return target.is_file()

    probe = subprocess.run(
        [str(existing_path), "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        logger.warning("PATH uv probe failed: %s", probe.stderr)
        return False

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(existing_path, target)
    try:
        target.chmod(target.stat().st_mode | 0o755)
    except OSError:
        logger.debug("Could not chmod copied uv at %s", target, exc_info=True)

    version = probe.stdout.strip() or "uv on PATH"
    print(f"  ✓ Copied existing uv from PATH into managed bin ({version})")
    return True


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
