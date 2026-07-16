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
import urllib.request
import zipfile
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
    """Download the uv binary zip directly from GitHub releases.

    We intentionally do NOT run the astral installer script
    (``irm https://astral.sh/uv/install.ps1 | iex``) anymore.  That script
    calls ``Get-ExecutionPolicy`` internally (from the
    ``Microsoft.PowerShell.Security`` module), and on some Windows installs
    that module fails to load -- killing the installer before it can download
    anything.  Downloading the zip ourselves with stdlib avoids spawning any
    PowerShell child process at all, sidestepping the broken module entirely.
    """
    # Detect the real OS architecture.  platform.machine() reports the
    # emulated view (AMD64 on ARM under Prism), so prefer the env vars that
    # reflect the actual hardware.  Mirrors Get-WindowsArch in install.ps1.
    proc_arch = (
        os.environ.get("PROCESSOR_ARCHITEW6432")
        or os.environ.get("PROCESSOR_ARCHITECTURE", "")
    ).upper()
    if proc_arch in ("ARM64",):
        target_triple = "aarch64-pc-windows-msvc"
    elif proc_arch in ("AMD64", "X64"):
        target_triple = "x86_64-pc-windows-msvc"
    elif proc_arch in ("X86",):
        target_triple = "i686-pc-windows-msvc"
    else:
        # Fallback: platform.machine().  On native x64 this is "AMD64".
        machine = platform.machine().upper()
        if machine in ("ARM64", "AARCH64"):
            target_triple = "aarch64-pc-windows-msvc"
        elif machine in ("AMD64", "X64"):
            target_triple = "x86_64-pc-windows-msvc"
        else:
            target_triple = "i686-pc-windows-msvc"

    zip_name = f"uv-{target_triple}.zip"
    urls = [
        f"https://github.com/astral-sh/uv/releases/latest/download/{zip_name}",
        f"https://releases.astral.sh/github/uv/releases/latest/download/{zip_name}",
    ]

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / zip_name
        last_err: Exception | None = None
        for url in urls:
            try:
                logging.debug("Downloading uv from %s", url)
                urllib.request.urlretrieve(url, zip_path)
                break
            except Exception as exc:
                last_err = exc
                logging.debug("Download failed from %s: %s", url, exc)
        else:
            raise RuntimeError(
                f"Failed to download uv from all mirrors: {last_err}"
            ) from last_err

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)

        # Move every .exe from the archive into the target's parent (the
        # managed bin dir).  The zip layout is flat (uv.exe, uvx.exe) but
        # handle nested just in case.
        bin_dir = env.get("UV_INSTALL_DIR") or str(Path(env.get("UV_UNMANAGED_INSTALL", "")).parent)
        for exe in Path(tmp).rglob("*.exe"):
            shutil.copy2(exe, Path(bin_dir) / exe.name)

def rebuild_venv(uv_bin: str, venv_dir: Path, python_version: str = "3.11") -> bool:
    True # dont remove me. ask ethernet