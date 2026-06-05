"""Managed uv — one path, no guessing.

Hermes owns its own uv binary at ``$HERMES_HOME/bin/uv`` (or ``uv.exe`` on
Windows).  Every code path that needs uv resolves it from that single location.
If the binary is missing, ``ensure_uv()`` bootstraps it via the official
standalone installer with ``UV_UNMANAGED_INSTALL`` / ``UV_INSTALL_DIR`` pointed
at ``$HERMES_HOME/bin`` so the installer writes directly there — no PATH
probing, no conda guards, no multi-location resolution chains.

``ensure_uv()`` returns a ``_UvResult`` (a ``str`` subclass) that is both
usable as the bare path and unpackable as ``(path, fresh_bootstrap)`` for
older call sites that straddle an update boundary (issue #39706).
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

    Why: ``ensure_uv()``'s return arity has flipped between a single path
    string and a ``(path, fresh_bootstrap)`` tuple across releases (issue
    #39706).  ``hermes update`` runs the call site from the *old*,
    already-imported ``hermes_cli.main`` against this *freshly pulled* module.
    An install parked on a 2-tuple release runs
    ``uv_bin, fresh_bootstrap = ensure_uv()`` against the single-value module:
    the returned path is a plain ``str``, which is iterable, so the 2-target
    unpack walks its characters and raises
    ``ValueError: too many values to unpack (expected 2)``.  This wrapper
    answers to both conventions:

        uv_bin = ensure_uv()         # behaves as the path str ("" when absent)
        uv_bin, fresh = ensure_uv()  # unpacks as (path|None, fresh_bootstrap)

    Missing uv is the empty string (falsy) instead of ``None`` so legacy
    2-target call sites can still unpack a failure without raising, while
    ``if not uv_bin`` keeps working for single-value callers.
    fresh_bootstrap is always ``False`` (the rebuild-venv path it gated was
    superseded by the unconditional _resolve_project_venv_root() approach).

    POSIX only — ``_UvResult`` is **never** returned on Windows because
    ``subprocess.list2cmdline`` iterates every argv entry *as a string*
    (per-character), so the overridden ``__iter__`` would inject the bool into
    the command line and crash with
    ``TypeError: sequence item 1: expected str instance, bool found``.

    What: str subclass with __iter__ that yields exactly ``(path|None,
    fresh_bootstrap)`` — a 2-tuple, always.

    WARNING: do NOT spread or iterate this object as a plain string.
    ``[*uv_bin]``, ``list(uv_bin)``, and ``for ch in uv_bin`` will yield
    the 2-tuple items ``(path|None, False)``, NOT the individual characters
    of the path.  This is intentional (it is the whole point of this class),
    but it means ``_UvResult`` must never be passed to APIs that iterate
    their argv entries as strings (e.g. ``subprocess.list2cmdline`` on
    Windows — which is why we return a plain str/None there instead).

    Test: ``a, b = ensure_uv()`` unpacks without error; ``if not ensure_uv()``
    stays falsy when uv is absent; Windows path returns plain str/None;
    ``list(ensure_uv())`` has exactly 2 elements (not len(path) characters).
    """

    fresh_bootstrap: bool

    def __new__(cls, path: Optional[str], fresh: bool = False) -> "_UvResult":
        self = super().__new__(cls, path or "")
        self.fresh_bootstrap = fresh
        # Runtime invariant: __iter__ must always yield exactly 2 items so the
        # dual-mode contract (path-str AND 2-tuple unpack) can never silently
        # drift if this class is ever modified.  Checked here rather than in
        # __iter__ so the assertion fires at construction time, not at unpack.
        assert len(list(self.__iter__())) == 2, (  # noqa: S101
            f"_UvResult.__iter__ must yield exactly 2 items for #39706 "
            f"update-boundary compat; got {list(self.__iter__())!r}"
        )
        return self

    def __iter__(self):  # type: ignore[override]  # 2026-06: intentional dual-mode return for #39706 update boundary
        # Tuple-unpacking hook for legacy ``uv_bin, fresh = ensure_uv()`` sites.
        # First element mirrors the historical contract: the path string, or
        # ``None`` when uv is unavailable.
        return iter(((str(self) or None), self.fresh_bootstrap))


def _ensure_uv_path() -> Optional[str]:
    """Resolve the managed uv path, installing it if necessary.

    Why: internal helper used by both the POSIX (_UvResult wrapper) and
    Windows (plain str) code paths.
    What: returns the path string on success, None on failure.
    Test: called indirectly via ensure_uv(); see TestEnsureUv.
    """
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


def ensure_uv() -> "_UvResult":
    """Return the managed uv path, installing it first if necessary.

    Why: the return arity has flipped across releases; this wrapper ensures
    callers on either side of an update boundary can unpack the result safely
    (issue #39706).  See ``_UvResult`` for the full explanation.

    Returns a :class:`_UvResult` (a ``str`` subclass) on POSIX — usable
    directly as the path *and* unpackable as ``(path, fresh_bootstrap)`` for
    older call sites.  On Windows returns a plain ``str``/``None`` because
    ``subprocess.list2cmdline`` iterates argv entries per-character and the
    overridden ``__iter__`` would break the command line.

    On failure the result is falsy (empty string on POSIX, ``None`` on
    Windows) — never raises — so callers can fall back to pip gracefully.

    What: resolves/installs managed uv; returns update-boundary-safe result.
    Test: ``a, b = ensure_uv()`` does not raise; ``if not ensure_uv()`` is
    falsy when absent; Windows path returns plain str/None.
    """
    path = _ensure_uv_path()
    if platform.system() == "Windows":
        return path  # type: ignore[return-value]
    return _UvResult(path)


def rebuild_venv(uv_bin: str, venv_dir: Path, python_version: str = "3.11") -> bool:
    """Nuke and recreate the venv with managed uv.

    Called when managed uv is first bootstrapped on an existing install — the
    old venv may point to a Python without FTS5, so we rebuild it with a
    fresh interpreter from the current managed uv.  Returns ``True`` on
    success.

    The old venv is moved aside *atomically* (``os.replace`` to ``<venv>.old``)
    before recreating — never deleted in place. On Windows a still-running
    ``hermes.exe`` (gateway/desktop) holds ``venv\\Scripts\\python.exe`` open;
    ``shutil.rmtree(ignore_errors=True)`` would delete everything it *can*
    (site-packages, certifi's cert bundle) and silently leave a half-gutted
    venv that the following ``uv venv`` then refuses to overwrite ("directory
    already exists") — bricking the install with no recovery (every later HTTPS
    call dies with ``FileNotFoundError`` for the missing cert bundle).
    ``--clear`` alone does not fix this: when the locked interpreter is *inside*
    the venv being rebuilt, neither ``rmtree`` nor ``uv venv --clear`` can
    delete the held ``python.exe``. ``os.replace`` of the parent directory *is*
    allowed (Windows tracks a running ``.exe`` by handle, not path), so the
    rebuild completes while the running process keeps using the moved-aside copy
    until it restarts. If the venv genuinely cannot be moved, we abort cleanly
    and leave it fully intact; and if the rebuild itself fails we move the old
    venv back so Hermes is never left with no venv at all.
    """
    backup: Optional[Path] = None
    if venv_dir.exists():
        print(f"  → Rebuilding venv (old Python may lack FTS5)...")
        backup = venv_dir.with_name(venv_dir.name + ".old")
        shutil.rmtree(backup, ignore_errors=True)  # clear any stale backup
        try:
            # Atomic move — fails (without partial deletion) if a process still
            # holds files inside the venv, which is exactly the Windows
            # file-lock case that previously bricked the install.
            os.replace(venv_dir, backup)
        except OSError as exc:
            logger.warning("venv rebuild aborted — venv in use: %s", exc)
            print(
                "  ✗ venv rebuild aborted — the venv is in use; stop the "
                f"gateway/desktop and retry ({exc})"
            )
            return False

    result = subprocess.run(
        [uv_bin, "venv", str(venv_dir), "--python", python_version, "--clear"],
        capture_output=True,
        text=True,
        check=False,
    )

    def _restore_backup() -> None:
        if backup is not None and backup.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)
            try:
                os.replace(backup, venv_dir)
                print("  ↩ Restored previous venv after failed rebuild.")
            except OSError:
                pass

    if result.returncode == 0:
        venv_python = (
            venv_dir
            / ("Scripts" if platform.system() == "Windows" else "bin")
            / "python"
        )
        # uv can exit 0 yet leave no usable interpreter (e.g. a half-written
        # venv). Don't report success on a venv that has no python — restore the
        # moved-aside copy so the caller can abort without losing a working env.
        if not venv_python.exists():
            logger.warning(
                "venv rebuild reported success but %s is missing", venv_python
            )
            print(
                f"  ✗ venv rebuild failed: Python interpreter missing at {venv_python}"
            )
            _restore_backup()
            return False
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
        py_ver = subprocess.run(
            [str(venv_python), "--version"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        print(f"  ✓ venv rebuilt ({py_ver})")
        return True
    else:
        # Rebuild failed — restore the old venv so we never leave Hermes with no
        # venv (the bricked-install failure mode this function exists to avoid).
        _restore_backup()
        logger.warning("venv rebuild failed: %s", result.stderr)
        print(f"  ✗ venv rebuild failed: {result.stderr.strip()}")
        return False


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
        logger.debug(
            "uv self update failed (rc=%d): %s", result.returncode, result.stderr
        )
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
    """Invoke the PowerShell installer."""
    cmd = "irm https://astral.sh/uv/install.ps1 | iex"
    subprocess.run(
        ["powershell", "-ExecutionPolicy", "Bypass", "-c", cmd],
        env=env,
        check=True,
        capture_output=True,
    )
