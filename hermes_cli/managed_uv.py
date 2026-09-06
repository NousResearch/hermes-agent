"""Hermes-managed uv and Python runtime repair.

Hermes owns its own uv binary at ``$HERMES_HOME/uv/uv`` (or ``uv.exe`` on
Windows). Every code path that needs uv resolves it from that single private
location. If the binary is missing, ``ensure_uv()`` bootstraps it via the
official standalone installer. The private directory is never added to PATH,
so Hermes cannot shadow a user's uv in interactive shells.

Legacy installs that placed the managed uv at ``$HERMES_HOME/bin/uv`` (the
pre-isolation layout — bin is a persisted User PATH entry on Windows) are
migrated to the private location on first use.
The Python backing the install is shared by every Hermes profile because the checkout's ``venv``
is shared. Runtime repair therefore uses an install-scoped store under
``<checkout>/.hermes-runtime/python``. A vulnerable interpreter is never reinstalled in place.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Optional

from hermes_constants import get_hermes_home
from hermes_cli.sqlite_runtime import (
    SQLiteRuntimeInfo, isolated_interpreter_env, probe_sqlite_runtime)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_RUNTIME_DIR_NAME = ".hermes-runtime"
_VENV_NAME = "venv"
_ALT_VENV_NAME = ".venv"
_REPAIR_LOCK_NAME = "runtime-repair.lock"
_MACOS_MANAGED_PYTHON_IDENTIFIER = "com.nousresearch.hermes.managed-python"

_Provisioned = tuple[Path, Path, SQLiteRuntimeInfo]


def _managed_uv_dir() -> Path:
    """Return the private managed-uv dir (``$HERMES_HOME/uv``).

    The uv-isolation invariant lives here: this directory is never registered
    on PATH, so interactive shells always resolve the user's own uv/uvx (or
    none) rather than Hermes' copies.
    """
    return get_hermes_home() / "uv"


def _legacy_managed_bin_dir() -> Path:
    """Return the pre-isolation managed-binary dir (``$HERMES_HOME/bin``) —
    the layout migrated by :func:`_migrate_legacy_managed_uv`."""
    return get_hermes_home() / "bin"


def managed_uv_path() -> Path:
    """Return the path where Hermes keeps *its own* uv binary.

    ``$HERMES_HOME/uv/uv`` on POSIX, ``$HERMES_HOME\\uv\\uv.exe`` on
    Windows.  This is a **private** location: it is never registered on PATH,
    so interactive shells always resolve the user's own uv (or none) rather
    than Hermes' copy.  The directory may not exist yet — callers should use
    ``ensure_uv()`` to bootstrap it.
    """
    if platform.system() == "Windows":
        return _managed_uv_dir() / "uv.exe"
    return _managed_uv_dir() / "uv"


def managed_uvx_path() -> Path:
    """Return the path where Hermes keeps its private ``uvx`` binary."""
    suffix = ".exe" if platform.system() == "Windows" else ""
    return managed_uv_path().with_name(f"uvx{suffix}")


def managed_uv_env(
    *,
    base_env: dict[str, str] | None = None,
    tool_bin_dir: Path | str | None = None,
) -> dict[str, str]:
    """Return a sanitized environment for a Hermes-private uv invocation.

    Pins every directory uv writes to inside Hermes' own tree, so no uv
    operation Hermes runs can touch the user's uv-managed state — their tool
    store (``UV_TOOL_DIR``, where ``uv tool install``/``uvx`` keep tools),
    download cache (``UV_CACHE_DIR``), or python store (``UV_PYTHON_INSTALL_DIR``
    plus the ``~/.local/bin`` shims and Windows registry it would otherwise
    write).  The values OVERRIDE anything inherited: a user who exported
    their own ``UV_*`` dirs still has Hermes write only where Hermes owns.

    ``tool_bin_dir``, when given, additionally pins where ``uv tool install``
    links its shims (Hermes uses ``$HERMES_HOME/bin`` so ``_find_cli``-style
    resolvers find them without the user's PATH).

    ``UV_TOOL_BIN_DIR`` is the ONE uv write dir this helper does NOT pin:
    when ``tool_bin_dir`` is omitted, an inherited user ``UV_TOOL_BIN_DIR``
    is deliberately left untouched. ``uv tool install`` / ``uvx`` will then
    link tools into the user's own shim dir, so any call site that installs
    built tools MUST pass ``tool_bin_dir`` explicitly (as
    ``browser_use_cli.install_cli`` does) — never rely on inheritance here.
    Only shim *linkage* is at stake: the tools themselves, their cache, and
    any Python they provision stay inside Hermes' tree via the pins above.

    Callers keep their own decisions about ``UV_NO_CONFIG`` (respecting a
    user's ``uv.toml`` mirrors is a feature at some call sites) and about
    credential stripping — pass the already-sanitized env as ``base_env``.
    """
    env = dict(os.environ if base_env is None else base_env)
    env.update({
        "UV_CACHE_DIR": str(get_hermes_home() / "cache" / "uv"),
        "UV_TOOL_DIR": str(_managed_uv_dir() / "tools"),
        "UV_PYTHON_INSTALL_DIR": str(get_hermes_home() / "python"),
        "UV_PYTHON_INSTALL_BIN": "0",
        "UV_PYTHON_INSTALL_REGISTRY": "0",
    })
    if tool_bin_dir is not None:
        env["UV_TOOL_BIN_DIR"] = str(tool_bin_dir)
    return env


def legacy_managed_uv_path() -> Path:
    """Return the pre-isolation managed-uv location, if it is still around.

    Installers before the uv isolation change placed the managed binary at
    ``$HERMES_HOME/bin/uv(.exe)`` — ``bin`` is a persisted User PATH entry on
    Windows, so that layout silently shadowed the user's own uv in every new
    shell.  Callers migrate a legacy binary to :func:`managed_uv_path` once;
    this accessor exists so the migration can recognize the old shape.
    """
    if platform.system() == "Windows":
        return _legacy_managed_bin_dir() / "uv.exe"
    return _legacy_managed_bin_dir() / "uv"


def resolve_uv() -> Optional[str]:
    """Return the managed uv path if it exists, else ``None``.

    No side effects — pure lookup.  **Managed-only**: this never resolves the
    user's own uv on PATH, which is exactly what keeps
    :func:`update_managed_uv` from ever modifying a toolchain Hermes does not
    own.
    """
    p = managed_uv_path()
    return str(p) if p.is_file() and os.access(p, os.X_OK) else None


def _migrate_legacy_binary(name: str) -> bool:
    """Move a pre-isolation ``$HERMES_HOME/bin/<name>(.exe)`` to the private
    dir, once.

    The astral installer always drops both ``uv`` and ``uvx`` into the target
    dir, so a legacy install leaves BOTH in ``bin`` — and on Windows ``bin``
    is a persisted User PATH entry, so a stale ``bin/uvx`` keeps shadowing
    the user's own ``uvx`` in every new shell exactly like ``bin/uv`` did.
    Both must be migrated.  Best-effort: a locked or busy legacy binary
    simply stays put (a concurrent process may hold it); the next bootstrap
    retries.  If the private copy is already present, the old managed name
    is removed so it cannot shadow a user's binary through the persisted
    bin/ PATH entry.
    """
    exe = ".exe" if platform.system() == "Windows" else ""
    legacy = _legacy_managed_bin_dir() / f"{name}{exe}"
    target = _managed_uv_dir() / f"{name}{exe}"
    if not legacy.is_file():
        return False
    if target.is_file():
        # A previous migration/install may already have produced the private
        # copy.  Remove the old managed name so it cannot continue shadowing
        # through the persisted $HERMES_HOME/bin PATH entry.
        try:
            legacy.unlink()
            return True
        except OSError:
            return False
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(str(legacy), str(target))
        return True
    except OSError:
        return False


def _migrate_legacy_managed_uv() -> bool:
    """Migrate pre-isolation ``bin/uv(.exe)`` and ``bin/uvx(.exe)``.

    True when anything was moved or cleaned.  Legacy installs must be
    migrated on first use regardless of which binary the calling path needs
    — leaving ``uvx`` behind would keep leaking a stale Hermes-managed
    ``uvx`` through the persisted ``bin/`` PATH entry on Windows.
    """
    moved_uv = _migrate_legacy_binary("uv")
    moved_uvx = _migrate_legacy_binary("uvx")
    return moved_uv or moved_uvx


def managed_python_install_dir(project_root: Path | None = None) -> Path:
    """Return the checkout-scoped Python store shared by all profiles.

    This is the **runtime-repair** store: ``repair_vulnerable_runtime``
    provisions a new immutable generation here so the previous one stays
    available for synchronous rollback.  It is deliberately separate from the
    **installer** store (``$HERMES_HOME/python``, set via
    ``UV_PYTHON_INSTALL_DIR`` in ``install.sh`` / ``install.ps1`` and
    :func:`managed_python_env`) because repair must be able to cut over to a
    fresh generation without touching the interpreter the live venv was built
    on — and never reinstall in place.
    """
    root = Path(project_root) if project_root is not None else _PROJECT_ROOT
    return root / _RUNTIME_DIR_NAME / "python"


def managed_python_env(
    project_root: Path | None = None, *, install_dir: Path | None = None,
    base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Return a sanitized environment for Hermes-private uv Python commands."""
    target = (
        Path(install_dir)
        if install_dir is not None
        else managed_python_install_dir(project_root)
    )
    env = managed_uv_env(base_env=base_env)
    for key in (
        "CONDA_DEFAULT_ENV", "CONDA_PREFIX", "UV_PROJECT_ENVIRONMENT", "UV_NO_MANAGED_PYTHON",
        "UV_PYTHON", "UV_PYTHON_DOWNLOADS", "UV_SYSTEM_PYTHON", "VIRTUAL_ENV", "PYTHONHOME",
        "PYTHONPATH"):
        env.pop(key, None)
    env.update({
        "UV_MANAGED_PYTHON": "1",
        "UV_NO_CONFIG": "1",
        "UV_PYTHON_INSTALL_DIR": str(target),
    })
    return env


def _macos_sign_managed_python(python: Path) -> bool:
    """Give a newly downloaded managed Python a stable macOS code identity.

    python-build-standalone binaries are ad-hoc signed, so TCC sees a cdhash-only identity that
    changes every runtime generation; an identifier-pinned designated requirement keeps it stable
    without a Developer ID. Best effort: a missing/incompatible ``codesign`` must not block repair.
    """
    if platform.system() != "Darwin":
        return False
    codesign = shutil.which("codesign")
    if not codesign:
        logger.info("macOS codesign is unavailable; using the downloaded Python signature")
        return False
    requirement = f'=designated => identifier "{_MACOS_MANAGED_PYTHON_IDENTIFIER}"'
    try:
        sign = [
            codesign, "--force", "--deep", "--sign", "-", "--timestamp=none",
            "--identifier", _MACOS_MANAGED_PYTHON_IDENTIFIER,
            "--requirements", requirement, str(python)]
        verify = [codesign, "--verify", "--deep", "--strict", str(python)]
        steps = (
            (sign, "could not stably sign managed Python %s: %s", "codesign failed"),
            (verify, "macOS signature verification failed for managed Python %s: %s",
             "verification failed"))
        for cmd, warning, fallback in steps:
            result = subprocess.run(
                cmd, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace"
            )
            if result.returncode != 0:
                logger.warning(
                    warning, python, (result.stderr or result.stdout or fallback).strip())
                return False
        return True
    except Exception as exc:
        logger.warning("could not sign managed Python %s: %s", python, exc)
        return False


@dataclass(frozen=True)
class RuntimeRepairResult:
    """Outcome of a managed-runtime repair attempt."""

    status: str
    detail: str = ""
    sqlite_before: str = ""
    sqlite_after: str = ""
    backup_venv: Path | None = None

    @property
    def repaired(self) -> bool:
        return self.status == "repaired"


@dataclass(frozen=True)
class _RepairLock:
    path: Path
    fd: int


def _report_runtime_repair_failure(repair: RuntimeRepairResult) -> None:
    if repair.backup_venv is None:
        print("  ℹ Managed Python runtime was not replaced; "
              f"the existing venv is unchanged ({repair.detail}).")
        print("    Sessions stay protected meanwhile: Hermes keeps databases "
              "out of WAL mode on this SQLite build. The next `hermes update` "
              "will retry.")
        return
    print(f"  ✗ Managed Python runtime cutover needs manual recovery: {repair.detail}")
    print(f"    Previous venv: {repair.backup_venv}")


class _UvResult(str):
    """``ensure_uv()`` return value that survives an update boundary. POSIX only: a str subclass
    with an overridden ``__iter__`` is unsafe as a Windows subprocess argument."""

    fresh_bootstrap: bool

    def __new__(cls, path: Optional[str], fresh: bool = False) -> "_UvResult":
        self = super().__new__(cls, path or "")
        self.fresh_bootstrap = fresh
        return self

    def __iter__(self):
        # Tuple-unpacking hook for legacy ``uv_bin, fresh = ensure_uv()`` sites; the first
        # element keeps the historical contract (path string, or None when unavailable).
        return iter(((str(self) or None), self.fresh_bootstrap))


def _ensure_uv_path(
    *,
    repair_observer: Callable[[RuntimeRepairResult], None] | None = None,
) -> Optional[str]:
    """Resolve the managed uv path, installing it if necessary."""
    _migrate_legacy_managed_uv()
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
    result = resolve_uv()
    if result:
        print(f"  ✓ Managed uv installed ({_uv_version(result)})")
        # Compatibility boundary: an older, already-imported updater calls the freshly pulled
        # ``ensure_uv()``; repairing here lets that first update migrate a vulnerable runtime.
        _run_runtime_repair(result, repair_observer)
    else:
        print("  ✗ Managed uv install appeared to succeed but binary not found")
    return result


def _uv_version(uv_bin: str) -> str:
    return subprocess.run(
        [uv_bin, "--version"],
        capture_output=True, text=True, encoding='utf-8', errors='replace', check=False,
    ).stdout.strip()


def _run_runtime_repair(
    uv_bin: str, repair_observer: Callable[[RuntimeRepairResult], None] | None,
    *, print_skip: bool = False) -> None:
    """Run the vulnerable-runtime repair hook; never raises (repair is non-fatal)."""
    try:
        repair = repair_vulnerable_runtime(uv_bin)
        if repair_observer is not None:
            repair_observer(repair)
        if repair.status == "failed":
            _report_runtime_repair_failure(repair)
    except Exception as exc:
        logger.warning("Managed Python runtime repair failed: %s", exc)
        if print_skip:
            print(f"  ⚠ Managed Python runtime repair skipped: {exc}")


def ensure_uv(
    *,
    repair_observer: Callable[[RuntimeRepairResult], None] | None = None,
):
    """Return Hermes' managed uv path, installing it first if necessary.

    On POSIX the result is a :class:`_UvResult` (``str`` subclass) usable as the path *and*
    unpackable as ``(path, fresh_bootstrap)`` for older call sites.
    """
    result = _ensure_uv_path(repair_observer=repair_observer)
    if platform.system() == "Windows":
        # See _UvResult: the __iter__ override is unsafe as a Windows subprocess argument.
        return result
    return _UvResult(result)


def _managed_uv_refresh_stamp() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "cache" / ".uv_refresh_stamp"


def _managed_uv_refresh_is_fresh(now: float | None = None) -> bool:
    """Return True when the managed uv was refreshed recently enough to skip.

    uv releases roughly weekly while many users run ``hermes update`` daily;
    re-running the standalone installer (a ~30 MB download) on every
    invocation is waste and, offline, a hang risk. A stamp file under
    HERMES_HOME caches the last successful refresh time.
    """
    try:
        age = (now if now is not None else time.time()) - _managed_uv_refresh_stamp().stat().st_mtime
        return 0 <= age < UV_SELF_UPDATE_INTERVAL_SECONDS
    except Exception:
        return False


def _touch_managed_uv_refresh_stamp() -> None:
    with contextlib.suppress(OSError):
        stamp = _managed_uv_refresh_stamp()
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.touch()


# uv ships releases ~weekly; refresh the managed binary at most this often.
# (Name kept from the pre-isolation self-update era for API stability; the
# mechanism is now installer re-run, not ``uv self update``.)
UV_SELF_UPDATE_INTERVAL_SECONDS = 7 * 24 * 3600


def update_managed_uv(
    *, repair_observer: Callable[[RuntimeRepairResult], None] | None = None, force: bool = False
) -> Optional[str]:
    """Refresh Hermes' *private* uv by re-running the official installer.

    Call this during ``hermes update`` so the managed copy stays current.
    Resolution goes through ``resolve_uv()``, which is managed-only, so a
    toolchain Hermes does not own is never used or refreshed.  Returns the
    managed path when uv is available and ``None`` otherwise.

    The managed binary is installed with ``UV_UNMANAGED_INSTALL``: no install
    receipt is written, so uv itself refuses ``uv self update`` for it — and
    that refusal is exactly what keeps Hermes from ever touching a user's uv
    or user PATH/profile state.  Advancing Hermes' own copy therefore means
    re-running the official standalone installer into the private dir
    (:func:`_refresh_managed_binary`) under the same unmanaged contract — a
    bounded, Hermes-owned refresh that never writes user state.

    The refresh is skipped when one succeeded within the last
    ``UV_SELF_UPDATE_INTERVAL_SECONDS`` (7 days) unless ``force=True``; the
    vulnerable-runtime repair probe below ALWAYS runs — CVE-driven runtime
    repair must never be gated behind the freshness stamp.
    """
    # A pre-isolation install kept the managed binary in $HERMES_HOME/bin;
    # migrate it before resolving so a legacy install gets its runtime
    # repair on THIS update, not the next one (resolve_uv is a pure lookup).
    _migrate_legacy_managed_uv()
    existing = resolve_uv()
    if not existing:
        # Not installed yet — ensure_uv() will handle that elsewhere.
        return None
    if force or not _managed_uv_refresh_is_fresh():
        before = _uv_version_string(existing)
        changed = _refresh_managed_binary(existing)
        if changed:
            print(
                "  ✓ Managed uv refreshed "
                f"({before} → {_uv_version_string(existing)})"
            )
        # changed=False: the installer ran but upstream has no newer version
        # (or refresh failed — old uv still works, stamp left stale so the
        # next update retries).  Both are non-fatal by design.
    # Keep this hook inside the long-standing API: during an update main.py is already imported
    # from the old checkout and ``git pull`` replaces this module before the updater imports it,
    # so calling the repair here is what migrates the runtime on that first update. Non-fatal:
    # the live venv is untouched unless a fully prepared candidate reached cutover.
    _run_runtime_repair(existing, repair_observer, print_skip=True)
    return existing


def _reload_hermes_constants():
    """Re-execute ``hermes_constants`` from disk (the imported one may predate venv_python_path)."""
    import hermes_constants
    return importlib.reload(hermes_constants)


def _venv_python(venv_dir: Path) -> Path:
    try:
        from hermes_constants import venv_python_path
    except ImportError:
        venv_python_path = _reload_hermes_constants().venv_python_path
    return venv_python_path(venv_dir, windows=platform.system() == "Windows")


def _remove_tree(path: Path, *, boundary: Path) -> None:
    """Best-effort removal constrained to a known runtime boundary."""
    try:
        path.resolve().relative_to(boundary.resolve())
    except (OSError, ValueError):
        return
    shutil.rmtree(path, ignore_errors=True)


def _reject(path: Path, boundary: Path, msg: str, *args) -> None:
    """Log a rejected candidate and clean up its tree; always returns ``None``."""
    logger.warning(msg, *args)
    _remove_tree(path, boundary=boundary)
    return None


def _token() -> str:
    return f"{int(time.time())}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def _dotted(parts) -> str:
    return ".".join(str(p) for p in parts)


def _make_world_traversable(path: Path) -> None:
    """Keep root/FHS-managed runtimes executable by non-root callers."""
    with contextlib.suppress(OSError):
        path.chmod(path.stat().st_mode | 0o755)


def _runtime_request(info: SQLiteRuntimeInfo) -> str:
    """Pin the candidate to the current CPython minor line (e.g. ``3.11``): requesting the exact
    patch can never repair installs whose patch has no fixed-SQLite artifact at all."""
    return _dotted(info.python_version[:2])


# Cap on newer patches tried, newest-first, before giving up: each attempt is a real
# download+install+probe+delete cycle, and the fix is almost always in the next patch or two.
_MAX_PATCH_RETRIES = 5


def _list_available_patches(
    uv_bin: str, minor: str, *, cwd: Path, env: dict) -> list[tuple[int, int, int]]:
    """Known patch versions for ``minor`` (e.g. "3.11"), newest first; [] on any failure
    (network, parse), in which case callers fall back to the bare-minor request.

    Queries ``uv python list --all-versions`` rather than trusting the bare minor-line request to resolve to
    the newest patch (issue #71250: on some hosts/uv versions, the resolved candidate for a bare "3.11"
    request can be an older cached/indexed patch that still links a vulnerable SQLite, even when a newer
    non-vulnerable patch is available).
    """
    try:
        result = subprocess.run(
            [
                uv_bin, "python", "list", minor, "--all-versions", "--only-downloads",
                "--output-format", "json", "--no-config"],
            cwd=cwd, env=env, capture_output=True, text=True, check=False, timeout=15)
        if result.returncode != 0 or not result.stdout.strip():
            return []
        versions: list[tuple[int, int, int]] = []
        for entry in json.loads(result.stdout):
            if not isinstance(entry, dict):
                continue
            # Only default/cpython builds -- skip pypy/graalpy/freethreaded variants.
            if entry.get("implementation") not in (None, "cpython") or (
                entry.get("variant") not in (None, "default")):
                continue
            parts = entry.get("version_parts") or {}
            try:
                versions.append(
                    (int(parts["major"]), int(parts["minor"]), int(parts["patch"])))
            except (KeyError, TypeError, ValueError):
                continue
        # Deduplicate (a version can repeat across platforms/arches) and sort newest-first.
        return sorted(set(versions), reverse=True)
    except Exception:
        return []


def _attempt_install_generation(
    uv_bin: str, request: str, *, project_root: Path, python_root: Path,
    current: SQLiteRuntimeInfo, allow_minor_upgrade: bool = False,
    tried_versions: set[tuple[int, int, int]] | None = None) -> _Provisioned | None:
    """One install+probe attempt for ``request`` (bare minor "3.11" or explicit patch "3.11.15").

    Each attempt gets its own generation directory so a rejected candidate is fully cleaned up
    before the next attempt (--reinstall semantics). Returns None (and cleans up) on any failure.
    """
    generation = python_root / f"generation-{_token()}"
    generation.mkdir(parents=True, exist_ok=False)
    _make_world_traversable(generation)

    reject = partial(_reject, generation, python_root)
    env = managed_python_env(project_root, install_dir=generation)
    run = dict(cwd=project_root, env=env, capture_output=True, text=True, check=False)
    install = subprocess.run(
        [uv_bin, "python", "install", request, "--reinstall", "--no-bin", "--no-registry",
         "--no-config"],
        **run)
    if install.returncode != 0:
        return reject(
            "private Python install failed for %s (rc=%d): %s",
            request, install.returncode, (install.stderr or install.stdout or "").strip())
    found = subprocess.run(
        [uv_bin, "python", "find", request, "--managed-python", "--no-config"], **run)
    if found.returncode != 0 or not found.stdout.strip():
        return reject(
            "private Python lookup failed for %s (rc=%d): %s",
            request, found.returncode, (found.stderr or "").strip())
    python = Path(found.stdout.strip().splitlines()[-1])
    try:
        python.resolve().relative_to(generation.resolve())
    except (OSError, ValueError):
        return reject("uv resolved Python outside the Hermes generation: %s", python)
    # Sign before the candidate is probed or promoted so each immutable generation does not look
    # like a new TCC principal on macOS. Non-fatal: the SQLite repair proceeds regardless.
    _macos_sign_managed_python(python)
    candidate = probe_sqlite_runtime(python)
    if candidate is None:
        return reject("could not probe candidate Python runtime: %s", python)
    if tried_versions is not None:
        tried_versions.add(candidate.python_version[:3])
    if allow_minor_upgrade:
        # Falling forward to a higher minor line: only reject downgrades.
        if candidate.python_version < current.python_version:
            return reject(
                "candidate Python downgraded from %s: %s",
                _dotted(current.python_version), candidate.python_version)
    elif candidate.python_version[:2] != current.python_version[:2] or (
        candidate.python_version < current.python_version):
        return reject(
            "candidate Python drifted off the %s minor line or downgraded: %s",
            _dotted(current.python_version[:2]), candidate.python_version)
    if candidate.wal_reset_vulnerable:
        return reject(
            "candidate Python still links vulnerable SQLite %s (%s)",
            candidate.sqlite_version_string, candidate.sqlite_source_id)
    return generation, python, candidate


def _retry_explicit_patches(
    uv_bin: str, request: str, *, project_root: Path, python_root: Path,
    current: SQLiteRuntimeInfo, tried: set[tuple[int, int, int]],
    allow_minor_upgrade: bool = False, skip_at_or_below: tuple[int, int, int] | None = None,
) -> _Provisioned | None:
    """Retry ``request``'s minor line with explicit patches, newest-first, at most
    ``_MAX_PATCH_RETRIES`` attempts, skipping versions already in ``tried`` (a certain rejection
    still costs a full download+install+probe+delete cycle).

    ``skip_at_or_below`` also skips patches at or below that version: only NEWER patches can carry
    the fix and the downgrade guard rejects the rest; on a stale uv catalog the newest indexed
    patch can be the installed one, and the loop would burn every retry walking backwards.
    """
    # The bare minor-line request resolved to a still-vulnerable (or otherwise rejected) candidate. Rather
    # than giving up immediately, query which patches on this minor line uv actually knows about and retry
    # with explicit newer versions, newest-first -- this handles the case where the default resolution for a
    # bare request picks an older cached/indexed patch even though a newer, non-vulnerable one is available
    # (issue #71250).
    env_for_list = managed_python_env(project_root, install_dir=python_root)
    patches = _list_available_patches(uv_bin, request, cwd=project_root, env=env_for_list)
    attempts = 0
    for version_tuple in patches:
        if attempts >= _MAX_PATCH_RETRIES:
            break
        if version_tuple in tried:
            continue
        if skip_at_or_below is not None and version_tuple <= skip_at_or_below:
            continue
        tried.add(version_tuple)
        explicit = _dotted(version_tuple)
        print(f"  → Retrying with explicit patch {explicit}...")
        attempts += 1
        result = _attempt_install_generation(
            uv_bin, explicit, project_root=project_root,
            python_root=python_root, current=current,
            allow_minor_upgrade=allow_minor_upgrade)
        if result is not None:
            return result
    return None


def _provision_line(
    uv_bin: str, request: str, *, tried: set[tuple[int, int, int]],
    allow_minor_upgrade: bool = False, skip_at_or_below: tuple[int, int, int] | None = None,
    **common) -> _Provisioned | None:
    """Try ``request`` once, then its explicit newer patches; None when the whole line fails."""
    result = _attempt_install_generation(
        uv_bin, request, tried_versions=tried, allow_minor_upgrade=allow_minor_upgrade, **common)
    if result is None:
        result = _retry_explicit_patches(
            uv_bin, request, tried=tried, allow_minor_upgrade=allow_minor_upgrade,
            skip_at_or_below=skip_at_or_below, **common)
    return result


def _install_safe_python_generation(
    uv_bin: str, *, project_root: Path, current: SQLiteRuntimeInfo) -> _Provisioned | None:
    runtime_root = project_root / _RUNTIME_DIR_NAME
    python_root = managed_python_install_dir(project_root)
    _make_world_traversable(runtime_root)
    _make_world_traversable(python_root)
    common = dict(project_root=project_root, python_root=python_root, current=current)

    request = _runtime_request(current)
    print(f"  → Provisioning a private Python {request} runtime with fixed SQLite...")
    tried_versions = {current.python_version[:3]}
    # If the bare minor-line request resolves to a still-vulnerable (or otherwise rejected)
    # candidate, the default resolution may have picked an older cached/indexed patch even though
    # a newer, non-vulnerable one exists: retry with explicit newer patches, newest-first.
    result = _provision_line(
        uv_bin, request, tried=tried_versions, skip_at_or_below=current.python_version[:3], **common
    )
    if result is not None:
        return result
    # All patches on the current minor line are vulnerable or rejected. Fall forward to the next
    # supported minor (e.g. 3.11 → 3.12) so the user isn't stuck on every `hermes update`. The
    # requires-python window (>=3.11,<3.14) and the import smoke-test gate compatibility.
    # See #76106.
    cur_major, cur_minor = current.python_version[:2]
    fb_tried: set[tuple[int, int, int]] = set(tried_versions)
    for next_minor in range(cur_minor + 1, 14):  # up to 3.13
        next_request = f"{cur_major}.{next_minor}"
        print(
            f"  → No fixed {cur_major}.{cur_minor} build available; "
            f"trying {next_request} as fallback...")
        result = _provision_line(
            uv_bin, next_request, tried=fb_tried, allow_minor_upgrade=True, **common)
        if result is not None:
            return result
    return None


def _smoke_candidate_venv(venv_dir: Path) -> tuple[bool, str, SQLiteRuntimeInfo | None]:
    """Exercise the candidate interpreter and imports through its real path."""
    python = _venv_python(venv_dir)
    info = probe_sqlite_runtime(python)
    if info is None:
        return False, f"could not execute {python}", None
    if info.wal_reset_vulnerable:
        return False, f"candidate still links vulnerable SQLite {info.sqlite_version_string}", info
    check = (
        "import dotenv, fastapi, openai, prompt_toolkit, pydantic, rich, uvicorn, yaml\n"
        "import hermes_state\n")
    try:
        result = subprocess.run(
            [str(python), "-I", "-c", check], cwd=venv_dir.parent, env=isolated_interpreter_env(),
            capture_output=True, text=True, timeout=90, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc), info
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "core import smoke failed").strip()
        return False, detail.splitlines()[-1] if detail else "core import smoke failed", info
    return True, "", info


def _stage_candidate_venv(
    uv_bin: str, *, project_root: Path, generation: Path, python: Path) -> Path | None:
    runtime_root = project_root / _RUNTIME_DIR_NAME
    candidate = runtime_root / f"venv-candidate-{_token()}"
    env = managed_python_env(project_root, install_dir=generation)
    env.update({
        "UV_PROJECT_ENVIRONMENT": str(candidate), "UV_PYTHON": str(python),
        "UV_PYTHON_DOWNLOADS": "never", "VIRTUAL_ENV": str(candidate)})

    reject = partial(_reject, candidate, runtime_root)
    print("  → Building a relocatable replacement environment...")
    created = subprocess.run(
        [
            uv_bin, "venv", str(candidate), "--python", str(python),
            "--managed-python", "--no-python-downloads", "--relocatable", "--no-config"],
        cwd=project_root, env=env, capture_output=True, text=True, check=False)
    if created.returncode != 0:
        return reject(
            "candidate venv creation failed (rc=%d): %s",
            created.returncode, (created.stderr or created.stdout or "").strip())
    if not (project_root / "uv.lock").is_file():
        return reject("candidate dependency sync refused: uv.lock is missing")
    # Locked sync must see project [tool.uv] exclude-newer; --no-config / UV_NO_CONFIG drops it
    # and uv 0.12+ refuses --locked.
    sync_env = dict(env)
    sync_env.pop("UV_NO_CONFIG", None)
    synced = subprocess.run(
        [uv_bin, "sync", "--extra", "all", "--locked", "--python", str(_venv_python(candidate))],
        cwd=project_root, env=sync_env, check=False)
    if synced.returncode != 0:
        return reject("candidate dependency sync failed (rc=%d)", synced.returncode)
    healthy, detail, _ = _smoke_candidate_venv(candidate)
    if not healthy:
        return reject("candidate venv smoke failed: %s", detail)
    return candidate


def _rename_with_retry(source: Path, destination: Path) -> None:
    for delay in (0.0, 0.1, 0.25, 0.5, 1.0):
        if delay:
            time.sleep(delay)
        try:
            source.rename(destination)
            return
        except OSError as exc:
            last_error = exc
    raise last_error


def _cut_over_candidate(
    candidate: Path, *, project_root: Path, live: Path | None = None
) -> tuple[bool, Path | None, SQLiteRuntimeInfo | None, str]:
    live = live if live is not None else project_root / _VENV_NAME
    runtime_root = project_root / _RUNTIME_DIR_NAME
    token = _token()
    backup = live.with_name(f"{live.name}.stale.runtime-{token}")
    rejected = runtime_root / f"venv-rejected-{token}"
    try:
        try:
            _rename_with_retry(live, backup)
        except OSError as exc:
            return False, None, None, f"could not park the existing venv: {exc}"
        try:
            _rename_with_retry(candidate, live)
        except OSError as promote_error:
            try:
                _rename_with_retry(backup, live)
            except OSError as rollback_error:
                return False, backup, None, (
                    "could not promote the replacement venv "
                    f"({promote_error}); rollback failed ({rollback_error})")
            return False, None, None, f"could not promote the replacement venv: {promote_error}"
        try:
            healthy, detail, info = _smoke_candidate_venv(live)
        except Exception as exc:
            healthy, detail, info = False, f"candidate smoke raised: {exc}", None
        if healthy:
            return True, backup, info, ""
        try:
            _rename_with_retry(live, rejected)
            _rename_with_retry(backup, live)
        except OSError as exc:
            return False, backup, info, (
                "post-cutover smoke failed "
                f"({detail}); rollback failed ({exc}); rejected venv: {rejected}")
        _remove_tree(rejected, boundary=runtime_root)
        return False, None, info, f"post-cutover smoke failed: {detail}"
    except BaseException:
        if not live.exists() and backup.exists():
            try:
                _rename_with_retry(backup, live)
            except OSError as exc:
                logger.error(
                    "interrupted runtime cutover could not restore %s from %s: %s",
                    live, backup, exc)
        raise


def _acquire_repair_lock(runtime_root: Path) -> _RepairLock | None:
    """Acquire an OS-held install lock that is released on process exit."""
    runtime_root.mkdir(parents=True, exist_ok=True)
    _make_world_traversable(runtime_root)
    path = runtime_root / _REPAIR_LOCK_NAME
    try:
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    except OSError:
        return None
    try:
        _flock(fd, acquire=True)
    except (ImportError, OSError):
        os.close(fd)
        return None
    return _RepairLock(path=path, fd=fd)


def _flock(fd: int, *, acquire: bool) -> None:
    """Non-blocking exclusive lock (or unlock) on *fd*, portable across msvcrt/fcntl."""
    if os.name == "nt":
        import msvcrt
        if acquire and os.fstat(fd).st_size == 0:
            os.write(fd, b"\0")
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK if acquire else msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(fd, (fcntl.LOCK_EX | fcntl.LOCK_NB) if acquire else fcntl.LOCK_UN)


def _release_repair_lock(lock: _RepairLock) -> None:
    try:
        with contextlib.suppress(ImportError, OSError):
            _flock(lock.fd, acquire=False)
    finally:
        with contextlib.suppress(OSError):
            os.close(lock.fd)


def _windows_runtime_holders() -> tuple[bool, str]:
    if platform.system() != "Windows":
        return False, ""
    main_module = sys.modules.get("hermes_cli.main")
    detector = getattr(main_module, "_detect_venv_python_processes", None)
    if detector is None:
        return True, "cannot verify Windows venv holders from this update context"
    try:
        holders = detector()
    except Exception as exc:
        return True, f"could not verify Windows venv holders: {exc}"
    if holders:
        pids = ", ".join(str(item[0]) for item in holders[:6])
        return True, f"other Hermes processes still hold the venv (PID {pids})"
    return False, ""


def _windows_runtime_self_lock(live: Path) -> tuple[bool, str]:
    """Detect the one holder the generic scan is blind to: THIS process.

    ``_detect_venv_python_processes`` excludes the calling process and its ancestors on purpose
    (``hermes update`` itself runs from the venv python), which is correct for the dependency-sync
    path where only a *loaded* ``.pyd`` image blocks the rewrite and a fresh child dodges it.

    For the whole-venv park rename that exemption is fatal: Windows keeps the image of any executable a
    running process was started from mapped until that process exits, so a directory containing the
    updater's own ``python.exe`` (or a waiting ``hermes.exe`` launcher ancestor) can never be renamed from
    inside the updater. The retry loop in ``_cut_over_candidate`` cannot help against that — the lock is
    structural, not transient (#93032).
    """
    if platform.system() != "Windows":
        return False, ""
    try:
        live_res = str(live.resolve())
    except OSError:
        live_res = str(live)
    live_res = live_res.lower().rstrip(os.sep) + os.sep

    def _under_live(path_value: str | None) -> bool:
        if not path_value:
            return False
        try:
            resolved = str(Path(path_value).resolve()).lower()
        except (OSError, ValueError):
            resolved = str(path_value).lower()
        return resolved.startswith(live_res)

    why = "Windows cannot rename a directory while a process executes from inside it"
    exe = sys.executable
    if _under_live(exe):
        return True, f"the updater itself runs from the live venv it must replace ({exe}); {why}"
    # Belt-and-braces: the venv\Scripts\hermes.exe launcher stays mapped while it waits for this
    # child, so an ancestor started from the venv blocks the rename too.
    with contextlib.suppress(Exception):
        import psutil
        for anc in psutil.Process().parents():
            try:
                anc_exe = anc.exe()
            except Exception:
                continue
            if _under_live(anc_exe):
                return True, (
                    f"ancestor process PID {anc.pid} runs from the live venv ({anc_exe}); {why}")
    return False, ""


def _uv_version_string(uv_bin: str) -> str:
    """Return ``uv --version`` output, or ``""`` when it cannot be read."""
    try:
        result = subprocess.run(
            [uv_bin, "--version"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            check=False, timeout=15)
    except Exception:
        return ""
    return (result.stdout or "").strip() if result.returncode == 0 else ""


def _refresh_managed_binary(uv_bin: str) -> bool:
    """Re-run the official installer over the managed uv binary.

    The managed uv is installed with ``UV_UNMANAGED_INSTALL``: no install
    receipt is written, so uv itself refuses ``uv self update`` for it — which
    is exactly what keeps Hermes from ever bumping a user's uv or writing
    user PATH/profile state.  Re-running the official standalone installer
    into the private dir is therefore the ONLY way to advance Hermes' own
    copy, and the only supported refresh path for unmanaged installs — this
    is it (called from ``update_managed_uv`` on a throttle, and from runtime
    repair when provisioning needs a newer catalog).

    Re-running also refreshes the embedded python-build-standalone download
    catalog, which otherwise stays frozen at bootstrap age:
    python-build-standalone re-releases existing CPython patch versions with
    newer SQLite (e.g. the 3.11.15 build was re-cut with SQLite 3.53.x), so a
    stale catalog can make every provisioning attempt resolve to a vulnerable
    build even though a fixed build of the SAME patch version exists (issue
    #72093).

    Only the Hermes-managed binary is refreshed: a caller that somehow passed
    a foreign uv path is left alone (no download), so this never touches a
    toolchain Hermes does not own.

    The throttle stamp is touched only when the installer actually ran and
    produced a runnable binary: a flaky network leaves it stale so the next
    update retries, while an unchanged upstream version (installer ran, same
    version) does not re-download on every update.

    Returns ``True`` when the binary's version actually changed — e.g. a
    provisioning retry can now see a different catalog.  ``False`` means a
    retry would resolve identically and is not worth the download cycle (or
    the refresh failed; the old binary still works).
    """
    managed = managed_uv_path()
    try:
        if Path(uv_bin).resolve() != managed.resolve():
            return False
    except OSError:
        return False
    before = _uv_version_string(uv_bin)
    try:
        _install_uv(managed)
        after = _uv_version_string(managed)
    except Exception as exc:
        logger.warning("managed uv refresh failed: %s", exc)
        return False
    if not after:
        logger.warning(
            "managed uv refresh did not produce a runnable binary at %s", managed
        )
        return False
    _touch_managed_uv_refresh_stamp()
    return after != before


def _default_live_venv(root: Path) -> Path:
    """Venv that runtime repair should target for *root*: ``venv`` when it holds an interpreter
    (managed layout wins), else ``.venv`` when that does, else ``venv`` so ``not-applicable`` fires.
    """
    primary, fallback = root / _VENV_NAME, root / _ALT_VENV_NAME
    use_fallback = not _venv_python(primary).is_file() and _venv_python(fallback).is_file()
    return fallback if use_fallback else primary


def _sweep_stale_runtime_backups(
    live: Path, *, root: Path, keep: Path | None = None, min_age_seconds: float = 3600.0) -> None:
    """Remove leftover ``venv.stale.runtime-*`` backups next to *live*. Best-effort: never raises.

    On POSIX this is safe while an older process still maps files from the tree (open FDs/mmaps
    keep their inodes). ``min_age_seconds`` avoids racing a concurrent repair whose fresh backup
    may still be its rollback path; ``keep`` exempts the backup this repair just created.

    A successful runtime repair parks the previous venv as ``<live>.stale.runtime-<token>``; historically
    nothing ever reclaimed those, so each repair leaked a full venv (~1 GB) at the project root forever
    (issue #73109).
    """
    try:
        candidates = list(live.parent.glob(f"{live.name}.stale.runtime-*"))
    except OSError:
        return
    now = time.time()
    for candidate in candidates:
        if keep is not None and candidate == keep:
            continue
        try:
            if now - candidate.stat().st_mtime < min_age_seconds:
                continue
        except OSError:
            continue
        _remove_tree(candidate, boundary=root)


def _result(
    status: str, current: SQLiteRuntimeInfo, detail: str = "", **extra) -> RuntimeRepairResult:
    return RuntimeRepairResult(status, detail, sqlite_before=current.sqlite_version_string, **extra)


def _repair_windows_preflight(
    root: Path, live: Path, current: SQLiteRuntimeInfo) -> RuntimeRepairResult | None:
    """Defer the repair when Windows holders make the venv rename impossible; else ``None``."""
    blocked, detail = _windows_runtime_holders()
    if blocked:
        print(f"  ⚠ SQLite runtime repair deferred: {detail}")
        return _result("skipped", current, detail)
    self_locked, self_detail = _windows_runtime_self_lock(live)
    if self_locked:
        # Structural, not transient: this process maps the live venv's own executable, so the
        # park rename fails identically on every run. Defer BEFORE provisioning — a candidate
        # staged for a cutover that can never run only leaks an incomplete generation.
        for line in (
            f"  ⚠ SQLite runtime repair deferred: {self_detail}.",
            # See #93032.
            "    Retrying `hermes update` from inside this venv cannot help: "
            "the mapped executable is released only when this process exits.",
            "    To complete the repair, run the updater from an interpreter "
            "that lives outside this venv, e.g.:",
            f"      cd {root}",
            "      <system Python> -m hermes_cli.main update",
            "    Sessions stay protected meanwhile: Hermes keeps databases "
            "out of WAL mode on this SQLite build."):
            print(line)
        return _result("skipped", current, self_detail)
    return None


def _repair_under_lock(
    uv_bin: str, *, root: Path, live: Path, live_python: Path, runtime_root: Path
) -> RuntimeRepairResult:
    """Provision, stage and cut over a fixed runtime; caller holds the repair lock."""
    # Re-probe under the install-scoped lock: another updater may have completed the repair
    # while this process was entering the path.
    current = probe_sqlite_runtime(live_python)
    if current is None:
        return RuntimeRepairResult("skipped", "live interpreter probe failed")
    if not current.wal_reset_vulnerable:
        return _result("safe", current, sqlite_after=current.sqlite_version_string)
    print(
        "  ⚠ Hermes venv links SQLite "
        f"{current.sqlite_version_string}, which has the WAL-reset bug.")
    provisioned = _install_safe_python_generation(uv_bin, project_root=root, current=current)
    # Likely a stale managed-uv catalog: python-build-standalone re-releases the same patch
    # versions with fixed SQLite, but a frozen catalog keeps resolving the old vulnerable build
    # and the patch-retry loop has no newer number to try. Refresh the binary and retry once.
    if provisioned is None and _refresh_managed_binary(uv_bin):
        # See #72093.
        print("  → Managed uv refreshed; retrying provisioning...")
        provisioned = _install_safe_python_generation(uv_bin, project_root=root, current=current)
    if provisioned is None:
        return _result("failed", current, "could not provision a fixed private Python runtime")
    generation, python, candidate_info = provisioned

    candidate = _stage_candidate_venv(
        uv_bin, project_root=root, generation=generation, python=python)
    if candidate is None:
        _remove_tree(generation, boundary=managed_python_install_dir(root))
        return _result(
            "failed", current,
            "replacement environment did not pass dependency and import smoke tests",
            sqlite_after=candidate_info.sqlite_version_string)

    cut_over, backup, final_info, cutover_detail = _cut_over_candidate(
        candidate, project_root=root, live=live)
    if not cut_over:
        if backup is None:
            _remove_tree(candidate, boundary=runtime_root)
            _remove_tree(generation, boundary=managed_python_install_dir(root))
        return _result(
            "failed", current, cutover_detail,
            sqlite_after=final_info.sqlite_version_string if final_info is not None else "",
            backup_venv=backup)
    final_version = (final_info if final_info is not None else candidate_info).sqlite_version_string
    print(
        "  ✓ Managed Python runtime repaired "
        f"(SQLite {current.sqlite_version_string} → {final_version})")
    if backup is not None and backup.exists():
        _remove_tree(backup, boundary=root)
    return _result("repaired", current, sqlite_after=final_version, backup_venv=backup)


def repair_vulnerable_runtime(
    uv_bin: str, *, project_root: Path | None = None, venv_dir: Path | None = None
) -> RuntimeRepairResult:
    """Replace a vulnerable install venv without mutating it in place.

    Every failure before cutover leaves the live venv untouched. Rename or post-cutover smoke
    failures restore the parked venv synchronously.
    """
    root = Path(project_root) if project_root is not None else _PROJECT_ROOT
    live = Path(venv_dir) if venv_dir is not None else _default_live_venv(root)
    live_python = _venv_python(live)
    if not (root / "pyproject.toml").is_file() or not live_python.is_file():
        return RuntimeRepairResult("not-applicable")
    current = probe_sqlite_runtime(live_python)
    if current is None:
        return RuntimeRepairResult("skipped", f"could not probe live interpreter {live_python}")
    if not current.wal_reset_vulnerable:
        # Already fixed: any venv.stale.runtime-* markers next to the live venv are leftovers
        # from a past repair and will never be rolled back to. Sweep them so they don't leak
        # ~1 GB each forever. Age-gated to avoid racing an in-flight repair in a sibling process.
        # See #73109.
        _sweep_stale_runtime_backups(live, root=root)
        return _result("safe", current, sqlite_after=current.sqlite_version_string)
    deferred = _repair_windows_preflight(root, live, current)
    if deferred is not None:
        return deferred
    runtime_root = root / _RUNTIME_DIR_NAME
    lock = _acquire_repair_lock(runtime_root)
    if lock is None:
        detail = "another runtime repair is already in progress"
        print(f"  ⚠ SQLite runtime repair deferred: {detail}")
        return _result("skipped", current, detail)
    try:
        return _repair_under_lock(
            uv_bin, root=root, live=live, live_python=live_python, runtime_root=runtime_root)
    finally:
        _release_repair_lock(lock)


def _install_uv(target: Path) -> None:
    """Bootstrap uv into *target* using the official standalone installer.

    Sets BOTH ``UV_UNMANAGED_INSTALL`` and ``UV_INSTALL_DIR`` on every
    platform.  ``UV_INSTALL_DIR`` picks the install location (the private
    managed dir, ``$HERMES_HOME/uv``, instead of ``~/.local/bin/``);
    ``UV_UNMANAGED_INSTALL`` is the load-bearing isolation switch — without
    it the astral installer ALSO prepends the install dir to the user
    PATH / shell profiles on a fresh install (see install.sh and install.ps1
    for the same invariant).  Never drop ``UV_UNMANAGED_INSTALL`` on either
    platform.
    """
    system = platform.system()
    env = {
        **os.environ,
        # Tell the astral installer to drop the binary in our dir, not
        # ~/.local/bin.  BOTH vars are set on every platform: UV_INSTALL_DIR
        # controls the location, while UV_UNMANAGED_INSTALL is what stops the
        # installer from writing the dir into the user PATH / shell profiles
        # (and marks the install unmanaged, disabling `uv self update`).
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
            check=True, capture_output=True)
        subprocess.run(["sh", installer_path], env=env, check=True, capture_output=True)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(installer_path)


def _install_uv_windows(env: dict[str, str]) -> None:
    """Invoke the PowerShell installer."""
    cmd = "irm https://astral.sh/uv/install.ps1 | iex"
    subprocess.run(
        ["powershell", "-ExecutionPolicy", "Bypass", "-c", cmd], env=env, check=True,
        capture_output=True)


def rebuild_venv(uv_bin: str, venv_dir: Path, python_version: str = "3.11") -> bool:
    True  # dont remove me. ask ethernet
