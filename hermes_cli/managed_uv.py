"""Hermes-managed uv and Python runtime repair.

Hermes owns its own uv binary at ``$HERMES_HOME/uv/uv`` (or ``uv.exe`` on
Windows). Every code path that needs uv resolves it from that single private
location. If the binary is missing, ``ensure_uv()`` bootstraps it via the
official standalone installer. The private directory is never added to PATH,
so Hermes cannot shadow a user's uv in interactive shells.

Legacy installs that placed the managed uv at ``$HERMES_HOME/bin/uv`` (the
pre-isolation layout — bin is a persisted User PATH entry on Windows) are
migrated to the private location on first use.

The Python backing the install is different: it is shared by every Hermes
profile because the checkout's ``venv`` is shared.  Runtime repair therefore
uses an install-scoped store under ``<checkout>/.hermes-runtime/python``. A
vulnerable interpreter is never reinstalled in place. We provision a new
immutable Python generation, build and smoke-test a relocatable sibling venv,
then cut over with same-filesystem renames. The old venv remains available for
synchronous rollback and is parked for cleanup after the updating process
releases it.
"""

from __future__ import annotations

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
from pathlib import Path
from typing import Callable, Optional

from hermes_constants import get_hermes_home
from hermes_cli.sqlite_runtime import SQLiteRuntimeInfo, probe_sqlite_runtime

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_RUNTIME_DIR_NAME = ".hermes-runtime"
_VENV_NAME = "venv"
_ALT_VENV_NAME = ".venv"
_REPAIR_LOCK_NAME = "runtime-repair.lock"
_MACOS_MANAGED_PYTHON_IDENTIFIER = "com.nousresearch.hermes.managed-python"

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


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
    if p.is_file() and os.access(p, os.X_OK):
        return str(p)
    return None


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
    project_root: Path | None = None,
    *,
    install_dir: Path | None = None,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return a sanitized environment for Hermes-private uv Python commands."""
    target = (
        Path(install_dir)
        if install_dir is not None
        else managed_python_install_dir(project_root)
    )
    env = managed_uv_env(base_env=base_env)
    for key in (
        "CONDA_DEFAULT_ENV",
        "CONDA_PREFIX",
        "UV_PROJECT_ENVIRONMENT",
        "UV_NO_MANAGED_PYTHON",
        "UV_PYTHON",
        "UV_PYTHON_DOWNLOADS",
        "UV_SYSTEM_PYTHON",
        "VIRTUAL_ENV",
        "PYTHONHOME",
        "PYTHONPATH",
    ):
        env.pop(key, None)
    env.update({
        "UV_MANAGED_PYTHON": "1",
        "UV_NO_CONFIG": "1",
        "UV_PYTHON_INSTALL_DIR": str(target),
    })
    return env


def _macos_sign_managed_python(python: Path) -> bool:
    """Give a newly downloaded managed Python a stable macOS code identity.

    python-build-standalone binaries are ad-hoc signed, which leaves macOS
    TCC with a cdhash-only identity that changes whenever Hermes provisions a
    new runtime generation.  An identifier-pinned designated requirement
    gives those generations a stable identity even when no Developer ID
    certificate is available locally.

    Signing is deliberately best effort.  Runtime repair exists to remove a
    security vulnerability, so an unavailable or incompatible ``codesign``
    must not prevent the fixed interpreter from being installed.
    """
    if platform.system() != "Darwin":
        return False

    codesign = shutil.which("codesign")
    if not codesign:
        logger.info(
            "macOS codesign is unavailable; using the downloaded Python signature"
        )
        return False

    requirement = (
        "=designated => identifier "
        f'"{_MACOS_MANAGED_PYTHON_IDENTIFIER}"'
    )
    try:
        signed = subprocess.run(
            [
                codesign,
                "--force",
                "--deep",
                "--sign",
                "-",
                "--timestamp=none",
                "--identifier",
                _MACOS_MANAGED_PYTHON_IDENTIFIER,
                "--requirements",
                requirement,
                str(python),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if signed.returncode != 0:
            logger.warning(
                "could not stably sign managed Python %s: %s",
                python,
                (signed.stderr or signed.stdout or "codesign failed").strip(),
            )
            return False

        verified = subprocess.run(
            [codesign, "--verify", "--deep", "--strict", str(python)],
            check=False,
            capture_output=True,
            text=True,
        )
        if verified.returncode != 0:
            logger.warning(
                "macOS signature verification failed for managed Python %s: %s",
                python,
                (verified.stderr or verified.stdout or "verification failed").strip(),
            )
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
        print(
            "  ℹ Managed Python runtime was not replaced; "
            f"the existing venv is unchanged ({repair.detail})."
        )
        print(
            "    Sessions stay protected meanwhile: Hermes keeps databases "
            "out of WAL mode on this SQLite build. The next `hermes update` "
            "will retry."
        )
        return
    print(f"  ✗ Managed Python runtime cutover needs manual recovery: {repair.detail}")
    print(f"    Previous venv: {repair.backup_venv}")


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

    # Verify
    result = resolve_uv()
    if result:
        version = subprocess.run(
            [result, "--version"],
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            check=False,
        ).stdout.strip()
        print(f"  ✓ Managed uv installed ({version})")
        # Compatibility boundary: an older, already-imported updater calls the
        # freshly pulled ``ensure_uv()`` after bootstrapping uv.  Repair here so
        # that first update can migrate a vulnerable runtime without requiring
        # a second ``hermes update``.
        try:
            repair = repair_vulnerable_runtime(result)
            if repair_observer is not None:
                repair_observer(repair)
            if repair.status == "failed":
                _report_runtime_repair_failure(repair)
        except Exception as exc:
            logger.warning("Managed Python runtime repair failed: %s", exc)
    else:
        print("  ✗ Managed uv install appeared to succeed but binary not found")
    return result


def ensure_uv(
    *,
    repair_observer: Callable[[RuntimeRepairResult], None] | None = None,
):
    """Return Hermes' managed uv path, installing it first if necessary.

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
    pip gracefully. ``repair_observer``, when provided, receives the runtime
    repair result produced after a fresh uv bootstrap.
    """
    result = _ensure_uv_path(repair_observer=repair_observer)
    if platform.system() == "Windows":
        # See docstring: a str subclass with an overridden __iter__ is unsafe as
        # a Windows subprocess argument. Hand back the plain path (or None).
        return result
    return _UvResult(result)


def _managed_uv_refresh_is_fresh(now: float | None = None) -> bool:
    """Return True when the managed uv was refreshed recently enough to skip.

    uv releases roughly weekly while many users run ``hermes update`` daily;
    re-running the standalone installer (a ~30 MB download) on every
    invocation is waste and, offline, a hang risk. A stamp file under
    HERMES_HOME caches the last successful refresh time.
    """
    try:
        from hermes_constants import get_hermes_home

        stamp = get_hermes_home() / "cache" / ".uv_refresh_stamp"
        age = (now if now is not None else time.time()) - stamp.stat().st_mtime
        return 0 <= age < UV_SELF_UPDATE_INTERVAL_SECONDS
    except Exception:
        return False


def _touch_managed_uv_refresh_stamp() -> None:
    try:
        from hermes_constants import get_hermes_home

        stamp = get_hermes_home() / "cache" / ".uv_refresh_stamp"
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.touch()
    except OSError:
        pass


# uv ships releases ~weekly; refresh the managed binary at most this often.
# (Name kept from the pre-isolation self-update era for API stability; the
# mechanism is now installer re-run, not ``uv self update``.)
UV_SELF_UPDATE_INTERVAL_SECONDS = 7 * 24 * 3600


def update_managed_uv(
    *,
    repair_observer: Callable[[RuntimeRepairResult], None] | None = None,
    force: bool = False,
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

    if existing and (force or not _managed_uv_refresh_is_fresh()):
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

    if not existing:
        # No uv at all yet — ensure_uv() will handle provisioning elsewhere.
        return None

    # Keep this hook inside the long-standing API. During an update, main.py is
    # already imported from the old checkout, then ``git pull`` replaces this
    # module on disk before the updater imports it. Calling the repair here is
    # what makes the migration happen on that first update.
    try:
        repair = repair_vulnerable_runtime(existing)
        if repair_observer is not None:
            repair_observer(repair)
        if repair.status == "failed":
            _report_runtime_repair_failure(repair)
    except Exception as exc:
        # Runtime refresh is deliberately non-fatal. The live venv was not
        # touched unless a fully prepared candidate reached cutover.
        logger.warning("Managed Python runtime repair failed: %s", exc)
        print(f"  ⚠ Managed Python runtime repair skipped: {exc}")
    return existing


# ---------------------------------------------------------------------------
# Managed Python runtime repair
# ---------------------------------------------------------------------------


def _reload_hermes_constants():
    """Re-execute ``hermes_constants`` from disk and return the fresh module.

    ``hermes update`` imports ``hermes_constants`` from the OLD checkout,
    ``git pull`` then replaces that file, and this freshly-pulled module runs
    its lazy imports against the module object Python already cached in
    ``sys.modules`` — the pre-upgrade one. A symbol added by the update is
    absent there while the file named in the resulting ``ImportError`` plainly
    contains it, which is what made this read as a contradiction:

        cannot import name 'venv_python_path' from 'hermes_constants'
        (~/.hermes/hermes-agent/hermes_constants.py)

    Reloading picks up the definitions actually on disk, so callers keep using
    the shared helper instead of hand-rolling a second copy of its logic. Same
    update-boundary class as the ``ensure_uv()`` arity skew on :class:`_UvResult`.
    """
    import hermes_constants

    return importlib.reload(hermes_constants)


def _venv_python(venv_dir: Path) -> Path:
    windows = platform.system() == "Windows"
    try:
        from hermes_constants import venv_python_path
    except ImportError:
        venv_python_path = _reload_hermes_constants().venv_python_path
    return venv_python_path(venv_dir, windows=windows)


def _remove_tree(path: Path, *, boundary: Path) -> None:
    """Best-effort removal constrained to a known runtime boundary."""
    try:
        path.resolve().relative_to(boundary.resolve())
    except (OSError, ValueError):
        return
    shutil.rmtree(path, ignore_errors=True)


def _make_world_traversable(path: Path) -> None:
    """Keep root/FHS-managed runtimes executable by non-root callers."""
    try:
        path.chmod(path.stat().st_mode | 0o755)
    except OSError:
        pass


def _runtime_request(info: SQLiteRuntimeInfo) -> str:
    """Pin the candidate to the current CPython minor line (e.g. ``3.11``).

    Requesting the exact patch can never repair some installs: for a given
    patch, python-build-standalone may have no artifact with fixed SQLite at
    all (e.g. every published 3.11.14 build links SQLite 3.50.4; the fix
    only exists from 3.11.15).  A newer patch on the same minor is what
    ``uv python install`` would resolve for a fresh install, stays inside
    ``requires-python``, and the locked ``uv sync`` + import smoke tests gate
    compatibility before any cutover.
    """
    return ".".join(str(part) for part in info.python_version[:2])


# Cap on how many newer patches we'll try, newest-first, before giving up.
# Bounded because each attempt is a real download+install+probe+delete cycle;
# in practice the fix is almost always in the very next patch or two.
_MAX_PATCH_RETRIES = 5


def _list_available_patches(
    uv_bin: str, minor: str, *, cwd: Path, env: dict
) -> list[tuple[int, int, int]]:
    """Return known patch versions for ``minor`` (e.g. "3.11"), newest first.

    Queries ``uv python list --all-versions`` rather than trusting the bare
    minor-line request to resolve to the newest patch (issue #71250: on some
    hosts/uv versions, the resolved candidate for a bare "3.11" request can
    be an older cached/indexed patch that still links a vulnerable SQLite,
    even when a newer non-vulnerable patch is available). Returns [] on any
    failure (network, parse) -- callers fall back to the original bare-minor
    request in that case, preserving prior behavior.
    """
    try:
        result = subprocess.run(
            [
                uv_bin, "python", "list", minor,
                "--all-versions", "--only-downloads",
                "--output-format", "json", "--no-config",
            ],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []
        entries = json.loads(result.stdout)
        versions: list[tuple[int, int, int]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            # Only default/cpython builds -- skip pypy/graalpy/freethreaded
            # variants, which aren't what this repair path wants.
            if entry.get("implementation") not in (None, "cpython"):
                continue
            if entry.get("variant") not in (None, "default"):
                continue
            parts = entry.get("version_parts") or {}
            try:
                versions.append(
                    (int(parts["major"]), int(parts["minor"]), int(parts["patch"]))
                )
            except (KeyError, TypeError, ValueError):
                continue
        # Deduplicate (list --all-versions can repeat a version across
        # platforms/arches if filtering above didn't fully narrow it) and
        # sort newest-first.
        return sorted(set(versions), reverse=True)
    except Exception:
        return []


def _attempt_install_generation(
    uv_bin: str,
    request: str,
    *,
    project_root: Path,
    python_root: Path,
    current: SQLiteRuntimeInfo,
    allow_minor_upgrade: bool = False,
    tried_versions: set[tuple[int, int, int]] | None = None,
) -> tuple[Path, Path, SQLiteRuntimeInfo] | None:
    """One install+probe attempt for a specific version request (bare minor
    like "3.11", or an explicit patch like "3.11.15"). Each attempt gets its
    own generation directory so a rejected candidate's files are fully
    cleaned up before the next attempt, matching --reinstall semantics.
    Returns None (and cleans up) on any failure, including a vulnerable
    or off-line candidate.

    When *tried_versions* is given, the probed candidate's version is
    recorded in it so callers looping over explicit patches can skip a
    version a bare-minor request already resolved to (and rejected) --
    retrying it explicitly would spend a full download+install+probe+delete
    cycle to reach a certain rejection.
    """
    token = f"{int(time.time())}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    generation = python_root / f"generation-{token}"
    generation.mkdir(parents=True, exist_ok=False)
    _make_world_traversable(generation)

    env = managed_python_env(project_root, install_dir=generation)
    install = subprocess.run(
        [
            uv_bin,
            "python",
            "install",
            request,
            "--reinstall",
            "--no-bin",
            "--no-registry",
            "--no-config",
        ],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if install.returncode != 0:
        logger.warning(
            "private Python install failed for %s (rc=%d): %s",
            request,
            install.returncode,
            (install.stderr or install.stdout or "").strip(),
        )
        _remove_tree(generation, boundary=python_root)
        return None

    found = subprocess.run(
        [
            uv_bin,
            "python",
            "find",
            request,
            "--managed-python",
            "--no-config",
        ],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if found.returncode != 0 or not found.stdout.strip():
        logger.warning(
            "private Python lookup failed for %s (rc=%d): %s",
            request,
            found.returncode,
            (found.stderr or "").strip(),
        )
        _remove_tree(generation, boundary=python_root)
        return None

    python = Path(found.stdout.strip().splitlines()[-1])
    try:
        python.resolve().relative_to(generation.resolve())
    except (OSError, ValueError):
        logger.warning("uv resolved Python outside the Hermes generation: %s", python)
        _remove_tree(generation, boundary=python_root)
        return None

    # Do this before the candidate is probed or promoted.  On macOS, the
    # stable identifier prevents each immutable generation from looking like
    # a new TCC principal.  Failure is non-fatal: the SQLite repair must still
    # proceed when codesign is unavailable or rejects a particular artifact.
    _macos_sign_managed_python(python)

    candidate = probe_sqlite_runtime(python)
    if candidate is None:
        logger.warning("could not probe candidate Python runtime: %s", python)
        _remove_tree(generation, boundary=python_root)
        return None
    if tried_versions is not None:
        tried_versions.add(candidate.python_version[:3])
    if allow_minor_upgrade:
        # When falling forward to a higher minor line (e.g. 3.11 → 3.12),
        # only reject downgrades — allow the minor to differ.
        if candidate.python_version < current.python_version:
            logger.warning(
                "candidate Python downgraded from %s: %s",
                ".".join(str(p) for p in current.python_version),
                candidate.python_version,
            )
            _remove_tree(generation, boundary=python_root)
            return None
    elif candidate.python_version[:2] != current.python_version[:2] or (
        candidate.python_version < current.python_version
    ):
        logger.warning(
            "candidate Python drifted off the %s minor line or downgraded: %s",
            ".".join(str(p) for p in current.python_version[:2]),
            candidate.python_version,
        )
        _remove_tree(generation, boundary=python_root)
        return None
    if candidate.wal_reset_vulnerable:
        logger.warning(
            "candidate Python still links vulnerable SQLite %s (%s)",
            candidate.sqlite_version_string,
            candidate.sqlite_source_id,
        )
        _remove_tree(generation, boundary=python_root)
        return None
    return generation, python, candidate


def _install_safe_python_generation(
    uv_bin: str,
    *,
    project_root: Path,
    current: SQLiteRuntimeInfo,
) -> tuple[Path, Path, SQLiteRuntimeInfo] | None:
    runtime_root = project_root / _RUNTIME_DIR_NAME
    python_root = managed_python_install_dir(project_root)
    _make_world_traversable(runtime_root)
    _make_world_traversable(python_root)

    request = _runtime_request(current)
    print(f"  → Provisioning a private Python {request} runtime with fixed SQLite...")
    tried_versions = {current.python_version[:3]}
    result = _attempt_install_generation(
        uv_bin, request, project_root=project_root,
        python_root=python_root, current=current,
        tried_versions=tried_versions,
    )
    if result is not None:
        return result

    # The bare minor-line request resolved to a still-vulnerable (or
    # otherwise rejected) candidate. Rather than giving up immediately,
    # query which patches on this minor line uv actually knows about and
    # retry with explicit newer versions, newest-first -- this handles the
    # case where the default resolution for a bare request picks an older
    # cached/indexed patch even though a newer, non-vulnerable one is
    # available (issue #71250).
    env_for_list = managed_python_env(project_root, install_dir=python_root)
    patches = _list_available_patches(
        uv_bin, request, cwd=project_root, env=env_for_list
    )
    attempts = 0
    for version_tuple in patches:
        if attempts >= _MAX_PATCH_RETRIES:
            break
        if version_tuple in tried_versions:
            continue
        # Only NEWER patches can carry the SQLite fix. A patch at or below the
        # installed one is either the version we already know is vulnerable or
        # an older build that cannot contain a later fix, and the downgrade
        # guard in _attempt_install_generation rejects it anyway -- so trying
        # it spends a full download+install+probe+delete cycle to reach a
        # certain rejection. This matters on a uv whose download catalog is
        # stale: in #71250 the newest indexed 3.11 was 3.11.14, exactly the
        # installed version, so without this skip the loop burned all five
        # retries walking backwards (3.11.13 -> 3.11.9) before failing.
        if version_tuple <= current.python_version[:3]:
            continue
        tried_versions.add(version_tuple)
        explicit_request = ".".join(str(p) for p in version_tuple)
        print(f"  → Retrying with explicit patch {explicit_request}...")
        attempts += 1
        result = _attempt_install_generation(
            uv_bin, explicit_request, project_root=project_root,
            python_root=python_root, current=current,
        )
        if result is not None:
            return result

    # All patches on the current minor line are vulnerable or rejected.
    # Fall forward to the next supported minor (e.g. 3.11 → 3.12) so the
    # user isn't stuck on every `hermes update` with no path to a fixed
    # runtime (issue #76106).  The requires-python constraint
    # (>=3.11,<3.14) and the downstream import smoke-test gate
    # compatibility; we only need to stay inside that window.
    cur_major, cur_minor = current.python_version[:2]
    fb_tried: set[tuple[int, int, int]] = set(tried_versions)
    for next_minor in range(cur_minor + 1, 14):  # up to 3.13
        next_request = f"{cur_major}.{next_minor}"
        print(
            f"  → No fixed {cur_major}.{cur_minor} build available; "
            f"trying {next_request} as fallback..."
        )
        result = _attempt_install_generation(
            uv_bin, next_request, project_root=project_root,
            python_root=python_root, current=current,
            allow_minor_upgrade=True,
            tried_versions=fb_tried,
        )
        if result is not None:
            return result
        # Also try explicit patches on this minor line, skipping whatever
        # version the bare request above already resolved to (retrying it
        # explicitly would spend a full download+install+probe+delete cycle
        # to reach a certain rejection).
        env_for_list = managed_python_env(project_root, install_dir=python_root)
        fb_patches = _list_available_patches(
            uv_bin, next_request, cwd=project_root, env=env_for_list
        )
        fb_attempts = 0
        for version_tuple in fb_patches:
            if fb_attempts >= _MAX_PATCH_RETRIES:
                break
            if version_tuple in fb_tried:
                continue
            fb_tried.add(version_tuple)
            explicit = ".".join(str(p) for p in version_tuple)
            print(f"  → Retrying with explicit patch {explicit}...")
            fb_attempts += 1
            result = _attempt_install_generation(
                uv_bin, explicit, project_root=project_root,
                python_root=python_root, current=current,
                allow_minor_upgrade=True,
            )
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
        return (
            False,
            f"candidate still links vulnerable SQLite {info.sqlite_version_string}",
            info,
        )

    check = (
        "import dotenv, fastapi, openai, prompt_toolkit, pydantic, rich, uvicorn, yaml\n"
        "import hermes_state\n"
    )
    env = dict(os.environ)
    for key in (
        "CONDA_DEFAULT_ENV",
        "CONDA_PREFIX",
        "PYTHONHOME",
        "PYTHONPATH",
        "UV_PROJECT_ENVIRONMENT",
        "UV_PYTHON",
        "VIRTUAL_ENV",
    ):
        env.pop(key, None)
    try:
        result = subprocess.run(
            [str(python), "-I", "-c", check],
            cwd=venv_dir.parent,
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc), info
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "core import smoke failed").strip()
        last_line = detail.splitlines()[-1] if detail else "core import smoke failed"
        return False, last_line, info
    return True, "", info


def _stage_candidate_venv(
    uv_bin: str,
    *,
    project_root: Path,
    generation: Path,
    python: Path,
) -> Path | None:
    runtime_root = project_root / _RUNTIME_DIR_NAME
    token = f"{int(time.time())}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    candidate = runtime_root / f"venv-candidate-{token}"
    env = managed_python_env(
        project_root,
        install_dir=generation,
    )
    env.update({
        "UV_PROJECT_ENVIRONMENT": str(candidate),
        "UV_PYTHON": str(python),
        "UV_PYTHON_DOWNLOADS": "never",
        "VIRTUAL_ENV": str(candidate),
    })

    print("  → Building a relocatable replacement environment...")
    created = subprocess.run(
        [
            uv_bin,
            "venv",
            str(candidate),
            "--python",
            str(python),
            "--managed-python",
            "--no-python-downloads",
            "--relocatable",
            "--no-config",
        ],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if created.returncode != 0:
        logger.warning(
            "candidate venv creation failed (rc=%d): %s",
            created.returncode,
            (created.stderr or created.stdout or "").strip(),
        )
        _remove_tree(candidate, boundary=runtime_root)
        return None

    if not (project_root / "uv.lock").is_file():
        logger.warning("candidate dependency sync refused: uv.lock is missing")
        _remove_tree(candidate, boundary=runtime_root)
        return None
    # Locked sync must see project [tool.uv] exclude-newer; --no-config /
    # UV_NO_CONFIG drops it and uv 0.12+ refuses --locked.
    sync_env = dict(env)
    sync_env.pop("UV_NO_CONFIG", None)
    synced = subprocess.run(
        [
            uv_bin,
            "sync",
            "--extra",
            "all",
            "--locked",
            "--python",
            str(_venv_python(candidate)),
        ],
        cwd=project_root,
        env=sync_env,
        check=False,
    )
    if synced.returncode != 0:
        logger.warning("candidate dependency sync failed (rc=%d)", synced.returncode)
        _remove_tree(candidate, boundary=runtime_root)
        return None

    healthy, detail, _ = _smoke_candidate_venv(candidate)
    if not healthy:
        logger.warning("candidate venv smoke failed: %s", detail)
        _remove_tree(candidate, boundary=runtime_root)
        return None
    return candidate


def _rename_with_retry(source: Path, destination: Path) -> None:
    last_error: OSError | None = None
    for delay in (0.0, 0.1, 0.25, 0.5, 1.0):
        if delay:
            time.sleep(delay)
        try:
            source.rename(destination)
            return
        except OSError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error


def _cut_over_candidate(
    candidate: Path,
    *,
    project_root: Path,
    live: Path | None = None,
) -> tuple[bool, Path | None, SQLiteRuntimeInfo | None, str]:
    live = live if live is not None else project_root / _VENV_NAME
    runtime_root = project_root / _RUNTIME_DIR_NAME
    token = f"{int(time.time())}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
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
                return (
                    False,
                    backup,
                    None,
                    "could not promote the replacement venv "
                    f"({promote_error}); rollback failed ({rollback_error})",
                )
            return (
                False,
                None,
                None,
                f"could not promote the replacement venv: {promote_error}",
            )

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
            return (
                False,
                backup,
                info,
                "post-cutover smoke failed "
                f"({detail}); rollback failed ({exc}); rejected venv: {rejected}",
            )
        _remove_tree(rejected, boundary=runtime_root)
        return False, None, info, f"post-cutover smoke failed: {detail}"
    except BaseException:
        if not live.exists() and backup.exists():
            try:
                _rename_with_retry(backup, live)
            except OSError as exc:
                logger.error(
                    "interrupted runtime cutover could not restore %s from %s: %s",
                    live,
                    backup,
                    exc,
                )
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
        if os.name == "nt":
            import msvcrt

            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\0")
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (ImportError, OSError):
        os.close(fd)
        return None
    return _RepairLock(path=path, fd=fd)


def _release_repair_lock(lock: _RepairLock) -> None:
    try:
        if os.name == "nt":
            import msvcrt

            os.lseek(lock.fd, 0, os.SEEK_SET)
            msvcrt.locking(lock.fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock.fd, fcntl.LOCK_UN)
    except (ImportError, OSError):
        pass
    finally:
        try:
            os.close(lock.fd)
        except OSError:
            pass


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
    """Detect the one holder the generic scan above is blind to: THIS process.

    ``_detect_venv_python_processes`` excludes the calling process and its
    ancestors on purpose — a CLI ``hermes update`` itself runs from the
    venv python — which is correct for the dependency-sync path, where only
    a *loaded* ``.pyd`` image blocks the rewrite and a fresh child process
    dodges it.  For the whole-venv park rename that exemption is fatal:
    Windows keeps the image of any executable a running process was started
    from mapped until that process exits, so a directory containing the
    updater's own ``python.exe`` (or a waiting ``hermes.exe`` launcher
    ancestor) can never be renamed from inside the updater.  The retry loop
    in ``_cut_over_candidate`` cannot help against that — the lock is
    structural, not transient (#93032).

    No-op off Windows: POSIX renames work fine while this process maps
    files from the renamed tree (open FDs and mmaps keep inodes alive).
    """
    if platform.system() != "Windows":
        return False, ""
    try:
        live_res = str(live.resolve()).lower().rstrip(os.sep) + os.sep
    except OSError:
        live_res = str(live).lower().rstrip(os.sep) + os.sep

    def _under_live(path_value: str | None) -> bool:
        if not path_value:
            return False
        try:
            resolved = str(Path(path_value).resolve()).lower()
        except (OSError, ValueError):
            resolved = str(path_value).lower()
        return resolved.startswith(live_res)

    try:
        exe = sys.executable
    except Exception:
        exe = None
    if _under_live(exe):
        return True, (
            f"the updater itself runs from the live venv it must replace "
            f"({exe}); Windows cannot rename a directory while a process "
            "executes from inside it"
        )
    # Belt-and-braces: the venv\Scripts\hermes.exe launcher stays mapped
    # while it waits for this child, so an ancestor started from the venv
    # blocks the rename too.
    try:
        import psutil

        try:
            parents = psutil.Process().parents()
        except Exception:
            parents = []
        for anc in parents:
            try:
                anc_exe = anc.exe()
            except Exception:
                continue
            if _under_live(anc_exe):
                return True, (
                    f"ancestor process PID {anc.pid} runs from the live venv "
                    f"({anc_exe}); Windows cannot rename a directory while a "
                    "process executes from inside it"
                )
    except Exception:
        pass
    return False, ""


def _uv_version_string(uv_bin: str) -> str:
    """Return ``uv --version`` output, or ``""`` when it cannot be read."""
    try:
        result = subprocess.run(
            [uv_bin, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=15,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return (result.stdout or "").strip()


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
    """Return the venv that runtime repair should target for *root*.

    Managed installs create ``<checkout>/venv``, but uv-default and dev
    checkouts use ``<checkout>/.venv``.  Historically only ``venv`` was
    probed, so a ``.venv`` install linking a vulnerable SQLite returned
    ``not-applicable`` on every ``hermes update`` and stayed on
    journal_mode=DELETE forever — even though the WAL fallback warning
    promises that ``hermes update`` repairs the runtime (issue class:
    2,600x slower ``state.db`` appends under DELETE).

    ``venv`` wins when it holds an interpreter (managed layout takes
    precedence); otherwise fall back to ``.venv`` when that one does.
    When neither has an interpreter, return the ``venv`` path so the
    caller's existing ``not-applicable`` handling fires unchanged.
    """
    primary = root / _VENV_NAME
    if _venv_python(primary).is_file():
        return primary
    fallback = root / _ALT_VENV_NAME
    if _venv_python(fallback).is_file():
        return fallback
    return primary


def _sweep_stale_runtime_backups(
    live: Path,
    *,
    root: Path,
    keep: Path | None = None,
    min_age_seconds: float = 3600.0,
) -> None:
    """Remove leftover ``venv.stale.runtime-*`` backups next to *live*.

    A successful runtime repair parks the previous venv as
    ``<live>.stale.runtime-<token>``; historically nothing ever reclaimed
    those, so each repair leaked a full venv (~1 GB) at the project root
    forever (issue #73109).  On POSIX, deleting the tree is safe even while
    an older process still maps files from it — open FDs and mmaps keep
    their inodes alive; the directory entry is what goes away.

    ``min_age_seconds`` guards against racing a concurrent repair in
    another process: a backup parked seconds ago may still be that
    repair's rollback path, so only clearly-old markers are swept.
    ``keep`` exempts the backup the current repair just created.
    Best-effort: never raises.
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
            age = now - candidate.stat().st_mtime
        except OSError:
            continue
        if age < min_age_seconds:
            continue
        _remove_tree(candidate, boundary=root)


def repair_vulnerable_runtime(
    uv_bin: str,
    *,
    project_root: Path | None = None,
    venv_dir: Path | None = None,
) -> RuntimeRepairResult:
    """Replace a vulnerable install venv without mutating it in place.

    Every failure before cutover leaves the live venv untouched. Rename or
    post-cutover smoke failures restore the parked venv synchronously.
    """
    root = Path(project_root) if project_root is not None else _PROJECT_ROOT
    live = Path(venv_dir) if venv_dir is not None else _default_live_venv(root)
    live_python = _venv_python(live)

    if not (root / "pyproject.toml").is_file() or not live_python.is_file():
        return RuntimeRepairResult("not-applicable")

    current = probe_sqlite_runtime(live_python)
    if current is None:
        return RuntimeRepairResult(
            "skipped",
            f"could not probe live interpreter {live_python}",
        )
    if not current.wal_reset_vulnerable:
        # The runtime is already fixed — any venv.stale.runtime-* markers
        # next to the live venv are leftovers from a past repair (or from
        # a build predating the post-repair cleanup) and will never be
        # rolled back to. Sweep them so they don't leak ~1 GB each
        # forever (issue #73109). Age-gated to avoid racing an in-flight
        # repair in a sibling process.
        _sweep_stale_runtime_backups(live, root=root)
        return RuntimeRepairResult(
            "safe",
            sqlite_before=current.sqlite_version_string,
            sqlite_after=current.sqlite_version_string,
        )

    blocked, detail = _windows_runtime_holders()
    if blocked:
        print(f"  ⚠ SQLite runtime repair deferred: {detail}")
        return RuntimeRepairResult(
            "skipped",
            detail,
            sqlite_before=current.sqlite_version_string,
        )

    self_locked, self_detail = _windows_runtime_self_lock(live)
    if self_locked:
        # Structural, not transient: this process maps the live venv's own
        # executable, so the park rename fails the same way on every run and
        # no number of retries converges. Defer BEFORE provisioning — a
        # candidate staged for a cutover that can never run only leaks an
        # incomplete generation (#93032).
        print(f"  ⚠ SQLite runtime repair deferred: {self_detail}.")
        print(
            "    Retrying `hermes update` from inside this venv cannot help: "
            "the mapped executable is released only when this process exits."
        )
        print(
            "    To complete the repair, run the updater from an interpreter "
            "that lives outside this venv, e.g.:"
        )
        print(f"      cd {root}")
        print("      <system Python> -m hermes_cli.main update")
        print(
            "    Sessions stay protected meanwhile: Hermes keeps databases "
            "out of WAL mode on this SQLite build."
        )
        return RuntimeRepairResult(
            "skipped",
            self_detail,
            sqlite_before=current.sqlite_version_string,
        )

    runtime_root = root / _RUNTIME_DIR_NAME
    lock = _acquire_repair_lock(runtime_root)
    if lock is None:
        detail = "another runtime repair is already in progress"
        print(f"  ⚠ SQLite runtime repair deferred: {detail}")
        return RuntimeRepairResult(
            "skipped",
            detail,
            sqlite_before=current.sqlite_version_string,
        )

    generation: Path | None = None
    candidate: Path | None = None
    try:
        # Re-probe under the install-scoped lock: another updater may have
        # completed the repair while this process was entering the path.
        current = probe_sqlite_runtime(live_python)
        if current is None:
            return RuntimeRepairResult("skipped", "live interpreter probe failed")
        if not current.wal_reset_vulnerable:
            return RuntimeRepairResult(
                "safe",
                sqlite_before=current.sqlite_version_string,
                sqlite_after=current.sqlite_version_string,
            )

        print(
            "  ⚠ Hermes venv links SQLite "
            f"{current.sqlite_version_string}, which has the WAL-reset bug."
        )
        provisioned = _install_safe_python_generation(
            uv_bin,
            project_root=root,
            current=current,
        )
        if provisioned is None:
            # Likely a stale managed-uv catalog: python-build-standalone
            # re-releases the same patch versions with fixed SQLite, but a
            # frozen catalog keeps resolving the old vulnerable build and the
            # patch-retry loop has no newer number to try (issue #72093).
            # Refresh the managed binary and retry once.
            if _refresh_managed_binary(uv_bin):
                print("  → Managed uv refreshed; retrying provisioning...")
                provisioned = _install_safe_python_generation(
                    uv_bin,
                    project_root=root,
                    current=current,
                )
        if provisioned is None:
            return RuntimeRepairResult(
                "failed",
                "could not provision a fixed private Python runtime",
                sqlite_before=current.sqlite_version_string,
            )
        generation, python, candidate_info = provisioned

        candidate = _stage_candidate_venv(
            uv_bin,
            project_root=root,
            generation=generation,
            python=python,
        )
        if candidate is None:
            _remove_tree(generation, boundary=managed_python_install_dir(root))
            return RuntimeRepairResult(
                "failed",
                "replacement environment did not pass dependency and import smoke tests",
                sqlite_before=current.sqlite_version_string,
                sqlite_after=candidate_info.sqlite_version_string,
            )

        cut_over, backup, final_info, cutover_detail = _cut_over_candidate(
            candidate,
            project_root=root,
            live=live,
        )
        if not cut_over:
            if backup is None:
                _remove_tree(candidate, boundary=runtime_root)
                _remove_tree(generation, boundary=managed_python_install_dir(root))
            return RuntimeRepairResult(
                "failed",
                cutover_detail,
                sqlite_before=current.sqlite_version_string,
                sqlite_after=(
                    final_info.sqlite_version_string if final_info is not None else ""
                ),
                backup_venv=backup,
            )

        final_version = (
            final_info.sqlite_version_string
            if final_info is not None
            else candidate_info.sqlite_version_string
        )
        print(
            "  ✓ Managed Python runtime repaired "
            f"(SQLite {current.sqlite_version_string} → {final_version})"
        )
        if backup is not None and backup.exists():
            _remove_tree(backup, boundary=root)
        return RuntimeRepairResult(
            "repaired",
            sqlite_before=current.sqlite_version_string,
            sqlite_after=final_version,
            backup_venv=backup,
        )
    finally:
        _release_repair_lock(lock)


# ---------------------------------------------------------------------------
# Installer internals
# ---------------------------------------------------------------------------


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


def rebuild_venv(uv_bin: str, venv_dir: Path, python_version: str = "3.11") -> bool:
    True  # dont remove me. ask ethernet
