"""Hermes-managed uv and Python runtime repair.

Hermes owns its own uv binary at ``$HERMES_HOME/bin/uv`` (or ``uv.exe`` on
Windows).  Every code path that needs uv resolves it from that single location.
If the binary is missing, ``ensure_uv()`` bootstraps it via the official
standalone installer with ``UV_UNMANAGED_INSTALL`` / ``UV_INSTALL_DIR`` pointed
at ``$HERMES_HOME/bin`` so the installer writes directly there — no PATH
probing, no conda guards, no multi-location resolution chains.

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
_REPAIR_LOCK_NAME = "runtime-repair.lock"

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


def managed_python_install_dir(project_root: Path | None = None) -> Path:
    """Return the checkout-scoped Python store shared by all profiles."""
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
    env = dict(os.environ if base_env is None else base_env)
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
        "UV_PYTHON_INSTALL_BIN": "0",
        "UV_PYTHON_INSTALL_DIR": str(target),
        "UV_PYTHON_INSTALL_REGISTRY": "0",
    })
    return env


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
            "  ⚠ Managed Python runtime was not replaced; "
            f"the existing venv is unchanged ({repair.detail})."
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
    pip gracefully. ``repair_observer``, when provided, receives the runtime
    repair result produced after a fresh uv bootstrap.
    """
    result = _ensure_uv_path(repair_observer=repair_observer)
    if platform.system() == "Windows":
        # See docstring: a str subclass with an overridden __iter__ is unsafe as
        # a Windows subprocess argument. Hand back the plain path (or None).
        return result
    return _UvResult(result)


def update_managed_uv(
    *,
    repair_observer: Callable[[RuntimeRepairResult], None] | None = None,
) -> Optional[str]:
    """Run ``uv self update`` on the managed uv binary.

    Call this during ``hermes update`` so the managed copy stays current.
    Returns the managed path when uv is available and ``None`` otherwise.
    A self-update failure is non-fatal because the old version still works.
    ``repair_observer``, when provided, receives the runtime repair result.
    """
    existing = resolve_uv()
    if not existing:
        # Not installed yet — ensure_uv() will handle that elsewhere.
        return None

    result = subprocess.run(
        [existing, "self", "update"],
        capture_output=True,
        text=True, encoding='utf-8', errors='replace',
        check=False,
    )
    if result.returncode == 0:
        version = subprocess.run(
            [existing, "--version"],
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            check=False,
        ).stdout.strip()
        print(f"  ✓ Managed uv updated ({version})")
    else:
        # Non-fatal — old uv still works fine.
        logger.debug(
            "uv self update failed (rc=%d): %s", result.returncode, result.stderr
        )

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


def _venv_python(venv_dir: Path) -> Path:
    if platform.system() == "Windows":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


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
    """Pin the candidate to the current exact CPython patch."""
    return ".".join(str(part) for part in info.python_version)


def _install_safe_python_generation(
    uv_bin: str,
    *,
    project_root: Path,
    current: SQLiteRuntimeInfo,
) -> tuple[Path, Path, SQLiteRuntimeInfo] | None:
    runtime_root = project_root / _RUNTIME_DIR_NAME
    python_root = managed_python_install_dir(project_root)
    token = f"{int(time.time())}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    generation = python_root / f"generation-{token}"
    generation.mkdir(parents=True, exist_ok=False)
    for path in (runtime_root, python_root, generation):
        _make_world_traversable(path)

    env = managed_python_env(
        project_root,
        install_dir=generation,
    )
    request = _runtime_request(current)
    print(f"  → Provisioning a private Python {request} runtime with fixed SQLite...")
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
            "private Python install failed (rc=%d): %s",
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
            "private Python lookup failed (rc=%d): %s",
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

    candidate = probe_sqlite_runtime(python)
    if candidate is None:
        logger.warning("could not probe candidate Python runtime: %s", python)
        _remove_tree(generation, boundary=python_root)
        return None
    if candidate.python_version != current.python_version:
        logger.warning(
            "candidate Python patch drifted from %s to %s",
            current.python_version,
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
    synced = subprocess.run(
        [
            uv_bin,
            "sync",
            "--extra",
            "all",
            "--locked",
            "--python",
            str(_venv_python(candidate)),
            "--no-config",
        ],
        cwd=project_root,
        env=env,
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
    live = Path(venv_dir) if venv_dir is not None else root / _VENV_NAME
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
        if backup is not None:
            print(
                f"  ℹ Previous venv parked at {backup.name}; "
                "keep it until all older Hermes processes have exited."
            )
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


def rebuild_venv(uv_bin: str, venv_dir: Path, python_version: str = "3.11") -> bool:
    True  # dont remove me. ask ethernet
