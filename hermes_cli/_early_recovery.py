"""Dependency-light venv recovery that runs BEFORE hermes_cli.main's imports.

The ``hermes`` console entry point is ``hermes_cli.main:main``.  Importing
``hermes_cli.main`` pulls in third-party packages at module level (``dotenv``
via ``hermes_cli.env_loader``, ``yaml`` via ``hermes_cli.config``, ...).  In
the exact failure state the update-recovery markers exist for — a failed lazy
backend refresh or interrupted core install that wiped a core package's
import files (#57828) — a normal launch crashes *while importing main.py*,
before ``_recover_from_interrupted_install()`` can run.  The marker system is
unreachable precisely when it is needed most.

This module is deliberately **stdlib-only** so importing it can never fail on
a corrupted venv.  ``hermes_cli.main`` imports and calls
:func:`recover_if_needed` at the very top of its module body, before any
third-party import.

Scope: this early pass only repairs enough for ``hermes_cli.main`` to become
importable again (force-reinstall of the known-fragile core packages, using
the pins from pyproject.toml).  It NEVER clears the recovery markers — the
full, confirmed marker lifecycle stays with ``_recover_from_interrupted_install()``
in main.py, which runs right after import succeeds.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Core packages a failed lazy ``uv pip install`` is known to leave with intact
# distribution metadata but wiped import files (#57828).  ``module`` is what we
# probe via a real import; ``attr`` guards against an empty/stub module.
# main.py's marker-recovery path reuses these tables — keep them here (the
# dependency-light module) so both layers probe and repair the same set.
LAZY_REFRESH_IMPORT_PROBES: tuple[tuple[str, str], ...] = (
    ("yaml", "SafeDumper"),
    ("dotenv", "load_dotenv"),
    ("click", "Command"),
    ("certifi", "contents"),
    ("rich", "print"),
    ("cryptography", "__version__"),
    ("jwt", "encode"),
)

LAZY_REFRESH_REPAIR_PACKAGES: dict[str, str] = {
    "yaml": "PyYAML",
    "dotenv": "python-dotenv",
    "click": "click",
    "certifi": "certifi",
    "rich": "rich",
    "cryptography": "cryptography",
    "jwt": "PyJWT",
}


# --- Windows entry-point shim quarantine -----------------------------------
#
# ``hermes update`` renames the live ``hermes*.exe`` shims aside
# (``hermes.exe.old.<unix-ms>``) so uv can write replacements. Putting them BACK
# is the safety-critical direction: losing that rename leaves the install with
# no ``hermes`` on PATH, and the command that would repair it IS ``hermes
# update`` (#75584).
#
# Three call sites restore a quarantined shim -- the updater, the
# early-recovery installer, and the startup sweep's orphan rescue. They used to
# be separate one-shot renames with swallowed errors; the two that had messages
# had already drifted apart. The logic lives here, in the one stdlib-only module
# all of them can import, so the ladder and the recovery wording stay in
# lockstep.

QUARANTINE_RESTORE_BACKOFF_MS: tuple[int, ...] = (0, 100, 250, 500, 1000)


def restore_quarantined_shims(
    moved: list[tuple[Path, Path]],
    *,
    stream=None,
    backoff_ms: tuple[int, ...] = QUARANTINE_RESTORE_BACKOFF_MS,
) -> list[tuple[Path, Path]]:
    """Rename quarantined shims back, retrying a lock instead of giving up.

    ``moved`` holds ``(original, quarantined)`` pairs. Returns the pairs that
    could NOT be restored, and prints an actionable recovery command for each.

    A pair is not a failure when ``original`` already exists or ``quarantined``
    has gone: the installer wrote a fresh shim, or a concurrent sweep won the
    race. Both are silent, so two processes sweeping the same orphan cannot
    produce a spurious error.

    Messages go to stderr by default -- the startup sweep runs on EVERY hermes
    invocation, and ``hermes acp`` speaks JSON-RPC on stdout.
    """
    if stream is None:
        stream = sys.stderr

    failed: list[tuple[Path, Path]] = []

    for original, quarantined in moved:
        last_exc: OSError | None = None

        for delay_ms in backoff_ms:
            try:
                if os.path.exists(original) or not os.path.exists(quarantined):
                    last_exc = None
                    break
                if delay_ms:
                    time.sleep(delay_ms / 1000.0)
                os.rename(quarantined, original)
                last_exc = None
                break
            except OSError as exc:
                last_exc = exc
                continue

        if last_exc is None:
            continue

        failed.append((original, quarantined))
        name = os.path.basename(str(original))
        stem = name[:-4] if name.lower().endswith(".exe") else name
        print(
            f"  ✖ FAILED to restore {name} "
            f"({last_exc.__class__.__name__}) — it is still quarantined "
            f"as {os.path.basename(str(quarantined))}.\n"
            f"    `{stem}` will NOT be on PATH until it is put back. Run this, "
            f"then re-run the update:\n"
            f'      move "{quarantined}" "{original}"',
            file=stream,
        )

    return failed


# Set only when this process successfully finishes a deferred core install for
# an ``update`` invocation.  The normal CLI import that follows must not resolve
# external secret sources: a configured source can map cryptography._rust and
# immediately recreate the self-lock marker this fresh process just consumed.
# Process-local state is intentional so child processes do not inherit the
# bootstrap exception.
_UPDATE_RETRY_RECOVERED = False


def _should_skip_external_secret_sources() -> bool:
    """Whether this updater already completed its deferred native install."""
    return _UPDATE_RETRY_RECOVERED


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _pid_is_running(pid: int) -> bool:
    """Best-effort stdlib-only process liveness probe.

    ``os.kill(pid, 0)`` is not a no-op on Windows, so use the Win32 process
    handle API there.  An access-denied result is conservatively live: racing
    an elevated updater is worse than postponing recovery for one launch.
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            import ctypes

            synchronize = 0x00100000
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = [
                ctypes.c_ulong,
                ctypes.c_int,
                ctypes.c_ulong,
            ]
            kernel32.OpenProcess.restype = ctypes.c_void_p
            kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
            kernel32.WaitForSingleObject.restype = ctypes.c_ulong
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel32.CloseHandle.restype = ctypes.c_int
            handle = kernel32.OpenProcess(synchronize, False, pid)
            if not handle:
                return ctypes.get_last_error() == 5  # ERROR_ACCESS_DENIED
            try:
                return kernel32.WaitForSingleObject(handle, 0) == 258
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return True
    try:
        os.kill(pid, 0)  # windows-footgun: ok — Windows returns above
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _marker_owner_is_live(marker: Path) -> bool:
    """True when a legacy update marker names a process still running."""
    try:
        body = marker.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    for line in body.splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "pid":
            try:
                return _pid_is_running(int(value.strip()))
            except ValueError:
                return False
    return False


def _pinned_specs(packages: list[str], project_root: Path) -> list[str]:
    """Map bare package names to their pinned specs from pyproject.toml.

    Stdlib-only (tomllib + naive requirement-head parsing — ``packaging`` may
    itself be broken in the failure state this module exists for).  Unknown
    packages fall back to their bare name.
    """
    pyproject = project_root / "pyproject.toml"
    if not pyproject.is_file():
        return packages
    try:
        import tomllib

        with open(pyproject, "rb") as f:
            raw_deps = tomllib.load(f).get("project", {}).get("dependencies", []) or []
    except Exception:
        return packages

    name_to_spec: dict[str, str] = {}
    for spec in raw_deps:
        head = spec.split(";", 1)[0].strip()
        bare = head
        for op in ("==", ">=", "<=", "~=", ">", "<", "!="):
            if op in bare:
                bare = bare.split(op, 1)[0]
                break
        key = bare.strip().split("[", 1)[0].strip().lower()
        if key:
            name_to_spec[key] = head
    return [name_to_spec.get(pkg.lower(), pkg) for pkg in packages]


def _certifi_bundle_broken() -> bool:
    """True when certifi imports but its ``cacert.pem`` is missing/corrupt.

    A brew Python upgrade or an interrupted venv rebuild can leave certifi's
    distribution metadata (and even the module) intact while the bundled
    ``cacert.pem`` is gone or a dangling symlink — every TLS connection then
    fails with an opaque ``Could not find a suitable TLS CA certificate
    bundle`` from deep inside httpx/requests (#29866). An attribute probe
    alone passes in that state, so validate the bundle path itself.
    """
    try:
        import certifi

        bundle = Path(certifi.where())
        # <1 KiB cannot hold a single PEM certificate — treat as corrupt.
        return not bundle.is_file() or bundle.stat().st_size < 1024
    except Exception:
        # Import failure is caught by the regular probe table; a failure to
        # even stat is treated as broken.
        return True


def _probe_broken_packages() -> list[str]:
    """Import-probe the fragile core packages in THIS process.

    Returns repair package names (deduped, probe order) for modules that fail
    to import or lack their sentinel attribute.  Failed imports leave nothing
    in ``sys.modules``, so a post-repair retry in the same process works.

    certifi additionally gets a bundle-file check: the module can import
    cleanly while ``cacert.pem`` is missing (#29866).
    """
    broken: list[str] = []
    for mod_name, attr in LAZY_REFRESH_IMPORT_PROBES:
        try:
            mod = importlib.import_module(mod_name)
            if not hasattr(mod, attr):
                raise ImportError(f"{mod_name} missing {attr}")
            if mod_name == "certifi" and _certifi_bundle_broken():
                raise ImportError("certifi cacert.pem missing or corrupt")
        except Exception:
            pkg = LAZY_REFRESH_REPAIR_PACKAGES.get(mod_name)
            if pkg and pkg not in broken:
                broken.append(pkg)
    return broken


def _find_uv_binary() -> str | None:
    """Locate a ``uv`` binary without importing third-party modules.

    uv-managed base interpreters carry an ``EXTERNALLY-MANAGED`` marker, so
    the stdlib ``pip`` fallback below refuses to touch them.  In that state
    the only sanctioned installer is uv itself, which Hermes already vendors
    (``~/.hermes/uv/uv.exe``) or the user has on PATH.  Stdlib-only.
    """
    exe = "uv.exe" if sys.platform == "win32" else "uv"

    # Resolve the Hermes home without importing hermes_constants: this module
    # is stdlib-only on purpose — it runs when the checkout is already broken
    # enough that importing a Hermes module may fail. Mirrors
    # hermes_constants._get_platform_default_hermes_home (LOCALAPPDATA on
    # Windows, ~/.hermes elsewhere); if that logic ever changes, keep the two
    # in sync.
    hermes_env = os.environ.get("HERMES_HOME", "").strip()
    if hermes_env:
        hermes_home = Path(hermes_env)
    elif sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
        base = Path(local_appdata) if local_appdata else Path.home() / "AppData" / "Local"
        hermes_home = base / "hermes"
    else:
        hermes_home = Path.home() / ".hermes"

    candidates = [
        # Private uv in Hermes home (custom HERMES_HOME, %LOCALAPPDATA%/hermes on Windows, ~/.hermes on POSIX)
        hermes_home / "uv" / exe,
        # Legacy pre-isolation layout; migrated to the private dir on use.
        hermes_home / "bin" / exe,
        Path.home() / ".local" / "bin" / exe,
        Path.home() / ".cargo" / "bin" / exe,
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return shutil.which(exe)


def _base_interpreter_is_externally_managed() -> bool:
    """True when ``sys.executable`` is a uv/standalone-builds managed install.

    Those interpreters ship an ``EXTERNALLY-MANAGED`` marker next to their
    stdlib (PEP 668), so ``python -m pip install`` aborts with
    ``externally-managed-environment``.  The early repair must then go
    through uv (or explicitly override pip) or the reinstall no-ops and the
    venv stays broken (#83569).
    """
    try:
        import sysconfig

        stdlib = Path(sysconfig.get_path("stdlib"))
        if (stdlib / "EXTERNALLY-MANAGED").exists():
            return True
        # uv 0.5+ moved the marker into a ``_uv_managed`` sentinel dir…
        if (stdlib.parent / "EXTERNALLY-MANAGED").exists():
            return True
    except Exception:
        pass
    return False


def _run_repair_install(specs: list[str], project_root: Path) -> bool:
    """``uv pip`` (or stdlib ``pip``) force-reinstall of the given specs.

    Streams nothing to stdout (``hermes acp`` speaks JSON-RPC on stdout);
    output is captured and replayed to stderr only on failure.  Never raises.

    Two installer paths, in priority order:

    1. ``uv pip install`` with ``VIRTUAL_ENV`` pointed at the project venv —
       required when the base interpreter is uv-managed (Windows git checkouts
       install exactly this way: uv's Python declares PEP 668
       ``EXTERNALLY-MANAGED`` and plain ``python -m pip`` refuses to run).
    2. ``sys.executable -m pip`` as before, for self-contained venvs whose
       interpreter carries no PEP 668 marker.
    """
    externally_managed = _base_interpreter_is_externally_managed()
    if externally_managed:
        uv = _find_uv_binary()
        if uv:
            env = {**os.environ, "VIRTUAL_ENV": str(project_root / "venv")}
            env.pop("PYTHONHOME", None)
            env.pop("PYTHONPATH", None)
            try:
                result = subprocess.run(
                    [uv, "pip", "install", "--force-reinstall", *specs],
                    cwd=project_root,
                    capture_output=True,
                    text=True, encoding="utf-8", errors="replace",
                    env=env,
                )
                if result.returncode == 0:
                    return True
                tail = (result.stderr or result.stdout or "")[-2000:]
                if tail:
                    print(tail, file=sys.stderr)
                return False
            except Exception as exc:
                print(f"  ✗ Early venv repair could not run uv: {exc}", file=sys.stderr)
                return False
        # No uv available: fall through to pip with the PEP 668 override so
        # the repair at least attempts to fix the venv instead of no-oping.
        print(
            "  ⚠ Base interpreter is externally managed and no uv binary was "
            "found; retrying repair via pip with PEP 668 override.",
            file=sys.stderr,
        )

    try:
        subprocess.run(
            [sys.executable, "-m", "ensurepip", "--upgrade", "--default-pip"],
            cwd=project_root,
            capture_output=True,
        )
    except Exception:
        pass
    pip_cmd = [sys.executable, "-m", "pip", "install", "--force-reinstall"]
    if externally_managed:
        pip_cmd.append("--break-system-packages")
    pip_cmd.extend(specs)
    try:
        result = subprocess.run(
            pip_cmd,
            cwd=project_root,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
    except Exception as exc:
        print(f"  ✗ Early venv repair could not run pip: {exc}", file=sys.stderr)
        return False
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "")[-2000:]
        if tail:
            print(tail, file=sys.stderr)
        return False
    return True


def _pytest_owns_live_checkout(root: Path) -> bool:
    """True when running under pytest AND ``root`` is this module's own
    checkout — the one whose venv is executing the suite right now.

    Lifecycle tests spawn real subprocesses that import ``hermes_cli.main``
    with recovery armed; ``PYTEST_CURRENT_TEST`` rides the inherited env into
    those children. Without this guard, a genuinely-broken dev venv gets a
    REAL ``ensurepip`` + ``pip install --force-reinstall`` from inside a
    running test suite. Tests that sandbox ``project_root`` to a tmp_path are
    unaffected (same posture as ``managed_scope._under_pytest``)."""
    return (
        "PYTEST_CURRENT_TEST" in os.environ
        and root == Path(__file__).resolve().parent.parent
    )


def recover_if_needed(
    project_root: Path | None = None,
    argv: list[str] | None = None,
) -> None:
    """Repair wiped core packages so ``hermes_cli.main`` can import at all.

    Fast path (no marker present) is two ``lstat`` calls.  Only acts when a
    recovery marker from a prior ``hermes update`` exists AND an import probe
    confirms a core package is actually broken.  Markers are intentionally
    NOT cleared here — ``_recover_from_interrupted_install()`` in main.py owns
    the confirmed marker lifecycle and runs immediately after import succeeds.

    Never raises: on any failure the import of main.py proceeds and surfaces
    the real error.
    """
    global _UPDATE_RETRY_RECOVERED

    try:
        args = sys.argv[1:] if argv is None else argv
        root = _project_root() if project_root is None else project_root
        if _pytest_owns_live_checkout(root):
            return
        core_marker = root / ".update-incomplete"
        lazy_marker = root / ".lazy-refresh-incomplete"
        if not core_marker.exists() and not lazy_marker.exists():
            return
        # Managed/Docker/PyPI installs have no source tree here — the marker
        # is not ours to act on; main.py's recovery clears it.
        if not (root / "pyproject.toml").is_file():
            return

        # Pending core install (.update-incomplete) — complete it NOW, before
        # any native extension can be imported by this process.  The lazy
        # import-probe path below only proves main.py is importable; the core
        # install here guarantees the WHOLE dependency set is replaced while
        # nothing pins venv .pyd files yet (#83569 self-lock: deferring the
        # install to main()'s post-import recovery re-locks it on Windows).
        # Bounded retries: a persistently failing install must not hammer
        # every launch, so attempts past the ceiling are left for main.py's
        # post-import recovery path (which can safely probe-import after this
        # process already holds whatever extensions it needs).
        # A live marker owner means another updater is currently inside the
        # marker-to-install window.  Never race it.  A dead owner means this is
        # a prior deferral/interruption and MUST be recovered even when this
        # launch is itself `hermes update`: CLI and Desktop retries preserve
        # that argv, and skipping solely on argv recreates the self-lock loop.
        if core_marker.exists():
            if _marker_owner_is_live(core_marker):
                return
            completed = _complete_pending_core_install(root, core_marker)
            if completed and "update" in args:
                _UPDATE_RETRY_RECOVERED = True
            return

        # Keep the historical update-argv exclusion for the lazy-refresh
        # marker.  Unlike the core marker it is not a deferred native install,
        # and the active update flow owns its probe/repair lifecycle.
        if "update" in args:
            return

        broken = _probe_broken_packages()
        if not broken:
            # Imports are fine — main.py will load and run full recovery.
            return

        # Single-flight: share main.py's recovery lock so an early repair
        # never races a concurrent full recovery into the same shared venv.
        lock_path = root / ".update-incomplete.lock"
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()}\n".encode())
            os.close(fd)
        except FileExistsError:
            try:
                if time.time() - lock_path.stat().st_mtime > 3600:
                    lock_path.unlink()
            except OSError:
                pass
            return
        except OSError:
            pass  # read-only fs / perms — proceed unlocked, install surfaces it

        try:
            specs = _pinned_specs(broken, root)
            print(
                "⚠ Core package(s) broken by an interrupted update — "
                f"repairing before launch: {', '.join(broken)}",
                file=sys.stderr,
            )
            if _run_repair_install(specs, root) and not _probe_broken_packages():
                print("  ✓ Core packages repaired.", file=sys.stderr)
            else:
                print(
                    "  ✗ Automatic repair incomplete. Recover manually with:",
                    file=sys.stderr,
                )
                print(
                    f"    {sys.executable} -m pip install --force-reinstall "
                    + " ".join(specs),
                    file=sys.stderr,
                )
        finally:
            try:
                lock_path.unlink()
            except OSError:
                pass
    except Exception:
        # Never block launch — the import of main.py will surface the truth.
        pass


# Cap on automatic early-pass install retries.  A persistently failing
# install (e.g. network down, index unreachable) must not reinstall-hammer
# every `hermes` launch: past this many attempts the early pass hands the
# marker to main.py's post-import recovery, which presents the manual
# recovery command.  The counter lives inside the marker file itself (JSON
# body) and is bumped on each failed attempt.
_EARLY_CORE_INSTALL_MAX_ATTEMPTS = 3


def _claim_recovery_lock(root: Path) -> bool:
    """Single-flight claim on the shared recovery lock.  Never raises."""
    lock_path = root / ".update-incomplete.lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()}\n".encode())
        os.close(fd)
        return True
    except FileExistsError:
        try:
            if time.time() - lock_path.stat().st_mtime > 3600:
                lock_path.unlink()
        except OSError:
            pass
        return False
    except OSError:
        # Read-only fs / perms — proceed unlocked; the install itself
        # surfaces the real problem.  Recoverable.
        return True


def _release_recovery_lock(root: Path) -> None:
    """Best-effort release of the shared recovery lock."""
    try:
        (root / ".update-incomplete.lock").unlink()
    except OSError:
        pass


def _complete_pending_core_install(root: Path, core_marker: Path) -> bool:
    """Run the pending core install BEFORE main.py can import native modules.

    ``recover_if_needed`` invokes this when ``.update-incomplete`` exists —
    a prior ``hermes update`` (or the self-lock preflight, #83569) left the
    dependency sync deliberately unfinished.  Completing it here matters on
    Windows: the deferral exists precisely because the process that wrote the
    marker had a native venv extension mapped; this process, running before
    ``hermes_cli.main``'s third-party imports, maps nothing yet, so the
    installer can replace ``.pyd`` files without hitting the lock.

    Marker lifecycle: cleared on success; kept (attempts counter bumped) on
    failure for the next launch or main.py's post-import recovery.  An
    attempts ceiling caps automatic retries so a persistent installer
    failure does not block every launch (``hermes acp`` included).

    Never raises: any failure leaves the marker for the post-import path and
    returns ``False``.  Returns ``True`` only after the install succeeds.
    """
    try:
        from hermes_cli import _install_repair as ir

        # Retry backoff: read current attempts before claiming the lock so a
        # persistently-failing install stops hammering early.  After the
        # increment the counter reflects THIS attempt.
        attempts = 0
        try:
            raw = core_marker.read_text(encoding="utf-8", errors="replace").strip()
            if raw:
                import json as _json

                try:
                    attempts = int(_json.loads(raw).get("attempts", 0))
                except (ValueError, AttributeError):
                    attempts = 0
        except OSError:
            attempts = 0

        if attempts >= _EARLY_CORE_INSTALL_MAX_ATTEMPTS:
            print(
                "⚠ Pending interrupted-update install has already failed "
                f"{attempts} times in the early pass — leaving it for the "
                "post-import recovery path.",
                file=sys.stderr,
            )
            return False

        if not _claim_recovery_lock(root):
            return False

        try:
            print(
                "⚠ A previous `hermes update` was interrupted mid-install — "
                "finishing dependency installation now (before any native "
                "extensions load)...",
                file=sys.stderr,
            )
            ir.run_core_install(root)
        except Exception as exc:
            new_attempts = ir.bump_marker_attempts(core_marker)
            print(
                f"  ✗ Early interrupted-install completion failed (attempt "
                f"{new_attempts}/{_EARLY_CORE_INSTALL_MAX_ATTEMPTS}): {exc}",
                file=sys.stderr,
            )
            print(
                "  The next launch will retry; hermes will keep working from "
                "the current venv in the meantime.",
                file=sys.stderr,
            )
            return False
        finally:
            _release_recovery_lock(root)

        try:
            core_marker.unlink()
        except OSError:
            pass
        print(
            "  ✓ Dependency installation completed in the early pass.",
            file=sys.stderr,
        )
        return True
    except Exception:
        # Never block launch — the marker stays for the post-import path.
        return False
