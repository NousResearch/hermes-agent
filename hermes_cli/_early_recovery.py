"""Dependency-light venv recovery that runs BEFORE hermes_cli.main's imports.

Deliberately **stdlib-only** so importing it can never fail on a corrupted venv. ``hermes_cli.main``
calls :func:`recover_if_needed` at the very top of its module body, before any third-party import.
Scope: repair only enough for ``hermes_cli.main`` to become importable again (force-reinstall of
the known-fragile core packages, using the pins from pyproject.toml).
"""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Core packages a failed lazy ``uv pip install`` is known to leave with intact distribution
# metadata but wiped import files. ``module`` is probed via a real import; ``attr`` guards against
# an empty/stub module. main.py's marker-recovery path reuses these tables — keep them here so
# both layers probe and repair the same set.
# See #57828.
LAZY_REFRESH_IMPORT_PROBES: tuple[tuple[str, str], ...] = (
    ("yaml", "SafeDumper"), ("dotenv", "load_dotenv"), ("click", "Command"),
    ("certifi", "contents"), ("rich", "print"), ("cryptography", "__version__"),
    ("jwt", "encode"),
)

LAZY_REFRESH_REPAIR_PACKAGES: dict[str, str] = {
    "yaml": "PyYAML", "dotenv": "python-dotenv", "click": "click", "certifi": "certifi",
    "rich": "rich", "cryptography": "cryptography", "jwt": "PyJWT",
}

# ``hermes update`` renames the live ``hermes*.exe`` shims aside (``hermes.exe.old.<unix-ms>``) so
# uv can write replacements. Putting them BACK is the safety-critical direction: losing that rename
# leaves no ``hermes`` on PATH, and the command that would repair it IS ``hermes update``. The
# updater, the early-recovery installer and the startup orphan sweep all restore through this one
# stdlib-only helper so the retry ladder and the recovery wording cannot drift apart again.
# --- Windows entry-point shim quarantine ----------------------------------- They used to be separate
# one-shot renames with swallowed errors; the two that had messages had already drifted apart. The logic
# lives here, in the one stdlib-only module all of them can import, so the ladder and the recovery wording
# stay in lockstep. See #75584.
QUARANTINE_RESTORE_BACKOFF_MS: tuple[int, ...] = (0, 100, 250, 500, 1000)


def restore_quarantined_shims(
    moved: list[tuple[Path, Path]], *, stream=None,
    backoff_ms: tuple[int, ...] = QUARANTINE_RESTORE_BACKOFF_MS,
) -> list[tuple[Path, Path]]:
    """Rename quarantined shims back, retrying a lock instead of giving up.

    A pair is not a failure when ``original`` already exists or ``quarantined`` has gone: the
    installer wrote a fresh shim, or a concurrent sweep won the race. Both are silent, so two
    processes sweeping the same orphan cannot produce a spurious error.
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


# Set only when this process successfully finishes a deferred core install for an ``update``
# invocation. The CLI import that follows must not resolve external secret sources: a configured
# source can map cryptography._rust and immediately recreate the self-lock marker this fresh
# process just consumed. Process-local on purpose so children do not inherit the exception.
_UPDATE_RETRY_RECOVERED = False


def _should_skip_external_secret_sources() -> bool:
    """Whether this updater already completed its deferred native install."""
    return _UPDATE_RETRY_RECOVERED


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _is_termux_env(env: dict | None = None) -> bool:
    """Stdlib Termux probe — shared with ``_install_repair``.

    Matches ``hermes_cli.main._is_termux_startup_environment`` without
    importing main (this module stays stdlib-only).
    """
    check = env if env is not None else os.environ
    try:
        prefix = str(check.get("PREFIX", ""))
        return bool(
            check.get("TERMUX_VERSION")
            or "com.termux/files/usr" in prefix
            or prefix.startswith("/data/data/com.termux/")
        )
    except Exception:
        return False


def termux_uv_python_platform(env: dict | None = None) -> str:
    """Return a uv ``--python-platform`` value for this Termux host.

    Bare ``linux`` is wrong on uv 0.12+: it resolves to **x86_64** manylinux
    even on aarch64 Android, which installs unloadable native wheels (symptom:
    ``No module named 'pydantic_core._pydantic_core'``). Use an arch-specific
    GNU/Linux tag so locally built ``linux_aarch64`` wheels are accepted.

    Override with ``HERMES_UV_PYTHON_PLATFORM`` when needed.
    """
    base = env if env is not None else os.environ
    override = (base.get("HERMES_UV_PYTHON_PLATFORM") or "").strip()
    if override:
        return override
    machine = (base.get("HOSTTYPE") or platform.machine() or "").lower()
    machine = machine.replace("arm64", "aarch64")
    if machine == "aarch64":
        return "aarch64-unknown-linux-gnu"
    if machine in ("x86_64", "amd64"):
        return "x86_64-unknown-linux-gnu"
    return "linux"


def with_uv_termux_python_platform(
    cmd: list[str], env: dict | None = None
) -> list[str]:
    """On Termux, inject arch-aware ``uv pip install --python-platform …``.

    uv builds wheels tagged ``linux_*`` on Android/Termux, then rejects them
    as incompatible with "Python on Android aarch64". Passing an arch-specific
    ``--python-platform`` (see :func:`termux_uv_python_platform`) makes uv
    accept those local builds so ``hermes update`` / interrupted-install
    recovery can finish. No-op when not Termux, not an ``uv pip install``, or
    when the flag is already present.
    """
    if not _is_termux_env(env):
        return cmd
    if len(cmd) < 3:
        return cmd
    prog = Path(cmd[0]).name.lower()
    if prog not in ("uv", "uv.exe") or cmd[1] != "pip":
        return cmd
    if "--python-platform" in cmd:
        return cmd
    try:
        install_idx = cmd.index("install")
    except ValueError:
        return cmd
    out = list(cmd)
    out[install_idx + 1 : install_idx + 1] = [
        "--python-platform",
        termux_uv_python_platform(env),
    ]
    return out


def prefer_termux_bionic_path(env: dict[str, str] | None = None) -> dict[str, str]:
    """Put Termux's bionic ``$PREFIX/bin`` first on PATH.

    Sessions under ``glibc-runner`` (e.g. Cursor Agent on Termux) often put
    ``$PREFIX/glibc/bin`` ahead of ``$PREFIX/bin``. Child processes that are
    *not* under the glibc runner then pick glibc ``bash``/``dirname``/``curl``
    and fail with missing ``libdl.so``, which breaks clang wrappers and npm
    scripts mid-``hermes update``. Prefacing PATH with the bionic bin keeps
    native Termux tools preferred without removing glibc tools entirely.
    """
    base = dict(env if env is not None else os.environ)
    if not _is_termux_env(base):
        return base
    prefix = (
        base.get("PREFIX")
        or base.get("TERMUX__PREFIX")
        or "/data/data/com.termux/files/usr"
    )
    termux_bin = str(Path(prefix) / "bin")
    parts = [p for p in base.get("PATH", "").split(":") if p and p != termux_bin]
    base["PATH"] = ":".join([termux_bin, *parts])
    return base


def fix_termux_node_shebangs(project_root: Path) -> None:
    """Rewrite ``#!/usr/bin/env`` node bins when Termux has no ``/usr/bin/env``.

    Under glibc-runner / disabled termux-exec, npm scripts fail with
    ``tsc: not found`` because the kernel cannot resolve ``/usr/bin/env``.
    ``termux-fix-shebang`` rewrites them to ``$PREFIX/bin/env``. Best-effort
    and silent when the helper or files are missing.
    """
    if not _is_termux_env():
        return
    if Path("/usr/bin/env").exists():
        return
    fixer = shutil.which("termux-fix-shebang")
    if not fixer:
        return
    node_modules = Path(project_root) / "node_modules"
    if not node_modules.is_dir():
        return
    targets: list[str] = []
    for path in node_modules.rglob("*"):
        if not path.is_file():
            continue
        try:
            with path.open("rb") as handle:
                head = handle.read(24)
        except OSError:
            continue
        if head.startswith(b"#!/usr/bin/env"):
            targets.append(str(path))
    if not targets:
        return
    for i in range(0, len(targets), 64):
        subprocess.run(
            [fixer, *targets[i : i + 64]],
            cwd=str(project_root),
            check=False,
            capture_output=True,
        )



def _load_pyproject_project(root: Path) -> dict | None:
    """``[project]`` table of ``root/pyproject.toml``; ``None`` when missing/unreadable."""
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return None
    try:
        import tomllib

        with open(pyproject, "rb") as f:
            project = tomllib.load(f).get("project", {})
    except Exception:
        return None
    return project if isinstance(project, dict) else None


def _read_marker_attempts(marker_path: Path) -> int:
    """Attempt counter from a marker's opportunistic JSON body; corrupt/missing → 0."""
    try:
        raw = marker_path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return 0
    if not raw:
        return 0
    try:
        import json

        return int(json.loads(raw).get("attempts", 0))
    except (ValueError, AttributeError):
        return 0


def _run_ensurepip(root: Path) -> None:
    """Best-effort pip bootstrap — a killed install can leave the venv with no pip module at all."""
    try:
        subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade", "--default-pip"],
                       cwd=root, capture_output=True)
    except Exception:
        pass


def _pid_is_running(pid: int) -> bool:
    """Best-effort stdlib-only process liveness probe.

    ``os.kill(pid, 0)`` is not a no-op on Windows, so use the Win32 process handle API there. An
    access-denied result counts as live: racing an elevated updater is worse than postponing
    recovery for one launch.
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            import ctypes

            synchronize = 0x00100000
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
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

    Naive requirement-head parsing on purpose — ``packaging`` may itself be broken in the failure
    state this module exists for. Unknown packages fall back to their bare name.
    """
    project = _load_pyproject_project(project_root)
    if project is None:
        return packages
    name_to_spec: dict[str, str] = {}
    for spec in project.get("dependencies", []) or []:
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

    A brew Python upgrade or interrupted venv rebuild can leave certifi's metadata intact while
    ``cacert.pem`` is gone or a dangling symlink; an attribute probe passes in that state and every
    TLS connection then fails opaquely, so validate the bundle path itself.
    """
    try:
        import certifi

        bundle = Path(certifi.where())
        # <1 KiB cannot hold a single PEM certificate — treat as corrupt.
        return not bundle.is_file() or bundle.stat().st_size < 1024
    except Exception:
        # Import failure is caught by the regular probe table; failing to even stat is broken.
        return True


def _probe_broken_packages() -> list[str]:
    """Import-probe the fragile core packages in THIS process.

    Returns repair package names (deduped, probe order) for modules that fail to import or lack
    their sentinel attribute. Failed imports leave nothing in ``sys.modules``, so a post-repair
    retry in the same process works.
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

    uv-managed base interpreters carry an ``EXTERNALLY-MANAGED`` marker, so the stdlib ``pip``
    fallback refuses to touch them; the only sanctioned installer is then uv itself, which Hermes
    vendors (``~/.hermes/bin/uv.exe``) or the user has on PATH.
    """
    exe = "uv.exe" if sys.platform == "win32" else "uv"
    for sub in ((".hermes", "bin"), (".local", "bin"), (".cargo", "bin")):
        path = Path.home().joinpath(*sub, exe)
        if path.is_file():
            return str(path)
    return shutil.which(exe)


def _base_interpreter_is_externally_managed() -> bool:
    """True when ``sys.executable`` is a uv/standalone-builds managed install.

    Those ship an ``EXTERNALLY-MANAGED`` marker next to their stdlib (PEP 668), so
    ``python -m pip install`` aborts with ``externally-managed-environment``; the early repair
    must then go through uv (or explicitly override pip) or the venv stays broken.

    See #83569.
    """
    try:
        import sysconfig

        stdlib = Path(sysconfig.get_path("stdlib"))
        # uv 0.5+ moved the marker one level up next to a ``_uv_managed`` sentinel dir.
        return ((stdlib / "EXTERNALLY-MANAGED").exists()
                or (stdlib.parent / "EXTERNALLY-MANAGED").exists())
    except Exception:
        return False


def _run_installer(tool: str, cmd: list[str], root: Path, env: dict | None = None) -> bool:
    """Run one installer command; captured output is replayed to stderr only on failure."""
    try:
        result = subprocess.run(cmd, cwd=root, capture_output=True, text=True, encoding="utf-8",
                                errors="replace", env=env)
    except Exception as exc:
        print(f"  ✗ Early venv repair could not run {tool}: {exc}", file=sys.stderr)
        return False
    if result.returncode == 0:
        return True
    tail = (result.stderr or result.stdout or "")[-2000:]
    if tail:
        print(tail, file=sys.stderr)
    return False


def _run_repair_install(specs: list[str], project_root: Path) -> bool:
    """``uv pip`` (or stdlib ``pip``) force-reinstall of the given specs. Never raises.

    Streams nothing to stdout (``hermes acp`` speaks JSON-RPC on stdout). uv is preferred when the
    base interpreter is externally managed; without uv, pip runs with the PEP 668 override.
    """
    externally_managed = _base_interpreter_is_externally_managed()
    if externally_managed:
        uv = _find_uv_binary()
        if uv:
            env = {**os.environ, "VIRTUAL_ENV": str(project_root / "venv")}
            env.pop("PYTHONHOME", None)
            env.pop("PYTHONPATH", None)
            repair_cmd = [uv, "pip", "install", "--force-reinstall", *specs]
            if _is_termux_env(env):
                env = prefer_termux_bionic_path(env)
                repair_cmd = with_uv_termux_python_platform(repair_cmd, env)
            return _run_installer("uv", repair_cmd, project_root, env)
        print("  ⚠ Base interpreter is externally managed and no uv binary was "
              "found; retrying repair via pip with PEP 668 override.", file=sys.stderr)
    _run_ensurepip(project_root)
    pip_cmd = [sys.executable, "-m", "pip", "install", "--force-reinstall"]
    if externally_managed:
        pip_cmd.append("--break-system-packages")
    return _run_installer("pip", pip_cmd + specs, project_root)


def _pytest_owns_live_checkout(root: Path) -> bool:
    """True under pytest when ``root`` is this module's own checkout — the venv running the suite.

    Lifecycle tests spawn real subprocesses that import ``hermes_cli.main`` with recovery armed and
    inherit ``PYTEST_CURRENT_TEST``; without this guard a broken dev venv would get a REAL
    ensurepip + force-reinstall from inside a running suite. tmp_path roots are unaffected.
    """
    return "PYTEST_CURRENT_TEST" in os.environ and root == Path(__file__).resolve().parent.parent


def recover_if_needed(project_root: Path | None = None, argv: list[str] | None = None) -> None:
    """Repair wiped core packages so ``hermes_cli.main`` can import at all.

    Fast path (no marker present) is two ``lstat`` calls. Only acts when a recovery marker from a
    prior ``hermes update`` exists AND an import probe confirms a core package is actually broken.
    Never raises: on any failure the import of main.py proceeds and surfaces the real error.
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
        # Managed/Docker/PyPI installs have no source tree here — the marker is not ours to act
        # on; main.py's recovery clears it.
        if not (root / "pyproject.toml").is_file():
            return

        # Pending core install: complete it NOW, before any native extension is imported, so the
        # WHOLE dependency set is replaced while nothing pins venv .pyd files yet (deferring to
        # main()'s post-import recovery re-locks it on Windows). A live marker owner is another
        # updater inside the marker-to-install window — never race it. A dead owner MUST be
        # recovered even when this launch is itself `hermes update`: CLI and Desktop retries keep
        # that argv, and skipping solely on argv recreates the self-lock loop.
        # Bounded retries: a persistently failing install must not hammer every launch, so attempts past the
        # ceiling are left for main.py's post-import recovery path (which can safely probe-import after this
        # process already holds whatever extensions it needs). See #83569.
        if core_marker.exists():
            if _marker_owner_is_live(core_marker):
                return
            if _complete_pending_core_install(root, core_marker) and "update" in args:
                _UPDATE_RETRY_RECOVERED = True
            return

        # The lazy-refresh marker keeps the update-argv exclusion: it is not a deferred native
        # install, and the active update flow owns its probe/repair lifecycle.
        if "update" in args:
            return

        broken = _probe_broken_packages()
        if not broken:
            return  # main.py will load and run full recovery.

        # Single-flight: share main.py's recovery lock so an early repair never races a
        # concurrent full recovery into the same shared venv.
        if not _claim_recovery_lock(root):
            return
        try:
            specs = _pinned_specs(broken, root)
            print("⚠ Core package(s) broken by an interrupted update — "
                  f"repairing before launch: {', '.join(broken)}", file=sys.stderr)
            if _run_repair_install(specs, root) and not _probe_broken_packages():
                print("  ✓ Core packages repaired.", file=sys.stderr)
            else:
                print("  ✗ Automatic repair incomplete. Recover manually with:", file=sys.stderr)
                print(f"    {sys.executable} -m pip install --force-reinstall " + " ".join(specs),
                      file=sys.stderr)
        finally:
            _release_recovery_lock(root)
    except Exception:
        pass  # Never block launch — the import of main.py will surface the truth.


# Cap on automatic early-pass install retries: a persistently failing install (network down) must
# not reinstall-hammer every launch. Past this the marker is left to main.py's post-import recovery,
# which presents the manual command. The counter lives in the marker's JSON body.
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
        # Read-only fs / perms — proceed unlocked; the install itself surfaces the real problem.
        return True


def _release_recovery_lock(root: Path) -> None:
    """Best-effort release of the shared recovery lock."""
    try:
        (root / ".update-incomplete.lock").unlink()
    except OSError:
        pass


def _complete_pending_core_install(root: Path, core_marker: Path) -> bool:
    """Run the pending core install BEFORE main.py can import native modules.

    Never raises: any failure leaves the marker for the post-import path and returns ``False``.
    Returns ``True`` only after the install succeeds.

    ``recover_if_needed`` invokes this when ``.update-incomplete`` exists — a prior ``hermes update`` (or
    the self-lock preflight, #83569) left the dependency sync deliberately unfinished. Completing it here
    matters on Windows: the deferral exists precisely because the process that wrote the marker had a native
    venv extension mapped; this process, running before ``hermes_cli.main``'s third-party imports, maps
    nothing yet, so the installer can replace ``.pyd`` files without hitting the lock.
    """
    try:
        from hermes_cli import _install_repair as ir

        # Read attempts before claiming the lock so a persistently-failing install stops early.
        attempts = _read_marker_attempts(core_marker)
        if attempts >= _EARLY_CORE_INSTALL_MAX_ATTEMPTS:
            print("⚠ Pending interrupted-update install has already failed "
                  f"{attempts} times in the early pass — leaving it for the "
                  "post-import recovery path.", file=sys.stderr)
            return False
        if not _claim_recovery_lock(root):
            return False
        try:
            print("⚠ A previous `hermes update` was interrupted mid-install — "
                  "finishing dependency installation now (before any native "
                  "extensions load)...", file=sys.stderr)
            ir.run_core_install(root)
        except Exception as exc:
            new_attempts = ir.bump_marker_attempts(core_marker)
            print(f"  ✗ Early interrupted-install completion failed (attempt "
                  f"{new_attempts}/{_EARLY_CORE_INSTALL_MAX_ATTEMPTS}): {exc}", file=sys.stderr)
            print("  The next launch will retry; hermes will keep working from "
                  "the current venv in the meantime.", file=sys.stderr)
            return False
        finally:
            _release_recovery_lock(root)

        try:
            core_marker.unlink()
        except OSError:
            pass
        print("  ✓ Dependency installation completed in the early pass.", file=sys.stderr)
        return True
    except Exception:
        return False  # Never block launch — the marker stays for the post-import path.
