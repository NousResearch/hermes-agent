"""Startup requests for PM recovery and rescue of orphaned launchers."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Older installers left renamed launchers behind on failure. The startup
# orphan sweep still restores them. Generation installs never rename live shims.
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


def _read_marker_attempts(marker_path: Path) -> int:
    """Attempt counter from a marker's opportunistic JSON body; corrupt/missing → 0."""
    try:
        raw = marker_path.read_text(encoding="utf-8-sig", errors="replace").strip()
    except OSError:
        return 0
    if not raw:
        return 0
    try:
        import json

        return int(json.loads(raw).get("attempts", 0))
    except (ValueError, AttributeError, TypeError):
        for line in reversed(raw.splitlines()):
            key, separator, value = line.partition("=")
            if separator and key.strip() == "attempts":
                try:
                    return max(0, int(value))
                except ValueError:
                    return 0
        return 0


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
        body = marker.read_text(encoding="utf-8-sig", errors="replace")
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


def _pytest_owns_live_checkout(root: Path) -> bool:
    """Keep lifecycle tests from repairing the checkout that runs the suite.

    Copied installations in tmp_path remain eligible for real recovery tests.
    """
    return "PYTEST_CURRENT_TEST" in os.environ and root == Path(__file__).resolve().parent.parent


def recover_if_needed(project_root: Path | None = None, argv: list[str] | None = None, *, explicit: bool = False) -> bool:
    """Ask PM to restore dependencies before activation; leave failed requests retryable."""
    global _UPDATE_RETRY_RECOVERED

    root = _project_root() if project_root is None else Path(project_root).resolve()
    if not explicit and _pytest_owns_live_checkout(root):
        return False
    from hermes_cli._parser import command_argv

    args = command_argv(sys.argv[1:] if argv is None else argv)
    if not explicit and args[:1] == ["pm"]:
        return False  # PM's command boundary owns the explicit repair.
    from hermes_cli.runtime_paths import install_state_dir, runtime_facts_path, selected_venv, site_packages

    missing_marker = install_state_dir(root) / ".repair-incomplete"
    marker_paths = (root / ".update-incomplete", root / ".lazy-refresh-incomplete", missing_marker)
    markers = [path for path in marker_paths if path.is_file()]

    def missing_environment():
        if not runtime_facts_path(root).is_file():
            return False
        try:
            return not site_packages(selected_venv(root)).is_dir()
        except RuntimeError:
            return True  # PM validates the recorded state before rebuilding.

    if not (root / "pyproject.toml").is_file() or not (explicit or markers or missing_environment()):
        return False
    lock = _claim_recovery_lock(root)
    if lock is None:
        return False
    try:
        # Recheck after locking: another launch can finish between discovery and claim.
        markers = [path for path in marker_paths if path.is_file()]
        if not markers:
            if not explicit and not missing_environment():
                return False
            missing_marker.write_text('{"attempts": 0}', encoding="utf-8")
            markers = [missing_marker]
        if any(_marker_owner_is_live(marker) for marker in markers):
            return False
        if not explicit and any(_read_marker_attempts(marker) >= _EARLY_CORE_INSTALL_MAX_ATTEMPTS for marker in markers):
            print("hermes: automatic dependency repair retry limit reached; run `hermes pm repair`", file=sys.stderr)
            return False
        from pm.recovery import repair_dependencies

        print("hermes: repairing the recorded dependency environment...", file=sys.stderr)
        repair_dependencies(root)
        for marker in markers:
            marker.unlink(missing_ok=True)
        _UPDATE_RETRY_RECOVERED = args[:1] == ["update"]
        print("hermes: dependency environment repaired", file=sys.stderr)
        return True
    except Exception as exc:
        import json

        for marker in markers:
            try:
                attempts = _read_marker_attempts(marker) + 1
                body = marker.read_text(encoding="utf-8-sig")
                if any(line.startswith("pid=") for line in body.splitlines()):
                    lines = [line for line in body.splitlines() if not line.startswith("attempts=")]
                    body = "\n".join([*lines, f"attempts={attempts}"]) + "\n"
                else:
                    body = json.dumps({"attempts": attempts})
                marker.write_text(body, encoding="utf-8")
            except OSError:
                pass
        print(f"hermes: dependency repair failed: {exc}; run `hermes pm repair`", file=sys.stderr)
        return False
    finally:
        os.close(lock)


# A failed network or build must not retry on every launch forever.
_EARLY_CORE_INSTALL_MAX_ATTEMPTS = 3


def _claim_recovery_lock(root: Path) -> int | None:
    """Hold a kernel lock in writable state; process exit releases it."""
    from hermes_cli.runtime_paths import install_state_dir
    from hermes_cli.runtime_state import _lock

    state = install_state_dir(root)
    state.mkdir(parents=True, exist_ok=True)
    fd = os.open(state / ".recovery.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if _lock(fd, wait=False):
            return fd
    except BaseException:
        os.close(fd)
        raise
    os.close(fd)
    return None
