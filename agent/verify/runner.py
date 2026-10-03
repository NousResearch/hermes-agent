"""Verification runner: execute a Recipe's phases and smoke-test the app.

Scoped port of the execution flow grok-cli's verify sub-agent performs
(install/bootstrap -> build -> test -> start in background -> curl-style
readiness loop -> teardown), reimplemented as a plain subprocess runner.

Commands come from the project's own recipe (its package.json scripts,
Makefile targets, etc.) and are executed with ``shell=True`` on purpose:
this is a developer tool running the project's own build commands in the
project's own checkout — the same trust level as the terminal tool. Python
recipes run in the git-ignored, project-owned ``.hermes/verify-venv`` so their
bootstrap and later phases cannot install into Hermes' caller environment.
"""

from __future__ import annotations

import errno
import hashlib
import os
import shutil
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import venv
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from agent.verify.recipes import Recipe

DEFAULT_PHASE_TIMEOUT = 600.0
DEFAULT_READY_TIMEOUT = 60.0
_TAIL_CHARS = 2000
PHASE_ORDER = ("bootstrap", "build", "test")
_PYTHON_RECIPE_KINDS = frozenset({"python", "django", "fastapi", "flask"})
_VERIFY_VENV_RELPATH = Path(".hermes") / "verify-venv"
_VERIFY_LOCK_NAME = "verify.lock"
_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _scripts_dir_for_venv(venv_dir: Path, os_name: str = os.name) -> Path:
    """Return the platform-specific command directory for a virtual environment."""
    return Path(venv_dir) / ("Scripts" if os_name == "nt" else "bin")


def _environment_for_venv(
    venv_dir: Path,
    base_environment: Mapping[str, str],
    os_name: str = os.name,
) -> dict[str, str]:
    """Build a child environment that resolves every Python command in ``venv_dir``."""
    environment = dict(base_environment)
    scripts_dir = _scripts_dir_for_venv(venv_dir, os_name=os_name)
    separator = ";" if os_name == "nt" else ":"
    inherited_path = environment.get("PATH", "")
    environment["VIRTUAL_ENV"] = str(venv_dir)
    environment["UV_PROJECT_ENVIRONMENT"] = str(venv_dir)
    environment["PATH"] = (
        f"{scripts_dir}{separator}{inherited_path}"
        if inherited_path
        else str(scripts_dir)
    )
    environment.pop("PYTHONHOME", None)
    # Ambient Python/pip/uv overrides can bypass the project environment even
    # when PATH and VIRTUAL_ENV point at it. Projects can set their own options
    # explicitly in recipe commands if they need a different destination.
    for key in (
        "PYTHONPATH",
        "PYTHONUSERBASE",
        "PIP_TARGET",
        "PIP_PREFIX",
        "PIP_USER",
        "PIP_ROOT",
        "PIP_PYTHON",
        "UV_SYSTEM_PYTHON",
        "UV_PYTHON",
    ):
        environment.pop(key, None)
    # pip may also redirect installs through user/global config files. Keep
    # recipe-owned options possible while ignoring caller-owned pip config.
    environment["PIP_CONFIG_FILE"] = os.devnull
    if os_name == "nt":
        # cmd.exe may otherwise search cwd before PATH for pip.exe/python.exe.
        environment["NoDefaultCurrentDirectoryInExePath"] = "1"
    return environment


def _python_for_venv(venv_dir: Path, os_name: str = os.name) -> Path:
    executable = "python.exe" if os_name == "nt" else "python"
    return _scripts_dir_for_venv(venv_dir, os_name=os_name) / executable


def _pip_launcher_present(venv_dir: Path) -> bool:
    scripts = _scripts_dir_for_venv(venv_dir)
    suffix = ".exe" if os.name == "nt" else ""
    names = ("pip", "pip3", f"pip{sys.version_info.major}.{sys.version_info.minor}")
    return all(
        (scripts / f"{name}{suffix}").is_file()
        and not (scripts / f"{name}{suffix}").is_symlink()
        for name in names
    )


def _path_is_redirect(
    path: Path,
    os_name: str = os.name,
    lstat: Callable[[Path], os.stat_result] = os.lstat,
) -> bool:
    """Return whether ``path`` redirects traversal via a symlink or reparse point."""
    info = lstat(path)
    if stat.S_ISLNK(info.st_mode):
        return True
    if os_name == "nt":
        attributes = getattr(info, "st_file_attributes", 0)
        return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
    return False


def _reject_redirect(path: Path) -> None:
    try:
        redirected = _path_is_redirect(path)
    except FileNotFoundError:
        return
    if redirected:
        raise OSError(f"refusing redirected verification path: {path}")


def _prepare_metadata_root(root: Path) -> Path:
    metadata = root / ".hermes"
    _reject_redirect(metadata)
    if metadata.exists() and not metadata.is_dir():
        raise OSError(f"verification metadata root is not a directory: {metadata}")
    metadata.mkdir(mode=0o700, exist_ok=True)
    _reject_redirect(metadata)
    if metadata.resolve() != metadata:
        raise OSError(f"verification metadata root escaped project: {metadata}")
    return metadata


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    """Return whether two stat results identify the same filesystem object."""
    if hasattr(left, "st_ino") and hasattr(right, "st_ino"):
        return left.st_dev == right.st_dev and left.st_ino == right.st_ino
    return False


def _open_metadata_directory(metadata: Path) -> int | None:
    """Open ``metadata`` itself when Python exposes handle-anchored directory ops.

    POSIX ``openat``/``dir_fd`` support lets us validate the metadata directory
    and create the lock file relative to that exact directory object.  This
    closes the parent-replacement gap left by path validation followed by a
    pathname ``open('.hermes/verify.lock')``.  On platforms without these
    primitives (notably native Windows in the stdlib) callers keep the existing
    path-based checks rather than claiming POSIX-strength anchoring.
    """
    if os.name == "nt" or not os.supports_dir_fd:
        return None
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(metadata, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(opened.st_mode):
            raise OSError(f"verification metadata root is not a directory: {metadata}")
        current = os.lstat(metadata)
        if not _same_file_identity(opened, current):
            raise OSError(f"verification metadata root changed during open: {metadata}")
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _verify_metadata_directory(metadata: Path, descriptor: int | None) -> None:
    """Fail closed if ``metadata`` no longer names ``descriptor``."""
    _reject_redirect(metadata)
    if descriptor is None:
        return
    opened = os.fstat(descriptor)
    current = os.lstat(metadata)
    if not _same_file_identity(opened, current):
        raise OSError(f"verification metadata root changed during use: {metadata}")


def _open_lock_file(metadata: Path, metadata_descriptor: int | None) -> Any:
    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    if metadata_descriptor is not None:
        descriptor = os.open(
            _VERIFY_LOCK_NAME, flags, 0o600, dir_fd=metadata_descriptor
        )
    else:
        lock_path = metadata / _VERIFY_LOCK_NAME
        _reject_redirect(lock_path)
        descriptor = os.open(lock_path, flags, 0o600)
    return os.fdopen(descriptor, "r+b", buffering=0)


def _lock_identity_for_root(root: Path) -> str:
    """Return a stable OS-lock identity for the current project root path.

    Threat model: the project root path is the caller-selected project boundary;
    an adversary with the same UID may mutate children such as ``.hermes`` and
    ``verify.lock`` while verification runs.  The identity therefore avoids any
    mutable child path and binds to the canonical root path plus the root
    directory's current device/inode when the platform exposes them.  Replacing
    the project root itself is outside this lock's guarantee and remains a
    caller/worktree ownership violation.
    """
    root_stat = os.stat(root)
    material = f"{os.path.normcase(str(root.resolve()))}\0{root_stat.st_dev}\0{root_stat.st_ino}"
    return hashlib.sha256(
        material.encode("utf-8", errors="surrogateescape")
    ).hexdigest()


def _try_abstract_socket_lock(root: Path, timeout: float) -> socket.socket | None:
    """Acquire a Linux abstract-namespace process lock for ``root`` if available.

    POSIX file locks are inode locks, so replacing ``.hermes/verify.lock`` at the
    same pathname can split contenders across two lock inodes.  Linux abstract
    Unix sockets are process-owned kernel objects, not filesystem entries; a
    same-path lock-file replacement cannot create a second holder for the same
    project identity.  Non-Linux platforms keep the file-lock fallback below and
    intentionally do not claim this stronger same-user mutation protection.
    """
    if not sys.platform.startswith("linux") or not hasattr(socket, "AF_UNIX"):
        return None
    address = "\0hermes-verify-" + _lock_identity_for_root(root)[:48]
    deadline = time.monotonic() + timeout
    while True:
        lock_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        lock_socket.set_inheritable(False)
        try:
            lock_socket.bind(address)
            lock_socket.listen(1)
            return lock_socket
        except OSError as exc:
            lock_socket.close()
            if exc.errno != errno.EADDRINUSE:
                raise
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for Python verification process lock for {root}"
                ) from exc
            time.sleep(0.05)


def _thread_lock_for(root: Path) -> threading.Lock:
    key = os.path.normcase(str(root))
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.Lock())


def _try_lock_file(file: Any) -> bool:
    if os.name == "nt":
        import msvcrt

        file.seek(0, os.SEEK_END)
        if file.tell() == 0:
            file.write(b"\0")
            file.flush()
        file.seek(0)
        try:
            msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    import fcntl

    try:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _unlock_file(file: Any) -> None:
    if os.name == "nt":
        import msvcrt

        file.seek(0)
        msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(file.fileno(), fcntl.LOCK_UN)


@contextmanager
def _project_python_lock(root: Path, timeout: float):
    """Serialize the complete Python verification lifecycle for one project."""
    thread_lock = _thread_lock_for(root)
    if not thread_lock.acquire(timeout=timeout):
        raise TimeoutError(f"timed out waiting for Python verification lock for {root}")
    file = None
    metadata_descriptor = None
    locked = False
    process_lock = None
    try:
        process_lock = _try_abstract_socket_lock(root, timeout)
        metadata = _prepare_metadata_root(root)
        lock_path = metadata / _VERIFY_LOCK_NAME
        metadata_descriptor = _open_metadata_directory(metadata)
        _verify_metadata_directory(metadata, metadata_descriptor)
        file = _open_lock_file(metadata, metadata_descriptor)
        if not stat.S_ISREG(os.fstat(file.fileno()).st_mode):
            raise OSError(f"verification lock is not a regular file: {lock_path}")
        _verify_metadata_directory(metadata, metadata_descriptor)
        if metadata_descriptor is None:
            _reject_redirect(lock_path)
        deadline = time.monotonic() + timeout
        while not _try_lock_file(file):
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for Python verification lock at {lock_path}"
                )
            time.sleep(0.05)
        locked = True
        _verify_metadata_directory(metadata, metadata_descriptor)
        if metadata_descriptor is None:
            _reject_redirect(lock_path)
        yield
    finally:
        if locked and file is not None:
            _unlock_file(file)
        if file is not None:
            file.close()
        if metadata_descriptor is not None:
            os.close(metadata_descriptor)
        if process_lock is not None:
            process_lock.close()
        thread_lock.release()


def _valid_python_environment(venv_dir: Path) -> bool:
    config = venv_dir / "pyvenv.cfg"
    python = _python_for_venv(venv_dir)
    if (
        not config.is_file()
        or not python.is_file()
        or not _pip_launcher_present(venv_dir)
    ):
        return False

    validation_environment = dict(os.environ)
    validation_environment.pop("PYTHONHOME", None)
    validation_environment.pop("PIP_PYTHON", None)
    try:
        proc = subprocess.run(
            [str(python), "-I", "-c", "import sys; print(sys.prefix)"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
            text=True,
            errors="replace",
            env=validation_environment,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if proc.returncode != 0:
        return False
    try:
        pip_probe = subprocess.run(
            [str(python), "-I", "-m", "pip", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
            text=True,
            errors="replace",
            env=validation_environment,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if pip_probe.returncode != 0:
        return False
    reported_prefix = proc.stdout.strip()
    if not reported_prefix:
        return False
    return os.path.normcase(str(Path(reported_prefix).resolve())) == os.path.normcase(
        str(venv_dir.resolve())
    )


def _ensure_python_environment(root: Path) -> dict[str, str]:
    """Create or reuse the project-owned verification environment."""
    metadata = _prepare_metadata_root(root)
    venv_dir = root / _VERIFY_VENV_RELPATH
    _reject_redirect(venv_dir)
    if not _valid_python_environment(venv_dir):
        _reject_redirect(metadata)
        _reject_redirect(venv_dir)
        if os.path.lexists(venv_dir):
            if venv_dir.is_dir():
                shutil.rmtree(venv_dir)
            else:
                venv_dir.unlink()
        _reject_redirect(metadata)
        venv.EnvBuilder(with_pip=True).create(str(venv_dir))
    _reject_redirect(metadata)
    _reject_redirect(venv_dir)
    if not _valid_python_environment(venv_dir):
        raise OSError(f"virtual environment validation failed at {venv_dir}")
    return _environment_for_venv(venv_dir, os.environ)


@dataclass
class PhaseResult:
    phase: str
    command: str
    exit_code: int | None
    duration: float
    output_tail: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "command": self.command,
            "exitCode": self.exit_code,
            "duration": round(self.duration, 3),
            "ok": self.ok,
            "timedOut": self.timed_out,
            "outputTail": self.output_tail,
        }


@dataclass
class ReadinessResult:
    url: str
    ready: bool
    status_code: int | None
    duration: float
    error: str | None = None
    output_tail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "ready": self.ready,
            "statusCode": self.status_code,
            "duration": round(self.duration, 3),
            "error": self.error,
            "outputTail": self.output_tail,
        }


@dataclass
class VerifyResult:
    recipe_name: str
    phases: list[PhaseResult] = field(default_factory=list)
    readiness: ReadinessResult | None = None

    @property
    def ok(self) -> bool:
        phases_ok = all(p.ok for p in self.phases)
        readiness_ok = self.readiness.ready if self.readiness is not None else True
        return phases_ok and readiness_ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe": self.recipe_name,
            "ok": self.ok,
            "phases": [p.to_dict() for p in self.phases],
            "readiness": self.readiness.to_dict() if self.readiness else None,
        }


def _tail(text: str, limit: int = _TAIL_CHARS) -> str:
    return text[-limit:] if len(text) > limit else text


def _run_phase_command(
    phase: str,
    command: str,
    root: Path,
    timeout: float,
    on_output: Callable[[str], None] | None = None,
    environment: Mapping[str, str] | None = None,
) -> PhaseResult:
    started = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            shell=True,  # project-authored commands; see module docstring
            cwd=str(root),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            text=True,
            errors="replace",
        )
        output = proc.stdout or ""
        exit_code: int | None = proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        raw = exc.output
        if isinstance(raw, bytes):
            output = raw.decode("utf-8", errors="replace")
        else:
            output = raw or ""
        exit_code = None
        timed_out = True
    duration = time.monotonic() - started
    if on_output and output:
        on_output(output)
    return PhaseResult(
        phase=phase,
        command=command,
        exit_code=exit_code,
        duration=duration,
        output_tail=_tail(output),
        timed_out=timed_out,
    )


def _poll_readiness(
    url: str, timeout: float, interval: float = 1.0
) -> tuple[bool, int | None, str | None]:
    deadline = time.monotonic() + timeout
    last_error: str | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                return True, resp.status, None
        except urllib.error.HTTPError as exc:
            # The server answered — it is up, even if it returned 4xx/5xx.
            return True, exc.code, None
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_error = str(exc)
        time.sleep(interval)
    return False, None, last_error


def _terminate_process_group(proc: subprocess.Popen) -> None:
    """Terminate the started app and its whole process group cleanly.

    On POSIX the child is spawned with ``start_new_session=True`` so we can
    signal the whole group; on Windows (no ``os.killpg``) we fall back to
    terminating just the direct child.
    """
    if proc.poll() is not None:
        return
    killpg = getattr(os, "killpg", None)
    getpgid = getattr(os, "getpgid", None)
    pgid = None
    if killpg is not None and getpgid is not None:
        try:
            pgid = getpgid(proc.pid)
        except (ProcessLookupError, PermissionError):
            pgid = None
    try:
        if pgid is not None and killpg is not None:
            killpg(
                pgid, signal.SIGTERM
            )  # windows-footgun: ok — POSIX-only branch (killpg checked above)
        else:
            proc.terminate()
    except (ProcessLookupError, PermissionError):
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            if pgid is not None and killpg is not None:
                killpg(
                    pgid, getattr(signal, "SIGKILL", signal.SIGTERM)
                )  # windows-footgun: ok — POSIX-only branch (killpg checked above)
            else:
                proc.kill()
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _run_start_phase(
    recipe: Recipe,
    root: Path,
    ready_timeout: float,
    port_override: int | None = None,
    environment: Mapping[str, str] | None = None,
) -> ReadinessResult:
    assert recipe.start is not None
    port = port_override or recipe.port or 8000
    url = f"http://127.0.0.1:{port}{recipe.readiness_path}"
    started = time.monotonic()
    proc = subprocess.Popen(
        recipe.start,
        shell=True,  # project-authored command; see module docstring
        cwd=str(root),
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # own process group for clean teardown
        text=True,
        errors="replace",
    )
    output = ""
    try:
        ready, status, error = _poll_readiness(url, ready_timeout)
    finally:
        _terminate_process_group(proc)
        try:
            if proc.stdout is not None:
                output = proc.stdout.read() or ""
        except (OSError, ValueError):
            output = ""
    return ReadinessResult(
        url=url,
        ready=ready,
        status_code=status,
        duration=time.monotonic() - started,
        error=error,
        output_tail=_tail(output),
    )


def _compose_live_state_reason(root: Path) -> str | None:
    """Refuse compose builds when existing containers might be replaced."""
    try:
        result = subprocess.run(
            ["docker", "compose", "ps", "--status", "running", "--format", "{{.Name}}"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=15,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        return (
            "docker compose ps timed out after 15s; live containers cannot be ruled out"
        )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        return f"docker compose ps failed (exit {result.returncode}): {detail[-1] if detail else 'no output'}"
    names = [line for line in result.stdout.splitlines() if line.strip()]
    return (
        f"this compose project already has running container(s): {', '.join(names)}"
        if names
        else None
    )


def run_verify(
    root: Path,
    recipe: Recipe,
    phases: tuple[str, ...] | list[str] | None = None,
    phase_timeout: float = DEFAULT_PHASE_TIMEOUT,
    ready_timeout: float = DEFAULT_READY_TIMEOUT,
    skip_start: bool = False,
    port_override: int | None = None,
    stop_on_failure: bool = True,
    on_output: Callable[[str], None] | None = None,
) -> VerifyResult:
    """Run a verify pass for ``recipe`` at project ``root``.

    Executes the selected command phases sequentially, then (unless
    ``skip_start`` or a phase failed) launches ``recipe.start`` in the
    background, polls the readiness URL, and tears the process group down.
    """
    root = Path(root).resolve()
    selected = tuple(phases) if phases else PHASE_ORDER + ("start",)
    result = VerifyResult(recipe_name=recipe.name)

    mutating = ("build" in selected) or ("start" in selected and not skip_start)
    if recipe.kind == "compose" and mutating:
        reason = _compose_live_state_reason(root)
        if reason:
            result.phases.append(
                PhaseResult(
                    phase="build",
                    command=recipe.build[0] if recipe.build else "docker compose build",
                    exit_code=1,
                    duration=0.0,
                    output_tail=(
                        f"Refusing to run: {reason}. `docker compose build` + `up` would replace "
                        "live containers on an image-hash change, destroying any container-local "
                        "state they carry. If you intend to rebuild this live deployment, run "
                        "`docker compose build`/`up` yourself."
                    ),
                )
            )
            return result

    environment: Mapping[str, str] | None = None
    lock_context = None
    lock_acquired = False
    venv_dir = (
        root / _VERIFY_VENV_RELPATH if recipe.kind in _PYTHON_RECIPE_KINDS else None
    )
    if recipe.kind in _PYTHON_RECIPE_KINDS:
        isolation_started = time.monotonic()
        lock_timeout = max(0.1, min(phase_timeout, DEFAULT_READY_TIMEOUT))
        try:
            lock_context = _project_python_lock(root, lock_timeout)
            lock_context.__enter__()
            lock_acquired = True
            environment = _ensure_python_environment(root)
        except Exception as exc:
            if lock_acquired and lock_context is not None:
                lock_context.__exit__(type(exc), exc, exc.__traceback__)
            result.phases.append(
                PhaseResult(
                    phase="isolation",
                    command=f"acquire lock; {sys.executable} -m venv {venv_dir}",
                    exit_code=1,
                    duration=time.monotonic() - isolation_started,
                    output_tail=(
                        f"Unable to lock, create, or validate the isolated Python environment at "
                        f"{venv_dir}: {type(exc).__name__}: {exc}"
                    ),
                )
            )
            return result

    try:
        failed = False
        for phase in PHASE_ORDER:
            if phase not in selected:
                continue
            for command in getattr(recipe, phase):
                if venv_dir is not None and not _pip_launcher_present(venv_dir):
                    result.phases.append(
                        PhaseResult(
                            phase="isolation",
                            command=command,
                            exit_code=1,
                            duration=0.0,
                            output_tail="Project pip launcher disappeared during verification; refusing to resolve a command through the caller's pip.",
                        )
                    )
                    return result
                phase_result = _run_phase_command(
                    phase,
                    command,
                    root,
                    phase_timeout,
                    on_output,
                    environment,
                )
                result.phases.append(phase_result)
                if not phase_result.ok:
                    failed = True
                    if stop_on_failure:
                        return result

        if skip_start or "start" not in selected or failed or not recipe.start:
            return result

        if venv_dir is not None and not _pip_launcher_present(venv_dir):
            result.phases.append(
                PhaseResult(
                    phase="isolation",
                    command=recipe.start,
                    exit_code=1,
                    duration=0.0,
                    output_tail="Project pip launcher disappeared during verification; refusing to start with a caller pip fallback.",
                )
            )
            return result

        result.readiness = _run_start_phase(
            recipe,
            root,
            ready_timeout,
            port_override,
            environment,
        )
        return result
    finally:
        if lock_acquired and lock_context is not None:
            lock_context.__exit__(None, None, None)
