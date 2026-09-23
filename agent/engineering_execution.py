"""Host-owned command execution for engineering checks and worker actions.

Only explicit backends run. The native backend isolates the environment but
does not provide filesystem or kernel isolation; Docker uses one ephemeral
container per command with no credential or home mounts.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import psutil

from agent.engineering_workflow import (
    ReceiptError,
    VerificationContext,
    VerificationReceipt,
)


class ExecutionUnavailable(RuntimeError):
    """The explicitly selected execution backend could not be started."""


@dataclass(frozen=True)
class CheckSpec:
    check_id: str
    argv: tuple[str, ...]
    timeout: float


@dataclass(frozen=True)
class HostProcessResult:
    exit_code: int
    complete: bool
    timed_out: bool
    output: str = ""


_RUNTIME_ENV_KEYS = (
    "PATH",
    "Path",
    "PATHEXT",
    "SYSTEMROOT",
    "SystemRoot",
    "WINDIR",
    "COMSPEC",
    "LANG",
    "LC_ALL",
    "TZ",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
)


_CREDENTIAL_FILE_NAMES = frozenset({
    ".env",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "id_rsa",
    "id_ed25519",
    "credentials",
    "credentials.json",
    "service-account.json",
})


def _is_link_or_reparse(path: Path) -> bool:
    attrs = getattr(path.lstat(), "st_file_attributes", 0)
    return path.is_symlink() or bool(
        attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def _assert_mountable_workspace(root: Path) -> None:
    """Refuse Docker bind mounts that would expose common credential stores."""
    examined = 0
    for parent, dirs, names in os.walk(root, followlinks=False):
        for name in (*dirs, *names):
            examined += 1
            if examined > 100_000:
                raise ExecutionUnavailable(
                    "Docker workspace has too many entries to inspect"
                )
            path = Path(parent) / name
            if _is_link_or_reparse(path):
                raise ExecutionUnavailable("Docker workspace contains a linked path")
            lower = name.casefold()
            if lower in _CREDENTIAL_FILE_NAMES or lower.startswith(".env."):
                raise ExecutionUnavailable(
                    "Docker workspace contains a credential file"
                )
            if name in {".venv", "venv", "node_modules"} and path.is_dir():
                raise ExecutionUnavailable(
                    "Docker workspace contains local dependency storage"
                )


def _fresh_env(isolated_home: Path) -> dict[str, str]:
    env = {name: os.environ[name] for name in _RUNTIME_ENV_KEYS if name in os.environ}
    home = str(isolated_home)
    env.update(
        HOME=home,
        USERPROFILE=home,
        TMP=home,
        TEMP=home,
        TMPDIR=home,
        PYTHONUTF8="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    return env


def _docker_argv(
    docker: str,
    argv: tuple[str, ...],
    workspace: Path,
    image: str,
    name: str,
) -> list[str]:
    if not image or any(mark in image for mark in ("\x00", "\n", "\r")):
        raise ExecutionUnavailable("Docker image is required")
    user = f"{os.getuid()}:{os.getgid()}" if hasattr(os, "getuid") else "1000:1000"
    return [
        docker,
        "run",
        "--rm",
        "--name",
        name,
        "--pull=never",
        "--network=none",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=128",
        "--memory=2g",
        "--read-only",
        "--env",
        "HOME=/tmp",
        "--env",
        "TMPDIR=/tmp",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--tmpfs=/tmp:rw,noexec,nosuid,size=256m",
        "--user",
        user,
        "--workdir=/workspace",
        "--mount",
        f"type=bind,source={workspace},target=/workspace",
        image,
        *argv,
    ]


def run_host_command(
    argv: tuple[str, ...],
    workspace: Path,
    *,
    backend: str,
    timeout: float,
    image: str = "",
    stop_requested=None,
) -> HostProcessResult:
    """Run one argv without a shell and with a fresh allow-listed environment."""
    if (
        not isinstance(argv, tuple)
        or not argv
        or any(not isinstance(arg, str) or not arg or "\x00" in arg for arg in argv)
        or sum(len(arg) for arg in argv) > 16_384
        or type(timeout) not in (int, float)
        or timeout <= 0
    ):
        raise ValueError("invalid command or timeout")
    root = Path(workspace).resolve(strict=True)
    if not root.is_dir():
        raise ExecutionUnavailable("workspace is not a directory")
    if backend == "docker":
        if (root / ".git").is_dir():
            raise ExecutionUnavailable(
                "Docker requires a worktree without a mounted Git credential store"
            )
        _assert_mountable_workspace(root)
        docker = shutil.which("docker")
        if not docker:
            raise ExecutionUnavailable("Docker backend unavailable")
        container_name = f"hermes-engineering-{uuid.uuid4().hex}"
        command = _docker_argv(docker, argv, root, image, container_name)
    elif backend == "native":
        command = list(argv)
    else:
        raise ExecutionUnavailable("unknown execution backend")

    # The transient home is outside the source snapshot; a regular clone
    # keeps it in .git, while a linked worktree uses the OS temporary root.
    git_dir = root / ".git"
    runtime_parent = git_dir if git_dir.is_dir() else None
    with tempfile.TemporaryDirectory(
        prefix=".engineering-runtime-", dir=runtime_parent
    ) as tmp:
        with tempfile.TemporaryFile(mode="w+b", dir=runtime_parent) as output:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=root,
                    env=_fresh_env(Path(tmp)),
                    shell=False,
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
            except OSError as exc:
                raise ExecutionUnavailable("execution backend could not start") from exc
            deadline = time.monotonic() + timeout
            timed_out = oversized = interrupted = False
            while process.poll() is None:
                timed_out = time.monotonic() >= deadline
                oversized = os.fstat(output.fileno()).st_size > 64 * 1024
                interrupted = bool(stop_requested and stop_requested())
                if timed_out or oversized or interrupted:
                    _stop_process_tree(process)
                    break
                time.sleep(0.05)
            process.wait()
            oversized |= os.fstat(output.fileno()).st_size > 64 * 1024
            output.seek(0)
            preview = output.read(8 * 1024).decode("utf-8", errors="replace")
            if backend == "docker" and (timed_out or oversized or interrupted):
                try:
                    cleanup = subprocess.run(
                        [docker, "rm", "-f", container_name],
                        cwd=root,
                        env=_fresh_env(Path(tmp)),
                        shell=False,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=10,
                        check=False,
                    )
                except (OSError, subprocess.TimeoutExpired) as exc:
                    raise ExecutionUnavailable(
                        "Docker container cleanup could not be confirmed"
                    ) from exc
                if cleanup.returncode != 0:
                    raise ExecutionUnavailable(
                        "Docker container cleanup could not be confirmed"
                    )
    if backend == "docker" and process.returncode == 125:
        raise ExecutionUnavailable("Docker could not start the isolated command")
    if timed_out or oversized or interrupted:
        reason = (
            "command interrupted"
            if interrupted
            else "command timed out"
            if timed_out
            else "command output exceeded limit"
        )
        return HostProcessResult(
            exit_code=-1,
            complete=False,
            timed_out=timed_out,
            output=reason,
        )
    return HostProcessResult(
        exit_code=process.returncode,
        complete=True,
        timed_out=False,
        output=_redact_env_values(preview),
    )


def _redact_env_values(text: str) -> str:
    """Prevent known parent credential values in command output reaching a model."""
    for name, value in os.environ.items():
        upper = name.upper()
        if len(value) >= 8 and any(
            word in upper
            for word in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
        ):
            text = text.replace(value, "[redacted]")
    return text


def _stop_process_tree(process: subprocess.Popen) -> None:
    """Stop only the process this executor launched and its observed descendants."""
    try:
        descendants = psutil.Process(process.pid).children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        descendants = []
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)
    for child in descendants:
        try:
            if child.is_running():
                child.terminate()
                child.wait(timeout=0.5)
        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass


def execute_checks(
    context: VerificationContext,
    checks: tuple[CheckSpec, ...],
    workspace: Path,
    *,
    backend: str,
    snapshot_digest,
    image: str = "",
    stop_requested=None,
) -> list[VerificationReceipt]:
    """Mint receipts only after running each operator-owned check on one snapshot."""
    if len(checks) != len(context.check_ids) or {
        check.check_id for check in checks
    } != set(context.check_ids):
        raise ReceiptError("check set does not match verification context")
    if snapshot_digest() != context.snapshot_digest:
        raise ReceiptError("workspace changed before verification")
    receipts = []
    for check in checks:
        result = run_host_command(
            check.argv,
            workspace,
            backend=backend,
            timeout=check.timeout,
            image=image,
            stop_requested=stop_requested,
        )
        if snapshot_digest() != context.snapshot_digest:
            raise ReceiptError("workspace changed during verification")
        receipts.append(
            VerificationReceipt(
                run_id=context.run_id,
                workspace_id=context.workspace_id,
                attempt_id=context.attempt_id,
                revision=context.revision,
                snapshot_digest=context.snapshot_digest,
                check_id=check.check_id,
                exit_code=result.exit_code,
                complete=result.complete,
                timed_out=result.timed_out,
            )
        )
        if not result.complete:
            break
    return receipts


_EXCLUDED_SNAPSHOT_DIRS = frozenset({
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
})


def workspace_digest(workspace: Path) -> str:
    """Hash a bounded source snapshot; reject symlinks and moving files."""
    root = Path(workspace).resolve(strict=True)
    digest = hashlib.sha256()
    files = 0
    bytes_seen = 0
    for parent, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(name for name in dirs if name not in _EXCLUDED_SNAPSHOT_DIRS)
        for name in dirs:
            if _is_link_or_reparse(Path(parent) / name):
                raise ReceiptError("workspace contains a linked directory")
        for name in sorted(names):
            path = Path(parent) / name
            if _is_link_or_reparse(path):
                raise ReceiptError("workspace contains a linked file")
            if not path.is_file():
                continue
            files += 1
            if files > 50_000:
                raise ReceiptError("workspace has too many files to verify")
            relative = path.relative_to(root).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(4, "big"))
            digest.update(relative)
            before = path.stat()
            with path.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    bytes_seen += len(block)
                    if bytes_seen > 2 * 1024 * 1024 * 1024:
                        raise ReceiptError("workspace exceeds verification size limit")
                    digest.update(block)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ReceiptError("workspace changed while computing its snapshot")
    return digest.hexdigest()
