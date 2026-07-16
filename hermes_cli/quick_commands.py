"""Validation and bounded I/O helpers for deterministic quick commands."""

from __future__ import annotations

import asyncio
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from typing import Any

import psutil


QUICK_COMMAND_TIMEOUT_SECONDS = 30
QUICK_COMMAND_INPUT_MAX_BYTES = 8192
QUICK_COMMAND_OUTPUT_MAX_BYTES = 65536
QUICK_COMMAND_METADATA_MAX_BYTES = 256
QUICK_COMMAND_DESTINATION_ALIAS_MAX_BYTES = 64

_OUTPUT_READ_CHUNK_BYTES = 8192
_PROCESS_STOP_GRACE_SECONDS = 2
_DESCENDANT_POLL_SECONDS = 0.05

_DESTINATION_ALIAS_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_TRUSTED_BASE_ENV_KEYS = (
    "HOME",
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "TMPDIR",
    # Windows process creation and executable lookup.
    "SYSTEMROOT",
    "PATHEXT",
)


class QuickCommandConfigError(ValueError):
    """Raised when a quick-command definition is unsafe or malformed."""


class QuickCommandOutputError(ValueError):
    """Raised when deterministic command output cannot be returned safely."""


def run_bounded_argv(
    argv: list[str],
    *,
    env: Mapping[str, str],
    timeout: float = QUICK_COMMAND_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[bytes]:
    """Run exact argv while enforcing the combined output cap as bytes arrive."""
    proc = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=dict(env),
        **_new_process_group_kwargs(),
    )
    descendant_tracker = _DescendantTracker(proc.pid)
    try:
        descendant_tracker.start()
        return _communicate_bounded_sync(proc, argv=argv, timeout=timeout)
    except BaseException:
        # Mirror the async path: cancellation and unexpected caller-thread
        # exceptions must not bypass leader/original-group cleanup.
        _terminate_sync_process(proc)
        raise
    finally:
        descendant_tracker.stop_and_terminate()


def _communicate_bounded_sync(
    proc: subprocess.Popen[bytes],
    *,
    argv: list[str],
    timeout: float,
) -> subprocess.CompletedProcess[bytes]:
    """Complete one already-spawned sync command with bounded output."""
    if proc.stdout is None or proc.stderr is None:  # pragma: no cover - Popen contract
        _terminate_sync_process(proc)
        raise QuickCommandOutputError("output streams are unavailable")

    stdout = bytearray()
    stderr = bytearray()
    output_lock = threading.Lock()
    overflow = threading.Event()
    reader_errors: list[BaseException] = []

    def _read_stream(stream: Any, destination: bytearray) -> None:
        try:
            while True:
                read_available = getattr(stream, "read1", stream.read)
                chunk = read_available(_OUTPUT_READ_CHUNK_BYTES)
                if not chunk:
                    return
                with output_lock:
                    if len(stdout) + len(stderr) + len(chunk) > QUICK_COMMAND_OUTPUT_MAX_BYTES:
                        overflow.set()
                        return
                    destination.extend(chunk)
        except BaseException as exc:  # surfaced on the caller thread
            reader_errors.append(exc)
        finally:
            stream.close()

    readers = [
        threading.Thread(
            target=_read_stream,
            args=(proc.stdout, stdout),
            name="quick-command-stdout",
            daemon=True,
        ),
        threading.Thread(
            target=_read_stream,
            args=(proc.stderr, stderr),
            name="quick-command-stderr",
            daemon=True,
        ),
    ]
    for reader in readers:
        reader.start()

    deadline = time.monotonic() + timeout
    timed_out = False
    while proc.poll() is None and not overflow.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        overflow.wait(min(0.05, remaining))

    if timed_out or overflow.is_set():
        _terminate_sync_process(proc)
    else:
        proc.wait()
    for reader in readers:
        reader.join(_PROCESS_STOP_GRACE_SECONDS)

    if timed_out:
        raise subprocess.TimeoutExpired(
            argv,
            timeout,
            output=bytes(stdout),
            stderr=bytes(stderr),
        )
    if overflow.is_set():
        raise QuickCommandOutputError(
            f"output exceeds {QUICK_COMMAND_OUTPUT_MAX_BYTES} UTF-8 bytes"
        )
    if any(reader.is_alive() for reader in readers):
        # The argv leader may have exited after forking a descendant that
        # inherited our pipes. Kill the dedicated process group before
        # reporting the hung readers so no background child survives.
        _terminate_sync_process(proc)
        for reader in readers:
            reader.join(_PROCESS_STOP_GRACE_SECONDS)
        raise QuickCommandOutputError("output streams did not close")
    if reader_errors:
        _terminate_sync_process(proc)
        raise OSError("quick-command output read failed") from reader_errors[0]

    _cleanup_completed_sync_process_tree(proc)

    return subprocess.CompletedProcess(
        argv,
        proc.returncode,
        stdout=bytes(stdout),
        stderr=bytes(stderr),
    )


async def communicate_bounded_async(
    proc: asyncio.subprocess.Process,
    *,
    timeout: float = QUICK_COMMAND_TIMEOUT_SECONDS,
) -> tuple[bytes, bytes]:
    """Read an asyncio subprocess without ever buffering beyond the output cap."""
    descendant_tracker = _DescendantTracker(proc.pid)
    try:
        descendant_tracker.start()
        return await _communicate_bounded_async(proc, timeout=timeout)
    finally:
        await _stop_descendant_tracker_async(descendant_tracker)


async def _communicate_bounded_async(
    proc: asyncio.subprocess.Process,
    *,
    timeout: float,
) -> tuple[bytes, bytes]:
    """Complete one already-spawned async command with bounded output."""
    if proc.stdout is None or proc.stderr is None:
        await _terminate_async_process(proc)
        raise QuickCommandOutputError("output streams are unavailable")

    stdout = bytearray()
    stderr = bytearray()

    async def _read_stream(
        stream: asyncio.StreamReader, destination: bytearray
    ) -> None:
        while True:
            chunk = await stream.read(_OUTPUT_READ_CHUNK_BYTES)
            if not chunk:
                return
            # This block has no await, so the event loop makes the combined
            # check-and-append atomic across the two reader tasks.
            if len(stdout) + len(stderr) + len(chunk) > QUICK_COMMAND_OUTPUT_MAX_BYTES:
                raise QuickCommandOutputError(
                    f"output exceeds {QUICK_COMMAND_OUTPUT_MAX_BYTES} UTF-8 bytes"
                )
            destination.extend(chunk)

    readers = [
        asyncio.create_task(_read_stream(proc.stdout, stdout)),
        asyncio.create_task(_read_stream(proc.stderr, stderr)),
    ]
    deadline = asyncio.get_running_loop().time() + timeout
    try:
        await asyncio.wait_for(asyncio.gather(*readers), timeout=timeout)
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise asyncio.TimeoutError
        await asyncio.wait_for(proc.wait(), timeout=remaining)
        await _cleanup_completed_async_process_tree(proc)
    except BaseException:
        for reader in readers:
            reader.cancel()
        await asyncio.gather(*readers, return_exceptions=True)
        await _terminate_async_process(proc)
        raise

    return bytes(stdout), bytes(stderr)


class _DescendantTracker:
    """Track identity-bound descendants while a configured command is live.

    Process-group cleanup remains the primary containment mechanism. Polling the
    verified leader and already-observed descendants additionally catches normal
    children that create a new session and escape that group. Every later lookup
    is bound to the observed process creation time so PID reuse cannot redirect a
    cleanup signal.

    This protects the trusted configured-command boundary. It does not claim to
    observe a deliberately unobservable double-fork that fully reparents before
    the leader or an observed descendant can be sampled.
    """

    def __init__(self, leader_pid: int) -> None:
        self._leader = _process_identity(leader_pid)
        self._descendants: dict[int, float] = {}
        self._identities_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._leader is None:
            return
        self._observe_once()
        self._thread = threading.Thread(
            target=self._observe_until_stopped,
            name="quick-command-descendants",
            daemon=True,
        )
        self._thread.start()

    def stop_and_terminate(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._observe_once()
        with self._identities_lock:
            identities = dict(self._descendants)

        terminating: list[psutil.Process] = []
        for pid, create_time in identities.items():
            process = _verified_process(pid, create_time)
            if process is None:
                continue
            try:
                process.terminate()
            except (psutil.Error, OSError):
                continue
            terminating.append(process)

        if not terminating:
            return
        _, alive = psutil.wait_procs(
            terminating, timeout=_PROCESS_STOP_GRACE_SECONDS
        )
        killed: list[psutil.Process] = []
        for process in alive:
            create_time = identities.get(process.pid)
            if create_time is None:
                continue
            verified = _verified_process(process.pid, create_time)
            if verified is None:
                continue
            try:
                verified.kill()
            except (psutil.Error, OSError):
                continue
            killed.append(verified)
        if killed:
            psutil.wait_procs(killed, timeout=_PROCESS_STOP_GRACE_SECONDS)

    def _observe_until_stopped(self) -> None:
        while not self._stop.wait(_DESCENDANT_POLL_SECONDS):
            self._observe_once()

    def _observe_once(self) -> None:
        if self._leader is None:
            return
        with self._identities_lock:
            roots = [self._leader, *self._descendants.items()]

        discovered: dict[int, float] = {}
        for pid, create_time in roots:
            process = _verified_process(pid, create_time)
            if process is None:
                continue
            try:
                children = process.children(recursive=True)
            except (psutil.Error, OSError):
                continue
            for child in children:
                identity = _identity_from_process(child)
                if identity is None or identity[0] == self._leader[0]:
                    continue
                discovered[identity[0]] = identity[1]

        if discovered:
            with self._identities_lock:
                self._descendants.update(discovered)


def _process_identity(pid: int) -> tuple[int, float] | None:
    """Return a PID-reuse-safe process identity when the process is inspectable."""
    if pid == os.getpid():
        return None
    try:
        return _identity_from_process(psutil.Process(pid))
    except (psutil.Error, OSError):
        return None


def _identity_from_process(process: psutil.Process) -> tuple[int, float] | None:
    try:
        identity = process.pid, process.create_time()
        if not process.is_running():
            return None
        return identity
    except (psutil.Error, OSError):
        return None


def _verified_process(pid: int, create_time: float) -> psutil.Process | None:
    """Reopen a process only when it still has the exact observed identity."""
    if pid == os.getpid():
        return None
    try:
        process = psutil.Process(pid)
        if _identity_from_process(process) != (pid, create_time):
            return None
        return process
    except (psutil.Error, OSError):
        return None


async def _stop_descendant_tracker_async(tracker: _DescendantTracker) -> None:
    """Finish blocking descendant cleanup even if the caller is cancelled."""
    cleanup = asyncio.create_task(asyncio.to_thread(tracker.stop_and_terminate))
    try:
        await asyncio.shield(cleanup)
    except asyncio.CancelledError:
        await cleanup
        raise


def _new_process_group_kwargs() -> dict[str, Any]:
    """Return platform-safe kwargs for a dedicated quick-command group."""
    if sys.platform == "win32":
        return {
            "creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)
        }
    return {"start_new_session": True}


def _posix_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        # Treat an indeterminate result as live and fail toward cleanup.
        return True


def _terminate_windows_tree(pid: int) -> None:
    try:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )  # windows-footgun: ok -- Windows-only process-tree primitive
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        pass


def _cleanup_completed_sync_process_tree(proc: subprocess.Popen[bytes]) -> None:
    """Reap descendants left behind by an otherwise completed argv leader."""
    if sys.platform == "win32":
        # The process handle retained by Popen prevents PID reuse while the
        # Windows tree primitive walks children of the completed leader.
        _terminate_windows_tree(proc.pid)
    elif _posix_group_exists(proc.pid):
        # ``proc.wait()`` already reaped the leader, so a surviving dedicated
        # group can only contain descendants created by this argv command.
        _terminate_sync_process(proc)


async def _cleanup_completed_async_process_tree(
    proc: asyncio.subprocess.Process,
) -> None:
    """Async counterpart of :func:`_cleanup_completed_sync_process_tree`."""
    if sys.platform == "win32":
        await asyncio.to_thread(_terminate_windows_tree, proc.pid)
    elif _posix_group_exists(proc.pid):
        await _terminate_async_process(proc)


def _terminate_sync_process(proc: subprocess.Popen[bytes]) -> None:
    """Terminate, escalate, and reap the child's entire process tree."""
    pid = proc.pid
    if sys.platform == "win32":
        _terminate_windows_tree(pid)
    else:
        try:
            os.killpg(pid, signal.SIGTERM)  # windows-footgun: ok -- POSIX branch
        except (ProcessLookupError, PermissionError, OSError):
            pass
        deadline = time.monotonic() + _PROCESS_STOP_GRACE_SECONDS
        while _posix_group_exists(pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        if _posix_group_exists(pid):
            try:
                os.killpg(pid, signal.SIGKILL)  # windows-footgun: ok -- POSIX branch
            except (ProcessLookupError, PermissionError, OSError):
                pass
    try:
        proc.wait(timeout=_PROCESS_STOP_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        proc.wait()


async def _terminate_async_process(proc: asyncio.subprocess.Process) -> None:
    """Terminate, escalate, and reap the child's entire process tree."""
    pid = proc.pid
    if sys.platform == "win32":
        await asyncio.to_thread(_terminate_windows_tree, pid)
    else:
        try:
            os.killpg(pid, signal.SIGTERM)  # windows-footgun: ok -- POSIX branch
        except (ProcessLookupError, PermissionError, OSError):
            pass
        deadline = asyncio.get_running_loop().time() + _PROCESS_STOP_GRACE_SECONDS
        while _posix_group_exists(pid) and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.01)
        if _posix_group_exists(pid):
            try:
                os.killpg(pid, signal.SIGKILL)  # windows-footgun: ok -- POSIX branch
            except (ProcessLookupError, PermissionError, OSError):
                pass
    try:
        await asyncio.wait_for(proc.wait(), timeout=_PROCESS_STOP_GRACE_SECONDS)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()


def prepare_argv_command(qcmd: Any, argument_text: str) -> list[str]:
    """Validate an ``argv`` quick command and return its exact process argv."""
    if not isinstance(qcmd, Mapping):
        raise QuickCommandConfigError("configuration must be a mapping")

    command = qcmd.get("command")
    if not isinstance(command, list) or not command:
        raise QuickCommandConfigError("command must be a non-empty list of strings")
    if any(
        not isinstance(item, str) or not item.strip() or "\x00" in item
        for item in command
    ):
        raise QuickCommandConfigError("command must contain only non-empty strings")

    argument_mode = qcmd.get("argument_mode", "none")
    if argument_mode not in ("none", "text"):
        raise QuickCommandConfigError("argument_mode must be 'none' or 'text'")

    argv = list(command)
    if argument_mode == "text":
        if not isinstance(argument_text, str) or not argument_text.strip():
            raise QuickCommandConfigError("argument_mode 'text' requires text")
        input_bytes = len(argument_text.encode("utf-8"))
        if input_bytes > QUICK_COMMAND_INPUT_MAX_BYTES:
            raise QuickCommandConfigError(
                f"text exceeds {QUICK_COMMAND_INPUT_MAX_BYTES} UTF-8 bytes"
            )
        if "\x00" in argument_text:
            raise QuickCommandConfigError("text must not contain NUL bytes")
        # One exact argv item: metacharacters and whitespace are never parsed by
        # a shell and therefore cannot change the configured executable/flags.
        argv.append(argument_text)
    return argv


def build_argv_environment(extra: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Build a minimal child environment without Hermes-managed credentials."""
    child_env = {
        key: value
        for key in _TRUSTED_BASE_ENV_KEYS
        if (value := os.environ.get(key)) is not None and "\x00" not in value
    }
    # ``HERMES_HOME`` may be a context-local profile override rather than a
    # process environment variable. Preserve the resolved home explicitly so
    # named CLI/TUI and multiplexed gateway argv children stay in their lane.
    from hermes_constants import get_hermes_home

    child_env["HERMES_HOME"] = str(get_hermes_home())
    for name, value in (extra or {}).items():
        child_env[name] = _bounded_metadata(value, name)
    return child_env


def build_gateway_argv_environment(
    qcmd: Any,
    *,
    platform: Any,
    message_id: Any,
    update_id: Any,
) -> dict[str, str]:
    """Build a secret-free gateway child environment with request provenance."""
    if not isinstance(qcmd, Mapping):
        raise QuickCommandConfigError("configuration must be a mapping")

    destination_alias = qcmd.get("destination_alias")
    if destination_alias is not None:
        if not isinstance(destination_alias, str) or not destination_alias:
            raise QuickCommandConfigError("destination_alias must be a non-empty string")
        if (
            len(destination_alias.encode("utf-8"))
            > QUICK_COMMAND_DESTINATION_ALIAS_MAX_BYTES
            or not _DESTINATION_ALIAS_RE.fullmatch(destination_alias)
        ):
            raise QuickCommandConfigError(
                "destination_alias must be 1-64 ASCII letters, digits, '.', '_', ':', or '-'"
            )

    provenance = {
            "HERMES_QUICK_COMMAND_PLATFORM": _bounded_metadata(
                platform, "platform", allow_empty=False
            ),
            "HERMES_QUICK_COMMAND_MESSAGE_ID": _bounded_metadata(
                message_id, "message_id"
            ),
            "HERMES_QUICK_COMMAND_UPDATE_ID": _bounded_metadata(
                update_id, "update_id"
            ),
        }
    if destination_alias is not None:
        provenance["HERMES_QUICK_COMMAND_DESTINATION_ALIAS"] = destination_alias
    return build_argv_environment(provenance)


def bounded_quick_command_output(stdout: Any, stderr: Any) -> str:
    """Return redacted stdout (or stderr fallback) after a combined byte cap."""
    stdout_text = _output_text(stdout)
    stderr_text = _output_text(stderr)
    output_bytes = len(stdout_text.encode("utf-8")) + len(stderr_text.encode("utf-8"))
    if output_bytes > QUICK_COMMAND_OUTPUT_MAX_BYTES:
        raise QuickCommandOutputError(
            f"output exceeds {QUICK_COMMAND_OUTPUT_MAX_BYTES} UTF-8 bytes"
        )

    output = stdout_text.strip() or stderr_text.strip()
    if not output:
        return ""
    try:
        from agent.redact import redact_sensitive_text

        # Deterministic command output crosses a user-visible gateway boundary;
        # force redaction even when the general display preference is disabled.
        return redact_sensitive_text(output, force=True)
    except Exception as exc:
        raise QuickCommandOutputError("output could not be safely redacted") from exc


def _bounded_metadata(
    value: Any, name: str, *, allow_empty: bool = True
) -> str:
    text = "" if value is None else str(value)
    if not allow_empty and not text:
        raise QuickCommandConfigError(f"{name} is required")
    if "\x00" in text:
        raise QuickCommandConfigError(f"{name} must not contain NUL bytes")
    if len(text.encode("utf-8")) > QUICK_COMMAND_METADATA_MAX_BYTES:
        raise QuickCommandConfigError(
            f"{name} exceeds {QUICK_COMMAND_METADATA_MAX_BYTES} UTF-8 bytes"
        )
    return text


def _output_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
