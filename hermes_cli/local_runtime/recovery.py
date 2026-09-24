"""Conservative recovery of the one router recorded by this managed runtime."""
from __future__ import annotations

import json
import logging
import math
import os
import stat
import time
from contextlib import contextmanager
from pathlib import Path

import psutil

logger = logging.getLogger(__name__)


@contextmanager
def state_lock(timeout_s: float = 2.0):
    """Serialize reset with publications, including watchdog restarts; never proceed unlocked."""
    from hermes_constants import mkdir_under_hermes_home
    from hermes_cli.local_runtime.bootstrap import _try_lock_boot_fd, _unlock_boot_fd
    from hermes_cli.local_runtime.supervisor import state_path

    path = state_path().with_suffix(".lock")
    mkdir_under_hermes_home(path.parent)
    try:
        if not stat.S_ISREG(path.lstat().st_mode):
            raise OSError("local runtime lock is not a regular file")
    except FileNotFoundError:
        pass
    fd = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    acquired = False
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError("local runtime lock is not a regular file")
        deadline = time.monotonic() + timeout_s
        while not (acquired := _try_lock_boot_fd(fd)):
            if time.monotonic() >= deadline:
                raise TimeoutError("local runtime state is busy; try again")
            time.sleep(0.05)
        yield
    finally:
        try:
            if acquired:
                _unlock_boot_fd(fd)
        finally:
            os.close(fd)


def _stale_record(state: dict, modified: float) -> bool:
    """Positive evidence of a dead/reused identity, not merely failure to verify ownership."""
    from urllib.parse import urlsplit
    from hermes_cli.local_runtime.supervisor import state_path

    try:
        if not isinstance(state, dict) or not _valid_pid(state.get("pid")):
            return False
        base_url = state.get("base_url")
        if not isinstance(base_url, str) or not base_url:
            return False
        url = urlsplit(base_url)
        if (url.scheme != "http" or url.hostname != "127.0.0.1" or not url.port
                or not isinstance(state.get("api_key"), str) or not state["api_key"]):
            return False
        modern = is_modern(state)
        if modern:
            if (not _valid_birth(state.get("create_time"))
                    or not _valid_pid(state.get("owner_pid"))
                    or not _valid_birth(state.get("owner_create_time"))
                    or state["owner_create_time"] > state["create_time"]
                    or not isinstance(state.get("executable"), str) or not state["executable"]
                    or not _owner_is_dead(state)):
                return False  # A live/unknown owner may be between watchdog incarnations.
        try:
            proc = psutil.Process(state["pid"])
            born = proc.create_time()
            if modern:
                return born > state["create_time"]
            if born > modified:
                return True
            executable = proc.exe()
            if not executable or executable.endswith(" (deleted)"):
                return False
            exe = Path(executable)
            if not exe.is_absolute():
                return False
            return not exe.resolve().is_relative_to(state_path().parent.resolve())
        except psutil.NoSuchProcess:
            return True
    except (KeyError, TypeError, ValueError, OSError, psutil.Error):
        return False


def stale_record_available() -> bool:
    """Read-only UI hint; reset must repeat this check under the publication lock."""
    from hermes_cli.local_runtime.supervisor import state_path

    try:
        path = state_path()
        if not stat.S_ISREG(path.lstat().st_mode):
            return False
        with path.open(encoding="utf-8") as stream:
            return _stale_record(json.load(stream), os.fstat(stream.fileno()).st_mtime)
    except (OSError, ValueError):
        return False


def reset_stale_record() -> bool:
    """Forget only a proven stale record; never signal a process, enable, or start the runtime."""
    from hermes_cli.local_runtime.supervisor import state_path

    with state_lock():
        path = state_path()
        try:
            if not stat.S_ISREG(path.lstat().st_mode):
                return False
            with path.open(encoding="utf-8") as stream:
                state, modified = json.load(stream), os.fstat(stream.fileno()).st_mtime
        except FileNotFoundError:
            return True
        except (OSError, ValueError):
            return False
        if not _stale_record(state, modified):
            return False
        path.unlink()
        return True


def read_state() -> dict:
    from hermes_cli.local_runtime.supervisor import state_path

    try:
        state = json.loads(state_path().read_text(encoding="utf-8"))
        return state if isinstance(state, dict) else {}
    except (OSError, ValueError):
        return {}


_MODERN_FIELDS = ("create_time", "executable", "owner_pid", "owner_create_time")


def is_modern(state: dict) -> bool:
    return any(key in state for key in _MODERN_FIELDS)


def _valid_pid(value) -> bool:
    return type(value) is int and value > 0


def _valid_birth(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def recorded_process(state: dict):
    """Match a modern process incarnation, never just its reusable PID."""
    try:
        pid, created = state["pid"], state["create_time"]
        owner_pid, owner_created = state["owner_pid"], state["owner_create_time"]
        exe = state["executable"]
        if (not _valid_pid(pid) or not _valid_birth(created)
                or not _valid_pid(owner_pid) or not _valid_birth(owner_created)
                or owner_created > created or not isinstance(exe, str) or not exe):
            return None
        proc = psutil.Process(pid)
        if (not proc.is_running() or proc.create_time() != created
                or Path(proc.exe()) != Path(exe)):
            return None
        # Windows retains the original parent PID; POSIX reparents orphans.
        if os.name == "nt" and proc.ppid() != owner_pid:
            return None
        parent = proc.parent()
        if parent is not None:
            if parent.pid == owner_pid:
                if parent.create_time() != owner_created:
                    return None
            elif os.name == "nt" or not _owner_is_dead(state):
                return None  # POSIX may reparent a router whose recorded owner exited.
        return proc
    except (KeyError, TypeError, ValueError, OverflowError, OSError, psutil.Error):
        return None


def _owner_is_dead(state: dict) -> bool:
    try:
        pid, created = state["owner_pid"], state["owner_create_time"]
        if not _valid_pid(pid) or not _valid_birth(created):
            return False
        try:
            owner = psutil.Process(pid)
            # A newer incarnation proves the recorded owner has exited.
            return owner.create_time() > created or not owner.is_running()
        except psutil.NoSuchProcess:
            return True
    except (KeyError, TypeError, ValueError, OverflowError, OSError, psutil.Error):
        return False


def _legacy_orphan_process(state: dict):
    """Older state lacks birth times: require the exact installed binary and launch arguments."""
    from urllib.parse import urlsplit
    from hermes_cli.local_runtime.bootstrap import models_dir
    from hermes_cli.local_runtime.supervisor import state_path

    # A damaged new record must not fall back to weaker legacy evidence.
    if os.name != "nt" or is_modern(state):
        return None
    try:
        if not _valid_pid(state["pid"]):
            return None
        proc = psutil.Process(state["pid"])
        root = state_path().parent
        exe = Path(proc.exe())
        relative = exe.relative_to(root)
        if len(relative.parts) != 3 or exe.name.lower() != "llama-server.exe":
            return None
        if proc.parent() is not None or proc.ppid() <= 0:
            return None
        if proc.create_time() > state_path().stat().st_mtime:
            return None  # the PID was reused after this record was written
        url = urlsplit(state["base_url"])
        if url.scheme != "http" or url.hostname != "127.0.0.1" or not url.port or not state["api_key"]:
            return None
        argv = proc.cmdline()

        def value(flag):
            if argv.count(flag) != 1:
                return None
            index = argv.index(flag) + 1
            return argv[index] if index < len(argv) else None

        if (value("--host") != "127.0.0.1" or value("--port") != str(url.port)
                or value("--api-key") != state["api_key"]):
            return None
        preset, directory = value("--models-preset"), value("--models-dir")
        if not ((preset and Path(preset) == root / "presets.ini" and directory is None)
                or (directory and Path(directory) == models_dir() and preset is None)):
            return None
        return proc
    except (KeyError, TypeError, ValueError, OSError, psutil.Error):
        return None


def stop_recorded_orphan() -> bool:
    """Explicit user stop only. Refuse uncertain identity or a living recorded owner."""
    from hermes_cli.local_runtime.supervisor import LlamaServerSupervisor, state_path

    try:
        state = json.loads(state_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return True  # already stopped and no record
    except (OSError, ValueError):
        return False
    if not isinstance(state, dict) or not _valid_pid(state.get("pid")):
        return False
    try:
        if not psutil.pid_exists(state["pid"]):
            return True  # keep the dead record; endpoint resolution rejects it
        if is_modern(state):
            proc = recorded_process(state)
            if proc is None or not _owner_is_dead(state):
                return False
        else:
            proc = _legacy_orphan_process(state)
        if proc is None or read_state() != state or not proc.is_running():
            return False
        # Retain the psutil incarnation object: destructive methods guard PID reuse.
        LlamaServerSupervisor._terminate_tree(proc, verified_root=True)
        proc.wait(timeout=5)
        logger.info("stopped orphaned managed llama-server pid=%s", proc.pid)
        return True
    except (OSError, psutil.Error):
        logger.warning("could not verify or stop recorded managed llama-server")
        return False
