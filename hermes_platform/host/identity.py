"""OS-account and process identities for cooperative cross-profile admission."""

from __future__ import annotations

import hashlib
import os
import socket
import sys
from pathlib import Path

import psutil


def account_home() -> Path:
    """Resolve the real account, never HOME, XDG_STATE_HOME or a Hermes profile."""
    if os.name != "nt":
        import pwd
        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    import ctypes
    buffer = ctypes.create_unicode_buffer(32768)
    # CSIDL_PROFILE resolves the current process token, not USERPROFILE.
    result = ctypes.windll.shell32.SHGetFolderPathW(None, 40, None, 0, buffer)
    if result:
        raise OSError("Cannot resolve the OS account profile directory")
    return Path(buffer.value)


def execution_namespace() -> str:
    """Bind PID interpretation to this host, boot and (on Linux) PID namespace."""
    identity = [socket.gethostname(), str(psutil.boot_time())]
    if sys.platform.startswith("linux"):
        proc_ns = Path("/proc/self/ns/pid")
        identity.append(str(proc_ns.stat().st_ino))
    return hashlib.sha256("\0".join(identity).encode()).hexdigest()


def process_birth(pid: int) -> float:
    return psutil.Process(pid).create_time()


def execution_owner() -> dict:
    return {"pid": os.getpid(), "birth": process_birth(os.getpid()), "namespace": execution_namespace()}


def owner_proven_gone(owner: dict) -> bool:
    """Uncertain/foreign identities retain capacity; TTLs are not death evidence."""
    # Out-of-range pid_t values can be reported as NoSuchProcess by psutil.
    if not 0 < owner["pid"] <= 0x7FFFFFFF:
        return False
    try:
        if owner.get("namespace") != execution_namespace():
            return False
    except (psutil.Error, OSError, OverflowError):
        return False
    try:
        return process_birth(owner["pid"]) != owner["birth"]
    except psutil.NoSuchProcess:
        return True
    except (psutil.AccessDenied, OSError, OverflowError):
        return False
