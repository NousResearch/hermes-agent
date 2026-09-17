"""Durable predecessor-resource barrier for gateway successors."""
from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from typing import Iterable

from hermes_constants import get_hermes_home

BARRIER_NAME = "gateway-restart-barrier.json"


def barrier_path(home: Path | None = None) -> Path:
    return Path(home or get_hermes_home()).expanduser().resolve() / "state" / BARRIER_NAME


def configured_restart_ports(home: Path | None = None) -> list[int]:
    """Return the configured shared webhook listener port without runtime imports."""
    root = Path(home or get_hermes_home()).expanduser().resolve()
    try:
        import yaml
        raw = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8")) or {}
    except (OSError, ValueError, TypeError):
        return []
    platforms = raw.get("platforms") if isinstance(raw, dict) else {}
    webhook = platforms.get("webhook") if isinstance(platforms, dict) else {}
    if not isinstance(webhook, dict) or webhook.get("enabled") is False:
        return []
    extra = webhook.get("extra") if isinstance(webhook.get("extra"), dict) else {}
    try:
        port = int(extra.get("port", 8644))
    except (TypeError, ValueError):
        return []
    return [port] if 0 < port < 65536 else []


def write_restart_barrier(
    predecessor_pid: int, *, home: Path | None = None, ports: Iterable[int] | None = None
) -> Path | None:
    if int(predecessor_pid) <= 0:
        return None
    path = barrier_path(home)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = {
        "predecessor_pid": int(predecessor_pid),
        "ports": sorted({int(p) for p in (ports if ports is not None else configured_restart_ports(home)) if int(p) > 0}),
        "created_at": time.time(),
    }
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return path


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _scoped_locks_released(predecessor_pid: int) -> bool:
    try:
        from gateway.status import _get_lock_dir
        lock_dir = _get_lock_dir()
    except Exception:
        return False
    if not lock_dir.exists():
        return True
    for path in lock_dir.glob("*.lock"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if isinstance(record, dict) and int(record.get("pid") or 0) == predecessor_pid:
            return False
    return True


def _cleanup_dead_predecessor_locks(predecessor_pid: int) -> None:
    """Remove only locks proven to belong to an exited predecessor."""
    try:
        from gateway.status import release_all_scoped_locks
        release_all_scoped_locks(owner_pid=predecessor_pid)
    except Exception:
        return


def _ports_released(ports: Iterable[int]) -> bool:
    for port in ports:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", int(port)))
        except OSError:
            return False
        finally:
            sock.close()
    return True


def wait_for_restart_barrier(
    *, home: Path | None = None, timeout: float = 120.0, poll: float = 0.1
) -> tuple[bool, dict]:
    """Wait until predecessor PID, scoped locks, and configured ports are all released."""
    path = barrier_path(home)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        pid = int(payload.get("predecessor_pid") or 0)
        ports = [int(p) for p in payload.get("ports") or []]
    except (OSError, ValueError, TypeError, AttributeError):
        return True, {"reason": "no-valid-barrier"}
    deadline = time.monotonic() + max(0.0, float(timeout))
    while True:
        predecessor_exited = not _pid_alive(pid)
        if predecessor_exited:
            _cleanup_dead_predecessor_locks(pid)
        checks = {
            "predecessor_exited": predecessor_exited,
            "scoped_locks_released": _scoped_locks_released(pid),
            "ports_released": _ports_released(ports),
        }
        if all(checks.values()):
            try:
                current = json.loads(path.read_text(encoding="utf-8"))
                if int(current.get("predecessor_pid") or 0) == pid:
                    path.unlink(missing_ok=True)
            except (OSError, ValueError, TypeError, AttributeError):
                pass
            return True, {"predecessor_pid": pid, "ports": ports, **checks}
        if time.monotonic() >= deadline:
            return False, {"predecessor_pid": pid, "ports": ports, **checks}
        time.sleep(max(0.01, float(poll)))
