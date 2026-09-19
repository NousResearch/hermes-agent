"""Foreground lifecycle for the isolated observe-only control plane."""

from __future__ import annotations

import fcntl
import os
import stat
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from plugins.agentops.control.api import ControlAPI, request_health
from plugins.agentops.control.config import AgentOpsConfig, StateDirectoryError, initialize_state_dir, load_agentops_config
from plugins.agentops.control.events import EventSpool
from plugins.agentops.control.models import AuthorityMode, ControlPlaneHealth
from plugins.agentops.control.store import AgentOpsStore, StoreIntegrityError, StoreMigrationError, inspect_store, open_store


class ProcessLockError(RuntimeError):
    """Another agentopsd process owns the dedicated state directory lock."""


class _ProcessLock:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._descriptor: int | None = None

    def acquire(self) -> None:
        if self.path.is_symlink() or self.path.parent.is_symlink() or self.path.exists() and not self.path.is_file():
            raise ProcessLockError("lock path rejected")
        no_follow = getattr(os, "O_NOFOLLOW", None)
        if no_follow is None:
            raise ProcessLockError("safe lock open unsupported")
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | no_follow, 0o600)
        try:
            status = os.fstat(descriptor)
            if status.st_uid != os.getuid() or not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
                raise ProcessLockError("lock ownership rejected")
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.ftruncate(descriptor, 0)
            os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
            os.fsync(descriptor)
            self._descriptor = descriptor
        except Exception:
            os.close(descriptor)
            raise

    def release(self) -> None:
        if self._descriptor is None:
            return
        try:
            fcntl.flock(self._descriptor, fcntl.LOCK_UN)
        finally:
            os.close(self._descriptor)
            self._descriptor = None


@dataclass
class DaemonHandle:
    socket_path: Path
    stop_event: threading.Event
    thread: threading.Thread
    exit_code: list[int | None]

    def health(self) -> dict:
        return request_health(self.socket_path)

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError("daemon did not stop")


class ObserveOnlyDaemon:
    def __init__(self, config: AgentOpsConfig):
        self.config = config
        self.store: AgentOpsStore | None = None
        self.spool = EventSpool(config.spool_dir, max_bytes=config.event_spool_max_bytes)
        self.reasons = list(config.safe_start_reasons)
        self.api: ControlAPI | None = None
        self._preflight_audit_valid: bool | None = None

    def _reason(self, reason: str) -> None:
        if reason not in self.reasons:
            self.reasons.append(reason)

    def _health(self) -> dict:
        store_available = self.store is not None
        audit_chain_valid: bool | None = self._preflight_audit_valid
        event_count = 0
        try:
            spool_depth = self.spool.depth()
            spool_bytes = self.spool._size_bytes()
            quarantine_bytes = self.spool.quarantine_size_bytes()
            spool_healthy = self.spool.healthy()
        except OSError:
            spool_depth = spool_bytes = quarantine_bytes = 0
            spool_healthy = False
            self._reason("spool_unavailable")
        if not spool_healthy:
            self._reason("spool_unhealthy")
        if self.store is not None:
            try:
                audit_chain_valid = self.store.verify_audit_chain()
                event_count = self.store.event_count()
                if not audit_chain_valid:
                    self._reason("audit_chain_invalid")
            except Exception:
                audit_chain_valid = False
                self._reason("store_unavailable")
        fatal = bool(self.reasons) or not store_available or audit_chain_valid is not True or not spool_healthy
        health = ControlPlaneHealth(
            ready=not fatal,
            authority_mode=AuthorityMode.OBSERVE_ONLY,
            safe_start_reasons=tuple(self.reasons),
            store_available=store_available,
            audit_chain_valid=audit_chain_valid,
            event_count=event_count,
            spool_depth=spool_depth,
            spool_bytes=spool_bytes,
            spool_quarantine_bytes=quarantine_bytes,
            spool_healthy=spool_healthy,
            global_write_enabled=False,
        )
        return health.to_dict()

    def run(self, stop_event: threading.Event) -> int:
        try:
            initialize_state_dir(self.config)
        except StateDirectoryError:
            return 1
        lock = _ProcessLock(self.config.lock_path)
        try:
            lock.acquire()
        except (OSError, ProcessLockError):
            return 1
        try:
            try:
                self.store = open_store(self.config)
                try:
                    replay = self.spool.replay(self.store)
                    if replay.dropped:
                        self._reason("spool_quarantine_budget_exceeded")
                    if replay.failed:
                        self._reason("spool_quarantine_failed")
                except Exception:
                    self._reason("spool_replay_failed")
                if not self.store.verify_audit_chain():
                    self._reason("audit_chain_invalid")
            except (StoreMigrationError, StoreIntegrityError):
                self.store = None
                self._preflight_audit_valid = inspect_store(self.config.sqlite_path).audit_chain_valid
                if self._preflight_audit_valid is False:
                    self._reason("audit_chain_invalid")
                self._reason("store_migration_failed")
            self.api = ControlAPI(
                self.config.socket_path,
                self.config.state_dir,
                self._health,
                allow_stale_reclaim=True,
            )
            try:
                self.api.start()
            except Exception:
                if self.store is not None:
                    self.store.close()
                return 1
            try:
                while not stop_event.wait(0.05):
                    pass
            finally:
                self.api.stop()
                if self.store is not None:
                    self.store.close()
            return 0
        finally:
            lock.release()


def run_daemon(config: AgentOpsConfig, stop_event: threading.Event) -> int:
    """Run a local daemon; this function has no target or Gateway effects."""
    return ObserveOnlyDaemon(config).run(stop_event)


def start_daemon_thread(config_path: Path, *, timeout_seconds: float = 5) -> DaemonHandle:
    """Test helper that starts a daemon in a managed foreground thread."""
    config = load_agentops_config(Path(config_path))
    stop_event = threading.Event()
    result: list[int | None] = [None]

    def _run() -> None:
        result[0] = run_daemon(config, stop_event)

    thread = threading.Thread(target=_run, name="agentops-test-daemon", daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if config.socket_path.exists():
            try:
                request_health(config.socket_path)
                return DaemonHandle(config.socket_path, stop_event, thread, result)
            except (OSError, RuntimeError, ValueError):
                pass
        if not thread.is_alive():
            break
        time.sleep(0.01)
    stop_event.set()
    thread.join(timeout=1)
    raise RuntimeError(f"agentops daemon did not create health socket (exit={result[0]})")
