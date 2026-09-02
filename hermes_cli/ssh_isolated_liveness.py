"""Child-side liveness for SSH-isolated ``hermes serve`` (#101626).

``serve --isolated`` is ``setsid``/``nohup`` detached so it survives SSH
close (#91668). ``PPID=1`` is therefore normal, not an orphan signal, and
the local Electron ``HERMES_PARENT_PID`` watchdog cannot see a laptop on
the other side of the tunnel.

This module is the child-side contract that *is* compatible with that
detach:

- tunneled loopback is a half-open path, so WS protocol ping stays on;
- after a grace window with no authenticated client, the process may exit;
- two SSH-isolated serves must not both hold the same HERMES_HOME.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

_log = logging.getLogger(__name__)

# 15 minutes: longer than a brief reconnect (#91668), shorter than a night
# of Power Nap accumulation.
DEFAULT_SSH_ISOLATED_IDLE_GRACE_S = 900.0
# Ping interval/timeout for the SSH-isolated loopback path. Timeout sits
# above the documented GIL-stall class (~226s, #53773) so a live tunnel is
# not dropped by a long agent turn; a sleeping laptop still fails to pong.
SSH_ISOLATED_WS_PING_INTERVAL_S = 60.0
SSH_ISOLATED_WS_PING_TIMEOUT_S = 600.0
SSH_ISOLATED_LOCK_NAME = ".ssh-isolated-serve.lock"
SSH_ISOLATED_STATE_NAME = ".ssh-isolated-serve.state.json"
SSH_ISOLATED_HANDOVER_REQUEST_NAME = ".ssh-isolated-serve.handover.json"
SSH_ISOLATED_HOME_LOCKED_SENTINEL = "BACKEND_SSH_ISOLATED_HOME_LOCKED"


def write_ssh_isolated_handover_request(
    hermes_home: Path | str,
    *,
    nonce: str,
    seq: int,
    newcomer_pid: Optional[int] = None,
) -> bool:
    """Atomically record a positively identified handover request from a newcomer."""
    root = Path(hermes_home)
    target = root / SSH_ISOLATED_HANDOVER_REQUEST_NAME
    try:
        root.mkdir(parents=True, exist_ok=True)
        payload = {
            "nonce": str(nonce),
            "seq": int(seq),
            "newcomer_pid": int(newcomer_pid or os.getpid()),
            "requested_at": time.time(),
        }
        tmp = root / f"{SSH_ISOLATED_HANDOVER_REQUEST_NAME}.{os.getpid()}.{threading.get_ident()}.{time.monotonic_ns()}.tmp"
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        for attempt in range(5):
            try:
                os.replace(str(tmp), str(target))
                return True
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(0.02)
        return False
    except Exception as exc:
        _log.warning("Could not write SSH-isolated handover request file: %s", exc)
        try:
            target.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            if "tmp" in locals():
                tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return False


def read_ssh_isolated_handover_request(
    hermes_home: Path | str,
    max_age_s: float = 15.0,
) -> Optional[dict[str, Any]]:
    """Read a pending handover request if present, valid, and recent."""
    root = Path(hermes_home)
    target = root / SSH_ISOLATED_HANDOVER_REQUEST_NAME
    try:
        if not target.is_file():
            return None
        text = target.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            return None
        data = json.loads(text)
        if not isinstance(data, dict):
            return None
        req_at = data.get("requested_at")
        if req_at is None:
            return None
        try:
            if (time.time() - float(req_at)) > max_age_s:
                return None
        except (ValueError, TypeError):
            return None
        if not data.get("nonce") or not isinstance(data.get("newcomer_pid"), int):
            return None
        return data
    except Exception:
        return None


def unlink_ssh_isolated_handover_request(hermes_home: Path | str) -> None:
    """Safely unlink the handover request file."""
    try:
        (Path(hermes_home) / SSH_ISOLATED_HANDOVER_REQUEST_NAME).unlink(missing_ok=True)
    except Exception:
        pass


def _process_create_time(pid: int) -> float:
    """Return creation time of process, using psutil or /proc/{pid} fallback on Linux."""
    try:
        import psutil

        return float(psutil.Process(pid).create_time())
    except Exception:
        pass
    if sys.platform.startswith("linux") and os.path.isdir(f"/proc/{pid}"):
        try:
            return float(os.stat(f"/proc/{pid}").st_mtime)
        except Exception:
            pass
    return time.time()


def write_ssh_isolated_state(
    hermes_home: Path | str,
    *,
    pid: int,
    state: str,
    active_clients: int = 0,
    turn_active: bool = False,
    create_time: Optional[float] = None,
    seq: Optional[int] = None,
) -> bool:
    """Atomically record the process state of this SSH-isolated serve."""
    root = Path(hermes_home)
    target = root / SSH_ISOLATED_STATE_NAME
    try:
        root.mkdir(parents=True, exist_ok=True)
        if create_time is None:
            create_time = _process_create_time(pid)

        # Stale write rejection: if existing file has higher seq for same PID, drop write
        if seq is not None and target.is_file():
            try:
                existing = json.loads(target.read_text(encoding="utf-8", errors="replace"))
                if (
                    isinstance(existing, dict)
                    and existing.get("pid") == int(pid)
                    and int(existing.get("seq", 0)) > int(seq)
                ):
                    return True
            except Exception:
                pass

        payload = {
            "pid": int(pid),
            "create_time": float(create_time),
            "state": str(state),
            "active_clients": max(0, int(active_clients)),
            "turn_active": bool(turn_active),
            "seq": int(seq) if seq is not None else 0,
            "updated_at": time.time(),
        }
        tmp = root / f"{SSH_ISOLATED_STATE_NAME}.{pid}.{threading.get_ident()}.{time.monotonic_ns()}.tmp"
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        for attempt in range(5):
            try:
                os.replace(str(tmp), str(target))
                return True
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(0.02)
        return False
    except Exception as exc:
        _log.warning("Could not write SSH-isolated state file: %s", exc)
        try:
            target.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            if "tmp" in locals():
                tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return False


def read_ssh_isolated_state(hermes_home: Path | str) -> Optional[dict[str, Any]]:
    """Read the recorded state of the SSH-isolated serve owning this home."""
    root = Path(hermes_home)
    target = root / SSH_ISOLATED_STATE_NAME
    for attempt in range(3):
        try:
            if not target.is_file():
                return None
            text = target.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                return None
            data = json.loads(text)
            if isinstance(data, dict):
                return data
            return None
        except (OSError, json.JSONDecodeError):
            if attempt == 2:
                return None
            time.sleep(0.01)
        except Exception:
            return None
    return None


def _pid_is_alive(pid: int) -> bool:
    """Return True if process with given pid exists and is running."""
    if pid <= 0:
        return False
    try:
        import psutil

        return psutil.pid_exists(pid)
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x1000 | 0x00100000, False, pid)
            if not handle:
                return False
            exit_code = ctypes.c_ulong()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                kernel32.CloseHandle(handle)
                return exit_code.value == 259  # STILL_ACTIVE
            kernel32.CloseHandle(handle)
            return False
        except Exception:
            return False
    try:
        os.kill(pid, 0)  # windows-footgun: ok
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _signal_retirement(pid: int) -> bool:
    """Request graceful shutdown of the given process (SIGTERM)."""
    if pid <= 0:
        return False
    try:
        import signal

        os.kill(pid, signal.SIGTERM)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _is_positive_flock_holder(
    pid: int,
    hermes_home: Path | str,
    state_info: dict[str, Any],
) -> bool:
    """Positively prove that PID is the process holding the contested flock.

    Fails closed on any ambiguity, missing proof, PID reuse, or error.
    """
    if pid <= 0 or pid == os.getpid():
        return False
    try:
        proc_create_time = _process_create_time(pid)
        cmdline = ""
        has_lock = False
        lock_path = (Path(hermes_home) / SSH_ISOLATED_LOCK_NAME).resolve()

        try:
            import psutil

            proc = psutil.Process(pid)
            if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
                return False
            cmdline = " ".join(proc.cmdline()).lower()
            open_files = proc.open_files()
            has_lock = any(Path(f.path).resolve() == lock_path for f in open_files)
        except Exception:
            # Fallback for Linux when psutil is unavailable or errors
            if sys.platform.startswith("linux") and os.path.isdir(f"/proc/{pid}"):
                try:
                    cmdline_raw = Path(f"/proc/{pid}/cmdline").read_bytes()
                    cmdline = cmdline_raw.replace(b"\x00", b" ").decode(errors="ignore").lower()
                except Exception:
                    pass

        if not cmdline:
            return False

        # 1. PID reuse protection: compare creation time with recorded create_time
        recorded_create_time = state_info.get("create_time")
        if recorded_create_time is not None and proc_create_time is not None:
            if abs(proc_create_time - float(recorded_create_time)) > 1.0:
                _log.warning(
                    "PID %s create_time mismatch (recorded %s vs actual %s); PID was reused.",
                    pid,
                    recorded_create_time,
                    proc_create_time,
                )
                return False

        # 2. Process identity: must be hermes serve (or pytest during test execution)
        is_hermes_serve = ("hermes" in cmdline and "serve" in cmdline) or (
            "PYTEST_CURRENT_TEST" in os.environ and "pytest" in cmdline
        )
        if not is_hermes_serve:
            _log.warning(
                "PID %s cmdline (%r) is not hermes serve; refusing signal.",
                pid,
                cmdline,
            )
            return False

        # 3. Flock ownership: process must hold open file descriptor to the lock file
        if not has_lock and sys.platform.startswith("linux"):
            try:
                fd_dir = f"/proc/{pid}/fd"
                target_str = str(lock_path)
                for fd_name in os.listdir(fd_dir):
                    try:
                        if os.readlink(f"{fd_dir}/{fd_name}") == target_str:
                            has_lock = True
                            break
                    except OSError:
                        continue
            except Exception:
                pass

        if not has_lock:
            _log.warning(
                "PID %s does not hold open lock file %s; refusing signal.",
                pid,
                lock_path,
            )
            return False

        return True
    except Exception as exc:
        _log.warning("Could not positively prove PID %s holds lock (%s); failing closed.", pid, exc)
        return False



def ssh_isolated_ws_ping_window(
    *,
    is_loopback: bool,
    ssh_session_token: Optional[str],
    default_interval: float,
    default_timeout: float,
) -> tuple[Optional[float], Optional[float]]:
    """Return uvicorn ``ws_ping_interval`` / ``ws_ping_timeout``.

    Plain loopback Desktop has no network hop, so ping stays disabled.
    SSH-isolated loopback is across a real SSH tunnel: enable ping so dead
    tunnels drop cleanly instead of hanging in half-open state.
    """
    if not is_loopback:
        return default_interval, default_timeout
    token = (ssh_session_token or "").strip()
    if not token:
        return None, None
    interval = max(float(default_interval), SSH_ISOLATED_WS_PING_INTERVAL_S)
    timeout = max(float(default_timeout), SSH_ISOLATED_WS_PING_TIMEOUT_S)
    if timeout < interval:
        timeout = interval * 2.0
    return interval, timeout


def ssh_isolated_should_exit(
    *,
    has_ssh_token: bool,
    now: float,
    last_client_at: float,
    grace_s: float,
    ppid: Optional[int] = None,
    turn_in_flight: bool = False,
) -> bool:
    """True when an SSH-isolated backend has been client-idle past grace.

    ``ppid`` is accepted and ignored: isolated remotes legitimately live at
    pid 1 after ``setsid``. An in-flight agent turn holds the process even
    with no client so a lid-close does not kill a running job.
    """
    del ppid
    if turn_in_flight:
        return False
    if not has_ssh_token:
        return False
    try:
        grace = float(grace_s)
    except (TypeError, ValueError):
        return False
    if grace <= 0:
        grace = DEFAULT_SSH_ISOLATED_IDLE_GRACE_S
    try:
        idle_for = float(now) - float(last_client_at)
    except (TypeError, ValueError):
        return False
    return idle_for >= grace


def acquire_ssh_isolated_home_lock(hermes_home) -> Optional[int]:
    """Non-blocking exclusive lock on ``{hermes_home}/.ssh-isolated-serve.lock``.

    Returns a held fd (keep it open for the process lifetime) or ``None``
    if another SSH-isolated serve already owns this home.
    """
    root = Path(hermes_home)
    try:
        root.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(root / SSH_ISOLATED_LOCK_NAME), os.O_CREAT | os.O_RDWR, 0o600)
    except OSError:
        return None
    try:
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd


def ensure_ssh_isolated_home_lock(
    ssh_session_token: Optional[str],
    hermes_home: Optional[Path | str] = None,
) -> Optional[int]:
    """Acquire exclusive home lock for SSH-isolated serve or exit.

    If ssh_session_token is absent or blank, no lock is taken (standalone mode).
    If another SSH-isolated backend holds the lock:
      - If recorded as an idle orphan (no live clients, no turn in flight), requests
        graceful retirement via SIGTERM and waits up to 2.0s for the lock to clear,
        allowing the live newcomer to take over immediately (Cooperative Handover).
      - If active or refusing to yield, prints sentinel and raises SystemExit.
    Returns the held file descriptor, or None if not applicable.
    """
    if not (ssh_session_token or "").strip():
        return None
    if hermes_home is None:
        from hermes_constants import get_hermes_home

        hermes_home = get_hermes_home()

    global _idle_tracker
    if _idle_tracker is not None and _idle_tracker._hermes_home is None:
        _idle_tracker._hermes_home = hermes_home

    fd = acquire_ssh_isolated_home_lock(hermes_home)
    if fd is not None:
        written = write_ssh_isolated_state(
            hermes_home,
            pid=os.getpid(),
            state="active",
            active_clients=0,
            turn_active=False,
        )
        if not written:
            try:
                (Path(hermes_home) / SSH_ISOLATED_STATE_NAME).unlink(missing_ok=True)
            except Exception:
                pass
        return fd

    # Contention: check if the incumbent is an idle orphan that should yield
    state_info = read_ssh_isolated_state(hermes_home)
    if (
        state_info
        and state_info.get("state") == "idle"
        and not state_info.get("turn_active")
    ):
        incumbent_pid = state_info.get("pid")
        if (
            isinstance(incumbent_pid, int)
            and incumbent_pid != os.getpid()
            and _is_positive_flock_holder(incumbent_pid, hermes_home, state_info)
        ):
            # Re-read sidecar immediately before signaling to ensure incumbent
            # hasn't transitioned to active/turn while holder proof was running
            fresh_state = read_ssh_isolated_state(hermes_home)
            if (
                not fresh_state
                or fresh_state.get("state") != "idle"
                or fresh_state.get("turn_active")
                or fresh_state.get("active_clients", 0) > 0
                or fresh_state.get("pid") != incumbent_pid
                or fresh_state.get("seq", 0) != state_info.get("seq", 0)
            ):
                _log.info(
                    "Incumbent PID %s transitioned to active/turn during contention check; refusing signal.",
                    incumbent_pid,
                )
            else:
                _log.info(
                    "Detected idle SSH-isolated orphan backend (PID %s) holding %s; requesting retirement for newcomer.",
                    incumbent_pid,
                    hermes_home,
                )
                import uuid

                nonce = uuid.uuid4().hex
                write_ssh_isolated_handover_request(
                    hermes_home,
                    nonce=nonce,
                    seq=int(fresh_state.get("seq", 0)),
                    newcomer_pid=os.getpid(),
                )
                try:
                    _signal_retirement(incumbent_pid)
                    deadline = time.monotonic() + 5.0
                    while time.monotonic() < deadline:
                        time.sleep(0.05)
                        fd = acquire_ssh_isolated_home_lock(hermes_home)
                        if fd is not None:
                            _log.info(
                                "Cooperative handover succeeded: acquired lock after PID %s yielded.",
                                incumbent_pid,
                            )
                            written = write_ssh_isolated_state(
                                hermes_home,
                                pid=os.getpid(),
                                state="active",
                                active_clients=0,
                                turn_active=False,
                            )
                            if not written:
                                try:
                                    (Path(hermes_home) / SSH_ISOLATED_STATE_NAME).unlink(missing_ok=True)
                                except Exception:
                                    pass
                            unlink_ssh_isolated_handover_request(hermes_home)
                            return fd
                finally:
                    unlink_ssh_isolated_handover_request(hermes_home)

    print(SSH_ISOLATED_HOME_LOCKED_SENTINEL, flush=True)
    raise SystemExit(
        "Another SSH-isolated Hermes backend already holds this "
        "HERMES_HOME database; refusing a second writer."
    )


def _idle_grace_s() -> float:
    return DEFAULT_SSH_ISOLATED_IDLE_GRACE_S


class SshIsolatedIdleTracker:
    """Authenticated-client clock for the SSH-isolated idle watchdog."""

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        hermes_home: Optional[Path | str] = None,
        turn_probe: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._pub_lock = threading.Lock()
        self._live = 0
        self._seq = 0
        self._zero_since = clock()
        self._hermes_home = hermes_home
        self._turn_probe = turn_probe
        self._create_time = _process_create_time(os.getpid())
        self._handover_done = threading.Event()

    def _publish_state(self, force_turn_active: Optional[bool] = None) -> None:
        """Serialized publication of state with monotonic versioning.

        Ensures active state cannot regress to idle through write reordering.
        """
        if not self._hermes_home:
            return
        with self._pub_lock:
            with self._lock:
                live = self._live
                seq = self._seq
            turn_active = False
            if force_turn_active is not None:
                turn_active = bool(force_turn_active)
            elif live == 0:
                probe = self._turn_probe or default_turn_probe
                turn_active = evaluate_turn_probe(probe)
                # Re-verify live under lock after probe evaluation in case a client joined
                with self._lock:
                    live = self._live
                    seq = self._seq
            st = "active" if (live > 0 or turn_active) else "idle"
            write_ssh_isolated_state(
                self._hermes_home,
                pid=os.getpid(),
                state=st,
                active_clients=live,
                turn_active=turn_active,
                create_time=self._create_time,
                seq=seq,
            )

    def on_open(self) -> None:
        with self._lock:
            self._live += 1
            self._seq += 1
        self._publish_state()

    def on_close(self) -> None:
        with self._lock:
            self._live = max(0, self._live - 1)
            self._seq += 1
            if self._live == 0:
                self._zero_since = self._clock()
        self._publish_state()

    def last_client_at(self, now: Optional[float] = None) -> float:
        now = self._clock() if now is None else now
        with self._lock:
            if self._live > 0:
                return now
            return self._zero_since

    def live_count(self) -> int:
        with self._lock:
            return self._live

    def touch(self) -> None:
        """Treat now as client activity (in-flight turn, live socket)."""
        with self._lock:
            self._zero_since = self._clock()

    def now(self) -> float:
        return self._clock()


_idle_tracker: Optional[SshIsolatedIdleTracker] = None


def note_ssh_isolated_client_open() -> None:
    if _idle_tracker is not None:
        _idle_tracker.on_open()


def note_ssh_isolated_client_close() -> None:
    if _idle_tracker is not None:
        _idle_tracker.on_close()


@contextmanager
def track_ssh_isolated_ws() -> Iterator[None]:
    """Count an authenticated WebSocket for the SSH-isolated idle clock."""
    note_ssh_isolated_client_open()
    try:
        yield
    finally:
        note_ssh_isolated_client_close()


def default_turn_probe() -> bool:
    """Default turn probe checking active sessions via tui_gateway."""
    try:
        from tui_gateway.server import _any_session_running

        return bool(_any_session_running())
    except Exception as exc:
        _log.warning(
            "ssh-isolated default turn probe failed; failing closed to preserve process: %s",
            exc,
        )
        return True


def evaluate_turn_probe(probe: Optional[Callable[[], bool]]) -> bool:
    """Evaluate turn probe with fail-closed semantics.

    An exception or indeterminate state must NEVER grant shutdown authority.
    If the probe raises or cannot determine state, treat as in-flight (True)
    so the process is preserved and the grace window refreshed.
    """
    if probe is None:
        return False
    try:
        return bool(probe())
    except Exception as exc:
        _log.warning(
            "ssh-isolated turn probe failed; failing closed to preserve process: %s",
            exc,
        )
        return True


def ssh_isolated_idle_step(
    *,
    has_ssh_token: bool,
    tracker: SshIsolatedIdleTracker,
    grace_s: float,
    turn_in_flight: bool = False,
    turn_probe: Optional[Callable[[], bool]] = None,
) -> bool:
    """One watchdog tick. True → request graceful shutdown.

    An in-flight turn (or an indeterminate probe result) refreshes the idle
    clock so the client gets a full grace window after the job finishes.
    """
    active = turn_in_flight
    if turn_probe is not None:
        active = evaluate_turn_probe(turn_probe)

    if tracker._hermes_home:
        tracker._publish_state(force_turn_active=active)

    if active:
        tracker.touch()
        return False
    return ssh_isolated_should_exit(
        has_ssh_token=has_ssh_token,
        now=tracker.now(),
        last_client_at=tracker.last_client_at(),
        grace_s=grace_s,
        turn_in_flight=False,
    )


def consume_handover_retirement(
    tracker: SshIsolatedIdleTracker,
    *,
    server: Any = None,
    request_shutdown: Optional[Callable[[], None]] = None,
) -> bool:
    """Consume a retirement/handover request with fail-closed in-memory validation.

    The incumbent rechecks its authoritative client count and fail-closed turn probe
    immediately at the shutdown boundary. If active, exit authority is refused and
    the flock is retained.
    """
    with tracker._lock:
        live = tracker._live
    turn_active = False
    if live == 0:
        probe = tracker._turn_probe or default_turn_probe
        turn_active = evaluate_turn_probe(probe)
        with tracker._lock:
            live = tracker._live

    if live > 0 or turn_active:
        _log.warning(
            "Retirement request rejected: incumbent is active (live=%d, turn=%s); retaining flock.",
            live,
            turn_active,
        )
        tracker._publish_state()
        return False

    _log.info(
        "Retirement request accepted: incumbent is idle; initiating graceful shutdown."
    )
    if request_shutdown is not None:
        request_shutdown()
    elif server is not None:
        setattr(server, "should_exit", True)
    return True


def start_ssh_isolated_idle_watchdog(
    *,
    has_ssh_token: bool,
    poll_s: float = 5.0,
    clock: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
    request_shutdown: Optional[Callable[[], None]] = None,
    server: Any = None,
    turn_probe: Optional[Callable[[], bool]] = None,
    tracker: Optional[SshIsolatedIdleTracker] = None,
    hermes_home: Optional[Path | str] = None,
) -> Optional[SshIsolatedIdleTracker]:
    """Daemon thread: ask uvicorn to exit after idle grace. No-op without ssh token.

    ``request_shutdown`` (or setting ``server.should_exit = True``) allows
    uvicorn to flush SQLite WAL and run lifespan hooks. ``os._exit`` is not used here.
    """
    global _idle_tracker
    if not has_ssh_token:
        return None
    if hermes_home is None:
        from hermes_constants import get_hermes_home

        hermes_home = get_hermes_home()
    probe = turn_probe if turn_probe is not None else default_turn_probe
    if tracker is not None:
        owned = tracker
        if owned._hermes_home is None:
            owned._hermes_home = hermes_home
        if owned._turn_probe is None:
            owned._turn_probe = probe
    else:
        owned = SshIsolatedIdleTracker(
            clock=clock, hermes_home=hermes_home, turn_probe=probe
        )
    _idle_tracker = owned
    grace = _idle_grace_s()
    poll = max(0.5, float(poll_s))
    shutdown = request_shutdown
    if shutdown is None and server is not None:
        shutdown = lambda: setattr(server, "should_exit", True)

    handover_event = threading.Event()
    handover_done = owned._handover_done

    def _loop() -> None:
        while True:
            # Check for non-blocking handover wake event
            triggered = handover_event.wait(timeout=poll)
            if triggered:
                handover_event.clear()
                req = read_ssh_isolated_handover_request(hermes_home)
                if req:
                    accepted = consume_handover_retirement(
                        owned, server=server, request_shutdown=shutdown
                    )
                    unlink_ssh_isolated_handover_request(hermes_home)
                    handover_done.set()
                    if accepted:
                        if shutdown is not None:
                            shutdown()
                        return
                handover_done.set()

            if ssh_isolated_idle_step(
                has_ssh_token=True,
                tracker=owned,
                grace_s=grace,
                turn_probe=probe,
            ):
                if shutdown is not None:
                    shutdown()
                return

    try:
        import signal

        if hasattr(signal, "SIGTERM"):
            orig_handler = signal.getsignal(signal.SIGTERM)

            def _sigterm_handover_handler(signum, frame):
                # 1. Non-blocking check: is there a pending handover request?
                # If no valid request exists on disk, this is an ordinary operational
                # SIGTERM (service stop, explicit kill, process supervisor) -> delegate immediately!
                req = read_ssh_isolated_handover_request(hermes_home)
                if not req:
                    if callable(orig_handler) and orig_handler not in (
                        signal.SIG_IGN,
                        signal.SIG_DFL,
                        _sigterm_handover_handler,
                    ):
                        orig_handler(signum, frame)
                    elif shutdown is not None:
                        shutdown()
                    return

                # 2. Positively identified handover request present:
                # Do NOT acquire tracker locks or evaluate turn probe in signal context (prevents deadlock).
                # Only perform a non-blocking wake/latch for the watchdog thread.
                handover_done.clear()
                handover_event.set()

            signal.signal(signal.SIGTERM, _sigterm_handover_handler)
    except Exception as exc:
        _log.debug("Could not install SIGTERM handover handler: %s", exc)

    threading.Thread(target=_loop, daemon=True, name="ssh-isolated-idle").start()
    return owned
