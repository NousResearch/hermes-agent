"""GatewaySupervisor — manages the memory-tencentdb Gateway Node.js sidecar process.

On initialize(), checks if the Gateway is already running. If not, starts
it as a subprocess and waits for /health to become available.

On shutdown(), sends a flush signal and waits for clean exit.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import time
from typing import IO, Optional

from .client import MemoryTencentdbSdkClient

logger = logging.getLogger(__name__)

# Default Gateway address
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8420

# Health check parameters
HEALTH_CHECK_INTERVAL = 0.5  # seconds between checks
HEALTH_CHECK_MAX_WAIT = 30   # max seconds to wait for Gateway to start
HEALTH_CHECK_RETRIES = 3     # retries for is_running check

# Log file rotation parameters
LOG_TAIL_BYTES_ON_CRASH = 2048  # bytes of stderr log to surface on startup crash


class GatewaySupervisor:
    """Manages the memory-tencentdb Gateway sidecar lifecycle."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        gateway_cmd: Optional[str] = None,
    ):
        self._host = host
        self._port = port
        self._base_url = f"http://{host}:{port}"
        self._client = MemoryTencentdbSdkClient(base_url=self._base_url, timeout=5)
        self._process: Optional[subprocess.Popen] = None
        # File handles for child's stdout/stderr. Kept open for the lifetime of
        # the process so the kernel pipe buffer never fills up (otherwise the
        # Gateway's event loop would block on write() after ~64 KB of logs).
        self._stdout_log: Optional[IO[bytes]] = None
        self._stderr_log: Optional[IO[bytes]] = None
        self._stderr_log_path: Optional[str] = None

        # Resolve Gateway command
        # Priority: explicit arg > MEMORY_TENCENTDB_GATEWAY_CMD env
        self._gateway_cmd = gateway_cmd or os.environ.get("MEMORY_TENCENTDB_GATEWAY_CMD", "")

    def is_running(self) -> bool:
        """Check if the Gateway is currently responding to health checks."""
        for _ in range(HEALTH_CHECK_RETRIES):
            try:
                result = self._client.health(timeout=2)
                return result.get("status") in ("ok", "degraded")
            except Exception:
                time.sleep(0.2)
        return False

    def is_process_alive(self) -> bool:
        """Return True iff we have spawned a child and it has not exited.

        Distinct from ``is_running()``:
          * ``is_running`` performs a network health check — slow, but works
            even when the Gateway was started externally (systemd, manual run).
          * ``is_process_alive`` only inspects our own ``Popen`` handle — fast,
            and lets the watchdog notice an exited child without paying for an
            HTTP round-trip every tick.

        Returns False when we never spawned a child, or when the child has
        exited (``poll()`` returns a non-None code). The watchdog combines
        both checks: ``is_process_alive() or is_running()`` — only when both
        say "no" do we attempt a re-spawn.
        """
        proc = self._process
        if proc is None:
            return False
        return proc.poll() is None

    def _reap_dead_process(self) -> None:
        """Drop the reference to a child we spawned that has since exited.

        Called from ``ensure_running`` so that a re-spawn after a crash does
        not leak the previous ``Popen`` handle (the kernel still owns the
        zombie until ``wait()``-style call). Safe to call when the process
        is still alive — it's a no-op in that case.
        """
        proc = self._process
        if proc is None:
            return
        if proc.poll() is None:
            return  # still alive
        try:
            # poll() already reaped the child via waitpid internally on POSIX,
            # so there is nothing more to do here. Just drop our handle and
            # close the log files we opened for this run.
            rc = proc.returncode
            logger.warning(
                "memory-tencentdb Gateway: previous child exited (code=%s); "
                "reaping before respawn.", rc,
            )
        finally:
            self._process = None
            self._close_log_handles()

    def ensure_running(self) -> bool:
        """Ensure the Gateway is running. Start it if not.

        Returns True if the Gateway is available, False if startup failed.
        """
        if self.is_running():
            logger.info("memory-tencentdb Gateway already running at %s", self._base_url)
            return True

        # If we previously spawned a child and it has since died, drop the
        # stale Popen handle so the new spawn below isn't shadowed by a
        # zombie reference. Without this, a crashed-then-respawned Gateway
        # would keep ``self._process`` pointing at the dead PID forever and
        # ``is_process_alive()`` would mislead the watchdog.
        self._reap_dead_process()

        # Try to start the Gateway
        if not self._gateway_cmd:
            logger.warning(
                "memory-tencentdb Gateway is not running and no gateway command configured. "
                "Set MEMORY_TENCENTDB_GATEWAY_CMD environment variable or pass gateway_cmd to supervisor. "
                "memory-tencentdb memory will be unavailable."
            )
            return False

        logger.info("Starting memory-tencentdb Gateway: %s", self._gateway_cmd)

        try:
            env = os.environ.copy()
            env["MEMORY_TENCENTDB_GATEWAY_PORT"] = str(self._port)
            env["MEMORY_TENCENTDB_GATEWAY_HOST"] = self._host

            # Redirect child stdout/stderr to log files instead of PIPE.
            # Using PIPE without an active reader will deadlock the child once
            # the pipe buffer (~64 KB) fills up. A log directory next to the
            # data dir keeps logs inspectable on crash while eliminating the
            # blocking risk entirely.
            log_dir = self._resolve_log_dir()
            try:
                os.makedirs(log_dir, exist_ok=True)
            except OSError as e:
                logger.warning(
                    "memory-tencentdb Gateway: failed to create log dir %s (%s); "
                    "falling back to DEVNULL", log_dir, e,
                )
                log_dir = None

            if log_dir is not None:
                stdout_path = os.path.join(log_dir, "gateway.stdout.log")
                stderr_path = os.path.join(log_dir, "gateway.stderr.log")
                # Append mode: preserve previous runs for postmortem.
                self._stdout_log = open(stdout_path, "ab", buffering=0)
                self._stderr_log = open(stderr_path, "ab", buffering=0)
                self._stderr_log_path = stderr_path
                stdout_target: object = self._stdout_log
                stderr_target: object = self._stderr_log
            else:
                stdout_target = subprocess.DEVNULL
                stderr_target = subprocess.DEVNULL

            self._process = subprocess.Popen(
                shlex.split(self._gateway_cmd),
                env=env,
                stdout=stdout_target,
                stderr=stderr_target,
                start_new_session=True,  # Detach from parent process group
            )
        except Exception as e:
            logger.error("Failed to start memory-tencentdb Gateway: %s", e)
            self._close_log_handles()
            return False

        # Wait for health check
        return self._wait_for_health()

    def _resolve_log_dir(self) -> str:
        """Pick a directory to store Gateway stdout/stderr logs.

        Priority:
          1. ``MEMORY_TENCENTDB_LOG_DIR`` env var
          2. ``~/.hermes/logs/memory_tencentdb`` (hermes-style log location)
          3. ``<cwd>/.memory-tencentdb-logs`` (last-resort fallback if $HOME
             is not set — unusual on real systems, but e.g. hermetic tests)

        Note: the supervisor intentionally does *not* derive this from the
        Gateway's data dir — the Gateway owns that path and the supervisor
        no longer tracks it. Keeping our log dir in the hermes log tree also
        avoids interleaving Gateway logs with user-facing memory data.
        """
        env_dir = os.environ.get("MEMORY_TENCENTDB_LOG_DIR")
        if env_dir:
            return env_dir
        home = os.environ.get("HOME") or os.environ.get("USERPROFILE")
        if home:
            return os.path.join(home, ".hermes", "logs", "memory_tencentdb")
        return os.path.join(os.getcwd(), ".memory-tencentdb-logs")

    def _close_log_handles(self) -> None:
        """Close log file handles; safe to call multiple times."""
        for attr in ("_stdout_log", "_stderr_log"):
            handle: Optional[IO[bytes]] = getattr(self, attr, None)
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
                setattr(self, attr, None)

    def _tail_stderr_log(self, max_bytes: int = LOG_TAIL_BYTES_ON_CRASH) -> str:
        """Return the last `max_bytes` of the stderr log for crash diagnostics."""
        path = self._stderr_log_path
        if not path:
            return ""
        try:
            size = os.path.getsize(path)
            with open(path, "rb") as f:
                if size > max_bytes:
                    f.seek(-max_bytes, os.SEEK_END)
                return f.read().decode("utf-8", errors="replace")
        except Exception:
            return ""

    def _wait_for_health(self) -> bool:
        """Wait for the Gateway to become healthy."""
        start = time.monotonic()
        while time.monotonic() - start < HEALTH_CHECK_MAX_WAIT:
            # Check if process died
            if self._process and self._process.poll() is not None:
                rc = self._process.returncode
                # stderr was redirected to a log file; tail it for diagnostics.
                stderr = self._tail_stderr_log()[:500]
                logger.error(
                    "memory-tencentdb Gateway process exited with code %d during startup. "
                    "stderr_log=%s tail=%s",
                    rc, self._stderr_log_path or "<none>", stderr,
                )
                self._close_log_handles()
                return False

            try:
                result = self._client.health(timeout=2)
                if result.get("status") in ("ok", "degraded"):
                    logger.info(
                        "memory-tencentdb Gateway is ready (took %.1fs)",
                        time.monotonic() - start,
                    )
                    return True
            except Exception:
                pass

            time.sleep(HEALTH_CHECK_INTERVAL)

        logger.error(
            "memory-tencentdb Gateway did not become healthy within %ds",
            HEALTH_CHECK_MAX_WAIT,
        )
        return False

    def shutdown(self) -> None:
        """Shut down the managed Gateway process (if we started it)."""
        if self._process is None:
            return

        logger.info("Shutting down memory-tencentdb Gateway...")

        try:
            # Send SIGTERM for graceful shutdown
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("memory-tencentdb Gateway did not exit in 10s, sending SIGKILL")
                self._process.kill()
                self._process.wait(timeout=5)
        except Exception as e:
            logger.warning("Error shutting down memory-tencentdb Gateway: %s", e)
        finally:
            self._process = None
            self._close_log_handles()

    @property
    def client(self) -> MemoryTencentdbSdkClient:
        """Get the HTTP client for making API calls."""
        return self._client
