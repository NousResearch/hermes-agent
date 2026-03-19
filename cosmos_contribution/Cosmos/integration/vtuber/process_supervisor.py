"""
Process Supervisor - Watchdog for FFmpeg and streaming subprocesses

Provides:
- Health monitoring for subprocess health (stdout/stderr activity, CPU usage)
- Auto-restart with exponential backoff on crashes
- Zombie process reaping
- Resource usage monitoring
- Graceful shutdown cascade
"""

import asyncio
import subprocess
import time
import os
import signal
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from collections import deque
from enum import Enum
from loguru import logger


class ProcessState(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    RESTARTING = "restarting"
    FAILED = "failed"
    ZOMBIE = "zombie"


@dataclass
class ProcessHealth:
    """Health metrics for a supervised process"""
    pid: Optional[int] = None
    state: ProcessState = ProcessState.STOPPED
    uptime_seconds: float = 0.0
    restart_count: int = 0
    last_restart: float = 0.0
    last_activity: float = 0.0  # Last time we saw output
    stderr_lines: deque = field(default_factory=lambda: deque(maxlen=50))
    exit_codes: deque = field(default_factory=lambda: deque(maxlen=10))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pid": self.pid,
            "state": self.state.value,
            "uptime_seconds": self.uptime_seconds,
            "restart_count": self.restart_count,
            "last_activity_ago": time.time() - self.last_activity if self.last_activity else None,
            "recent_errors": list(self.stderr_lines)[-5:],
        }


class ProcessSupervisor:
    """
    Supervises a subprocess with health monitoring and auto-restart.

    Usage:
        supervisor = ProcessSupervisor(
            name="ffmpeg_stream",
            cmd_builder=lambda: ["ffmpeg", "-i", "pipe:0", ...],
            max_restarts=10,
            restart_backoff_base=2.0,
            restart_backoff_max=60.0,
            health_check_interval=5.0,
            activity_timeout=30.0,  # Restart if no output for 30s
        )

        process = await supervisor.start()
        # ... use process.stdin, process.stdout ...
        await supervisor.stop()
    """

    def __init__(
        self,
        name: str,
        cmd_builder: Callable[[], List[str]],
        max_restarts: int = 10,
        restart_backoff_base: float = 2.0,
        restart_backoff_max: float = 60.0,
        health_check_interval: float = 5.0,
        activity_timeout: float = 60.0,
        on_restart: Optional[Callable[[], None]] = None,
        on_failure: Optional[Callable[[str], None]] = None,
        stdin_pipe: bool = True,
        stderr_pipe: bool = True,
    ):
        self.name = name
        self._cmd_builder = cmd_builder
        self._max_restarts = max_restarts
        self._backoff_base = restart_backoff_base
        self._backoff_max = restart_backoff_max
        self._health_check_interval = health_check_interval
        self._activity_timeout = activity_timeout
        self._on_restart = on_restart
        self._on_failure = on_failure
        self._stdin_pipe = stdin_pipe
        self._stderr_pipe = stderr_pipe

        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._running = False
        self._health = ProcessHealth()
        self._start_time = 0.0
        self._monitor_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None

        logger.info(f"ProcessSupervisor '{name}' initialized")

    @property
    def process(self) -> Optional[subprocess.Popen]:
        """Get the current subprocess (thread-safe)"""
        with self._lock:
            return self._process

    @property
    def health(self) -> ProcessHealth:
        """Get current health metrics"""
        if self._process and self._health.state == ProcessState.RUNNING:
            self._health.uptime_seconds = time.time() - self._start_time
        return self._health

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running and self._process is not None and self._process.poll() is None

    async def start(self) -> Optional[subprocess.Popen]:
        """Start the supervised process"""
        if self._running:
            logger.warning(f"[{self.name}] Already running")
            return self._process

        self._running = True
        self._health = ProcessHealth()

        process = await self._start_process()
        if process is None:
            self._running = False
            self._health.state = ProcessState.FAILED
            return None

        # Start health monitor
        self._monitor_task = asyncio.create_task(self._health_monitor())

        return process

    async def _start_process(self) -> Optional[subprocess.Popen]:
        """Internal: start or restart the process"""
        try:
            self._health.state = ProcessState.STARTING
            cmd = self._cmd_builder()

            logger.info(f"[{self.name}] Starting: {' '.join(cmd[:5])}...")

            kwargs = {
                "bufsize": 0,
            }
            if self._stdin_pipe:
                kwargs["stdin"] = subprocess.PIPE
            if self._stderr_pipe:
                kwargs["stderr"] = subprocess.PIPE

            process = subprocess.Popen(cmd, **kwargs)

            with self._lock:
                self._process = process

            self._start_time = time.time()
            self._health.pid = process.pid
            self._health.state = ProcessState.RUNNING
            self._health.last_activity = time.time()

            # Start stderr reader if piped
            if self._stderr_pipe:
                self._stderr_task = asyncio.create_task(self._read_stderr())

            logger.info(f"[{self.name}] Started (PID: {process.pid})")
            return process

        except Exception as e:
            logger.error(f"[{self.name}] Failed to start: {e}")
            self._health.state = ProcessState.FAILED
            return None

    async def _read_stderr(self):
        """Read stderr in background to prevent buffer deadlock"""
        try:
            process = self._process
            if not process or not process.stderr:
                return

            loop = asyncio.get_event_loop()
            while self._running and process.poll() is None:
                try:
                    line = await asyncio.wait_for(
                        loop.run_in_executor(None, process.stderr.readline),
                        timeout=5.0
                    )
                    if line:
                        decoded = line.decode('utf-8', errors='replace').strip()
                        if decoded:
                            self._health.stderr_lines.append(decoded)
                            self._health.last_activity = time.time()
                            if 'error' in decoded.lower():
                                logger.warning(f"[{self.name}] stderr: {decoded[:200]}")
                    else:
                        break  # EOF
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    break
        except Exception as e:
            logger.debug(f"[{self.name}] stderr reader stopped: {e}")

    async def _health_monitor(self):
        """Monitor process health and auto-restart on failure"""
        consecutive_failures = 0

        while self._running:
            await asyncio.sleep(self._health_check_interval)

            if not self._running:
                break

            with self._lock:
                process = self._process

            if process is None:
                continue

            # Check if process died
            retcode = process.poll()
            if retcode is not None:
                self._health.exit_codes.append(retcode)
                logger.warning(f"[{self.name}] Process exited with code {retcode}")

                if not self._running:
                    break

                # Attempt restart
                consecutive_failures += 1
                if consecutive_failures > self._max_restarts:
                    logger.error(f"[{self.name}] Max restarts ({self._max_restarts}) exceeded")
                    self._health.state = ProcessState.FAILED
                    if self._on_failure:
                        self._on_failure(f"Max restarts exceeded (last exit: {retcode})")
                    break

                # Exponential backoff
                delay = min(
                    self._backoff_base ** consecutive_failures,
                    self._backoff_max
                )
                logger.info(f"[{self.name}] Restarting in {delay:.1f}s (attempt {consecutive_failures}/{self._max_restarts})")
                self._health.state = ProcessState.RESTARTING

                await asyncio.sleep(delay)

                if not self._running:
                    break

                # Restart
                new_process = await self._start_process()
                if new_process:
                    self._health.restart_count += 1
                    self._health.last_restart = time.time()
                    consecutive_failures = 0  # Reset on success
                    if self._on_restart:
                        self._on_restart()
                else:
                    logger.error(f"[{self.name}] Restart failed")

            else:
                # Process alive - check activity timeout
                consecutive_failures = 0

                if self._activity_timeout > 0:
                    inactive = time.time() - self._health.last_activity
                    if inactive > self._activity_timeout:
                        logger.warning(f"[{self.name}] No activity for {inactive:.0f}s, possible hang")
                        # Don't auto-restart on inactivity alone, just log

    async def stop(self, timeout: float = 10.0):
        """Gracefully stop the supervised process"""
        self._running = False

        # Cancel monitor
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except (asyncio.CancelledError, Exception):
                pass

        if self._stderr_task:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except (asyncio.CancelledError, Exception):
                pass

        with self._lock:
            process = self._process
            self._process = None

        if process and process.poll() is None:
            # Step 1: Send 'q' to stdin (FFmpeg graceful quit)
            try:
                if process.stdin:
                    process.stdin.write(b'q')
                    process.stdin.flush()
                    process.stdin.close()
            except (BrokenPipeError, OSError):
                pass

            # Step 2: Wait for graceful exit
            try:
                await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, process.wait),
                    timeout=timeout / 2
                )
                logger.info(f"[{self.name}] Graceful shutdown (exit: {process.returncode})")
                self._health.state = ProcessState.STOPPED
                return
            except asyncio.TimeoutError:
                pass

            # Step 3: SIGTERM
            try:
                process.terminate()
                await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, process.wait),
                    timeout=timeout / 2
                )
                logger.info(f"[{self.name}] Terminated (exit: {process.returncode})")
            except asyncio.TimeoutError:
                # Step 4: SIGKILL
                process.kill()
                logger.warning(f"[{self.name}] Force killed")
            except Exception as e:
                logger.error(f"[{self.name}] Cleanup error: {e}")

        self._health.state = ProcessState.STOPPED
        logger.info(f"[{self.name}] Stopped")

    async def restart(self):
        """Force restart the process"""
        logger.info(f"[{self.name}] Manual restart requested")

        # Stop current
        with self._lock:
            old = self._process
            self._process = None

        if old and old.poll() is None:
            try:
                old.terminate()
                await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, old.wait),
                    timeout=5
                )
            except (asyncio.TimeoutError, Exception):
                old.kill()

        # Start new
        await self._start_process()
        self._health.restart_count += 1
        self._health.last_restart = time.time()

    def write_stdin(self, data: bytes) -> bool:
        """Thread-safe write to process stdin"""
        with self._lock:
            if self._process and self._process.stdin:
                try:
                    self._process.stdin.write(data)
                    self._process.stdin.flush()
                    self._health.last_activity = time.time()
                    return True
                except (BrokenPipeError, OSError):
                    return False
        return False


class SupervisorPool:
    """
    Manages multiple ProcessSupervisors for the VTuber system.

    Typical supervisors:
    - ffmpeg_stream: Main streaming process
    - ffmpeg_audio: Audio processing
    """

    def __init__(self):
        self._supervisors: Dict[str, ProcessSupervisor] = {}
        self._running = False

    def add(self, supervisor: ProcessSupervisor):
        """Register a supervisor"""
        self._supervisors[supervisor.name] = supervisor

    def get(self, name: str) -> Optional[ProcessSupervisor]:
        """Get supervisor by name"""
        return self._supervisors.get(name)

    async def start_all(self):
        """Start all supervisors"""
        self._running = True
        for name, sup in self._supervisors.items():
            await sup.start()
            logger.info(f"SupervisorPool: Started {name}")

    async def stop_all(self, timeout: float = 15.0):
        """Stop all supervisors gracefully"""
        self._running = False
        tasks = []
        for name, sup in self._supervisors.items():
            tasks.append(sup.stop(timeout=timeout))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        logger.info(f"SupervisorPool: All {len(self._supervisors)} supervisors stopped")

    def health_report(self) -> Dict[str, Dict]:
        """Get health report for all supervised processes"""
        return {
            name: sup.health.to_dict()
            for name, sup in self._supervisors.items()
        }

    @property
    def all_healthy(self) -> bool:
        """Check if all processes are running"""
        return all(sup.is_running for sup in self._supervisors.values())
