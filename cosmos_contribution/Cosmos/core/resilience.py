"""
cosmos Resilience Layer.

"I have made my device strong enough to withstand even my own incompetence!"

This module implements resilience patterns to ensure the Nexus and Agent Swarm 
remain stable under high cognitive load or external failure.

Features:
1. Circuit Breaker: Prevents cascading failures when a subsystem (e.g. LLM API) is struggling.
2. Cognitive Backpressure: Detects when the system is thinking too fast for the I/O to handle.
3. Entropy Monitor: Detects when agent reasoning is degrading (hallucination loops).
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Optional
from loguru import logger
from functools import wraps

class SystemState(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    PANIC = "panic"

class CircuitBreaker:
    """
    Protects expensive operations (like LLM calls or Projection) from failing repeatedly.
    """
    def __init__(self, name: str, failure_threshold: int = 3, reset_timeout: int = 60):
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        
        self.failures = 0
        self.last_failure_time = 0
        self.is_open = False
        
    def __call__(self, func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if self.is_open:
                if time.time() - self.last_failure_time > self.reset_timeout:
                    logger.info(f"CircuitBreaker '{self.name}': Half-open (Testing)")
                    self.is_open = False # Half-open trial
                else:
                    logger.warning(f"CircuitBreaker '{self.name}': Call blocked")
                    return None # Or raise specific exception

            try:
                result = await func(*args, **kwargs)
                if self.failures > 0:
                    logger.info(f"CircuitBreaker '{self.name}': Recovered")
                    self.failures = 0
                return result
            except Exception as e:
                self.failures += 1
                self.last_failure_time = time.time()
                logger.error(f"CircuitBreaker '{self.name}': Failure {self.failures}/{self.failure_threshold} - {e}")
                
                if self.failures >= self.failure_threshold:
                    self.is_open = True
                    logger.critical(f"CircuitBreaker '{self.name}': OPENED. Blocking calls for {self.reset_timeout}s")
                raise e
        return wrapper

class EntropyMonitor:
    """
    Monitors the semantic entropy of the agent's output.
    If the agent starts repeating itself or outputting garbage, this triggers a STOP signal.
    """
    def __init__(self):
        self.history = []
        
    def check_hallucination(self, content: str) -> bool:
        # Simple heuristic: Check for exact repetitions
        # Real implementation would use semantic similarity or compression ratio
        if content in self.history[-5:]:
            return True
        self.history.append(content)
        if len(self.history) > 20:
            self.history.pop(0)
        return False

# Global resilience monitors
projector_breaker = CircuitBreaker("HolographicProjection", failure_threshold=2, reset_timeout=10)
network_breaker = CircuitBreaker("NetworkIO", failure_threshold=5, reset_timeout=30)


class HealthMonitor:
    """
    Monitors system health by running registered health checks.
    Provides real-time status of all cosmos components.
    """

    def __init__(self):
        self._checks: dict[str, Callable] = {}
        self._last_check: Optional[float] = None
        self._status: SystemState = SystemState.HEALTHY
        self._component_status: dict[str, str] = {}
        self._metrics: dict = {}

    def register_check(self, name: str, check_fn: Callable[[], str]):
        """
        Register a health check function.

        Args:
            name: Component name
            check_fn: Function returning status string ("healthy", "degraded", "unhealthy", etc.)
        """
        self._checks[name] = check_fn
        logger.debug(f"Registered health check: {name}")

    def unregister_check(self, name: str):
        """Remove a health check."""
        self._checks.pop(name, None)

    async def check_health(self) -> "HealthStatus":
        """
        Run all health checks and return aggregated status.
        """
        self._last_check = time.time()
        self._component_status = {}

        unhealthy_count = 0
        degraded_count = 0

        for name, check_fn in self._checks.items():
            try:
                if asyncio.iscoroutinefunction(check_fn):
                    status = await check_fn()
                else:
                    status = check_fn()
                self._component_status[name] = status

                if status in ("unhealthy", "error", "critical", "missing"):
                    unhealthy_count += 1
                elif status in ("degraded", "warning", "uninitialized"):
                    degraded_count += 1

            except Exception as e:
                logger.error(f"Health check '{name}' failed: {e}")
                self._component_status[name] = f"error: {e}"
                unhealthy_count += 1

        # Determine overall status
        if unhealthy_count > 0:
            self._status = SystemState.CRITICAL
        elif degraded_count > 0:
            self._status = SystemState.DEGRADED
        else:
            self._status = SystemState.HEALTHY

        # Collect system metrics
        self._metrics = self._collect_system_metrics()

        return HealthStatus(
            status=self._status.value,
            components=self._component_status,
            system_metrics=self._metrics,
            timestamp=datetime.now().isoformat(),
        )

    def _collect_system_metrics(self) -> dict:
        """Collect basic system metrics."""
        try:
            import os
            import sys

            metrics = {
                "python_version": sys.version.split()[0],
                "pid": os.getpid(),
            }

            # Try to get memory info if psutil available
            try:
                import psutil
                process = psutil.Process()
                metrics["memory_mb"] = round(process.memory_info().rss / 1024 / 1024, 2)
                metrics["cpu_percent"] = process.cpu_percent()
            except ImportError:
                pass

            return metrics
        except Exception:
            return {}

    def get_status(self) -> dict:
        """Get current status without running checks."""
        return {
            "status": self._status.value,
            "components": self._component_status,
            "last_check": self._last_check,
        }


@dataclass
class HealthStatus:
    """Result of a health check."""
    status: str
    components: dict[str, str]
    system_metrics: dict
    timestamp: str


class BackupManager:
    """
    Manages automatic backups of cosmos data.
    Ensures data durability and recovery capability.
    """

    def __init__(
        self,
        data_dir: str,
        backup_dir: str,
        backup_interval_hours: float = 24.0,
        max_backups: int = 7,
    ):
        self.data_dir = Path(data_dir)
        self.backup_dir = Path(backup_dir)
        self.backup_interval = backup_interval_hours * 3600  # Convert to seconds
        self.max_backups = max_backups

        self._is_running = False
        self._task: Optional[asyncio.Task] = None
        self._last_backup: Optional[datetime] = None

        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    async def start(self):
        """Start the automatic backup scheduler."""
        if self._is_running:
            return

        self._is_running = True
        self._task = asyncio.create_task(self._backup_loop())
        logger.info(f"Backup Manager started (interval: {self.backup_interval/3600}h)")

    async def stop(self):
        """Stop the backup scheduler."""
        self._is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Backup Manager stopped")

    async def _backup_loop(self):
        """Background loop for scheduled backups."""
        while self._is_running:
            try:
                await asyncio.sleep(self.backup_interval)
                await self.create_backup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Backup failed: {e}")

    async def create_backup(self, label: Optional[str] = None) -> Optional[Path]:
        """
        Create a backup of the data directory.

        Args:
            label: Optional label for the backup (default: timestamp)

        Returns:
            Path to the backup archive, or None if failed
        """
        try:
            import shutil
            from datetime import datetime

            # Generate backup name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"cosmos_backup_{label or timestamp}"
            backup_path = self.backup_dir / backup_name

            # Create backup (copy directory)
            if self.data_dir.exists():
                shutil.copytree(
                    self.data_dir,
                    backup_path,
                    ignore=shutil.ignore_patterns("*.tmp", "__pycache__", "*.pyc"),
                )

                # Create archive
                archive_path = shutil.make_archive(
                    str(backup_path),
                    'zip',
                    self.backup_dir,
                    backup_name,
                )

                # Remove uncompressed copy
                shutil.rmtree(backup_path)

                self._last_backup = datetime.now()
                logger.info(f"Backup created: {archive_path}")

                # Cleanup old backups
                await self._cleanup_old_backups()

                return Path(archive_path)
            else:
                logger.warning(f"Data directory not found: {self.data_dir}")
                return None

        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return None

    async def _cleanup_old_backups(self):
        """Remove old backups exceeding max_backups limit."""
        try:
            backups = sorted(
                self.backup_dir.glob("cosmos_backup_*.zip"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )

            for old_backup in backups[self.max_backups:]:
                old_backup.unlink()
                logger.info(f"Removed old backup: {old_backup.name}")

        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")

    async def restore_backup(self, backup_path: Path) -> bool:
        """
        Restore from a backup archive.

        Args:
            backup_path: Path to the backup zip file

        Returns:
            True if restore succeeded
        """
        try:
            import shutil

            if not backup_path.exists():
                logger.error(f"Backup not found: {backup_path}")
                return False

            # Create restore point first
            await self.create_backup(label="pre_restore")

            # Extract backup
            shutil.unpack_archive(backup_path, self.data_dir.parent)

            logger.info(f"Restored from backup: {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False

    def list_backups(self) -> list[dict]:
        """list available backups."""
        backups = []
        for backup_file in sorted(
            self.backup_dir.glob("cosmos_backup_*.zip"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            stat = backup_file.stat()
            backups.append({
                "name": backup_file.name,
                "path": str(backup_file),
                "size_mb": round(stat.st_size / 1024 / 1024, 2),
                "created": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
        return backups

    def get_status(self) -> dict:
        """Get backup manager status."""
        return {
            "is_running": self._is_running,
            "last_backup": self._last_backup.isoformat() if self._last_backup else None,
            "backup_count": len(list(self.backup_dir.glob("cosmos_backup_*.zip"))),
            "backup_dir": str(self.backup_dir),
        }
