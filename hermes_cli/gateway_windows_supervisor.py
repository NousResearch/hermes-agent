"""Bounded Windows gateway supervisor used by the Scheduled Task launcher.

Task Scheduler is the login-time owner; this process is the crash-recovery
authority.  It owns exactly one ``gateway run --external-supervisor`` child,
preserves semantic exit codes, and never starts a messaging poller itself.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from gateway.restart import (
    EXTERNAL_GATEWAY_SUPERVISOR_ENV,
    GATEWAY_FATAL_CONFIG_EXIT_CODE,
    GATEWAY_SERVICE_RESTART_EXIT_CODE,
)
from utils import atomic_json_write


logger = logging.getLogger(__name__)

RECOVERY_MARKER_NAME = ".gateway_recovery.json"
SUPERVISOR_LOCK_NAME = ".gateway-supervisor.lock"
RECOVERY_SCHEMA = "hermes.gateway-recovery.r3"
RECOVERY_BACKOFF_SECONDS = (5.0, 15.0, 30.0, 60.0, 60.0)
RECOVERY_STABLE_SECONDS = 120.0
FAILURE_NOTIFICATION_TIMEOUT_SECONDS = 15.0
RECOVERY_INCIDENT_ENV = "HERMES_GATEWAY_RECOVERY_INCIDENT_ID"


def recovery_marker_path(home: Path) -> Path:
    return Path(home) / RECOVERY_MARKER_NAME


def read_recovery_marker(home: Path) -> dict[str, Any] | None:
    """Return a validated recovery marker, or ``None`` for absent/malformed data."""
    try:
        value = json.loads(recovery_marker_path(home).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(value, dict) or value.get("schema") != RECOVERY_SCHEMA:
        return None
    return value


def recovery_marker_for_pid(home: Path, pid: int) -> dict[str, Any] | None:
    """Return the pending marker only when it names this exact gateway life."""
    marker = read_recovery_marker(home)
    if marker is None or marker.get("state") != "starting":
        return None
    try:
        return marker if int(marker.get("new_pid")) == int(pid) else None
    except (TypeError, ValueError):
        return None


def claim_recovery_marker(home: Path, incident_id: str, pid: int) -> bool:
    """Bind a pending recovery to the exact gateway runtime PID.

    On Windows the venv launcher is a waiting shim whose PID differs from the
    Python runtime recorded in ``gateway_state.json``.  Only the child that
    inherited this incident id may claim the marker, so notification authority
    never binds to the shim or to an unrelated gateway life.
    """
    marker = read_recovery_marker(home)
    if (
        marker is None
        or marker.get("incident_id") != incident_id
        or marker.get("state") != "starting"
    ):
        return False
    existing = marker.get("new_pid")
    if existing not in (None, int(pid)):
        return False
    marker.update(new_pid=int(pid), claimed_at=time.time())
    atomic_json_write(recovery_marker_path(home), marker, indent=None)
    return True


def format_recovery_message(marker: Mapping[str, Any], *, now: float | None = None) -> str:
    """Render the user-facing recovery message without exposing command/env details."""
    current = time.time() if now is None else float(now)
    try:
        downtime = max(0, round(current - float(marker.get("detected_at") or current)))
    except (TypeError, ValueError):
        downtime = 0
    old_pid = marker.get("old_pid", "?")
    new_pid = marker.get("new_pid", "?")
    if marker.get("kind") == "crash":
        attempt = marker.get("attempt", "?")
        maximum = marker.get("max_attempts", len(RECOVERY_BACKOFF_SECONDS))
        return (
            f"✅ Hermes автоматически восстановлен после сбоя за {downtime} с. "
            f"PID {old_pid} → {new_pid}, попытка {attempt}/{maximum}."
        )
    return f"✅ Hermes восстановлен за {downtime} с. PID {old_pid} → {new_pid}."


def mark_recovery_notification_delivered(home: Path, pid: int, *, now: float | None = None) -> bool:
    """Consume notification authority for this exact recovered gateway PID."""
    marker = recovery_marker_for_pid(home, pid)
    if marker is None:
        return False
    marker.update(
        state="recovered",
        recovered_at=time.time() if now is None else float(now),
        notification_delivered=True,
    )
    atomic_json_write(recovery_marker_path(home), marker, indent=None)
    return True


def _profile_args(profile: str | None) -> list[str]:
    return ["--profile", profile] if profile else []


def build_gateway_child_argv(python_exe: str, profile: str | None) -> list[str]:
    return [
        python_exe,
        "-m",
        "hermes_cli.main",
        *_profile_args(profile),
        "gateway",
        "run",
        "--external-supervisor",
    ]


def build_failure_send_argv(python_exe: str, profile: str | None, message: str) -> list[str]:
    return [
        python_exe,
        "-m",
        "hermes_cli.main",
        *_profile_args(profile),
        "send",
        "--to",
        "telegram",
        "--quiet",
        message,
    ]


def _failure_notification_enabled() -> bool:
    """Critical recovery alerts use their own opt-out, separate from routine restart noise."""
    try:
        from gateway.config import Platform, load_gateway_config

        platform = load_gateway_config().platforms.get(Platform.TELEGRAM)
        return bool(platform and platform.enabled and platform.gateway_restart_failure_notification)
    except Exception:
        logger.warning("Could not resolve Telegram failure-notification config", exc_info=True)
        return False


def send_failure_notification(
    python_exe: str,
    profile: str | None,
    message: str,
    *,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> bool:
    """Best-effort outbound-only alert. No gateway adapter or poller is started."""
    if not _failure_notification_enabled():
        return False
    try:
        result = run(
            build_failure_send_argv(python_exe, profile, message),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=FAILURE_NOTIFICATION_TIMEOUT_SECONDS,
            check=False,
            env=os.environ.copy(),
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        logger.warning("Gateway recovery failure alert could not be delivered", exc_info=True)
        return False


@contextmanager
def supervisor_lock(home: Path) -> Iterator[bool]:
    """Hold a profile-scoped OS lock for the supervisor's full lifetime."""
    from gateway.status import _release_file_lock, _try_acquire_file_lock

    path = Path(home) / SUPERVISOR_LOCK_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+b")
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        acquired = _try_acquire_file_lock(handle)
        if acquired:
            with suppress(OSError):
                handle.seek(0)
                handle.truncate()
                handle.write(str(os.getpid()).encode("ascii"))
                handle.flush()
        try:
            yield acquired
        finally:
            if acquired:
                _release_file_lock(handle)
    finally:
        handle.close()


class GatewayWindowsSupervisor:
    """Pure state machine around an injected child-process and clock boundary."""

    def __init__(
        self,
        *,
        home: Path,
        python_exe: str,
        profile: str | None,
        popen: Callable[..., Any] = subprocess.Popen,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        wall_time: Callable[[], float] = time.time,
        notifier: Callable[[str, str | None, str], bool] = send_failure_notification,
        backoffs: Sequence[float] = RECOVERY_BACKOFF_SECONDS,
        stable_seconds: float = RECOVERY_STABLE_SECONDS,
    ) -> None:
        self.home = Path(home)
        self.python_exe = python_exe
        self.profile = profile
        self.popen = popen
        self.sleep = sleep
        self.monotonic = monotonic
        self.wall_time = wall_time
        self.notifier = notifier
        self.backoffs = tuple(float(v) for v in backoffs)
        self.stable_seconds = float(stable_seconds)
        self._marker: dict[str, Any] | None = None

    def _write_marker(self, **fields: Any) -> dict[str, Any]:
        marker = dict(self._marker or {})
        marker.update(fields)
        marker.setdefault("schema", RECOVERY_SCHEMA)
        marker.setdefault("incident_id", str(uuid.uuid4()))
        marker.setdefault("max_attempts", len(self.backoffs))
        marker.setdefault("notification_delivered", False)
        atomic_json_write(recovery_marker_path(self.home), marker, indent=None)
        self._marker = marker
        return marker

    def _start_child(self, *, attempt: int | None = None) -> tuple[Any | None, float]:
        env = os.environ.copy()
        env[EXTERNAL_GATEWAY_SUPERVISOR_ENV] = "1"
        if self._marker is not None:
            updates: dict[str, Any] = {"state": "starting", "new_pid": None}
            if attempt is not None:
                updates["attempt"] = attempt
            marker = self._write_marker(**updates)
            env[RECOVERY_INCIDENT_ENV] = str(marker["incident_id"])
        started_at = self.monotonic()
        try:
            process = self.popen(build_gateway_child_argv(self.python_exe, self.profile), env=env)
        except OSError:
            logger.exception("Gateway child spawn failed")
            return None, started_at
        return process, started_at

    def _runtime_pid(self, fallback: int) -> int:
        """Resolve the gateway runtime PID preserved in its terminal status record."""
        try:
            payload = json.loads((self.home / "gateway_state.json").read_text(encoding="utf-8"))
            runtime_pid = int(payload.get("pid"))
            return runtime_pid if runtime_pid > 0 else int(fallback)
        except (OSError, ValueError, TypeError, AttributeError):
            return int(fallback)

    def _new_incident(self, *, kind: str, old_pid: int, exit_code: int, attempt: int = 0) -> None:
        self._marker = None
        self._write_marker(
            schema=RECOVERY_SCHEMA,
            incident_id=str(uuid.uuid4()),
            kind=kind,
            state="waiting" if kind == "crash" else "starting",
            detected_at=self.wall_time(),
            old_pid=int(old_pid),
            last_exit_code=int(exit_code),
            attempt=attempt,
            max_attempts=len(self.backoffs),
            next_attempt_at=None,
            new_pid=None,
            recovered_at=None,
            notification_delivered=False,
        )

    def _fail(self, exit_code: int) -> int:
        if self._marker is None:
            self._new_incident(kind="crash", old_pid=0, exit_code=exit_code)
        marker = self._write_marker(
            state="failed",
            last_exit_code=int(exit_code),
            next_attempt_at=None,
        )
        attempts = marker.get("attempt", len(self.backoffs))
        message = (
            f"🚨 Hermes не удалось восстановить после {attempts} попыток. "
            f"Последний exit {exit_code}. Требуется ручной запуск."
        )
        self.notifier(self.python_exe, self.profile, message)
        return int(exit_code or 1)

    def run(self) -> int:
        recovery_attempt = 0
        process, started_at = self._start_child()
        while True:
            if process is None:
                exit_code = 1
                child_pid = 0
            else:
                child_pid = int(process.pid)
                try:
                    exit_code = int(process.wait())
                except OSError:
                    logger.exception("Gateway child wait failed")
                    exit_code = 1
                child_pid = self._runtime_pid(child_pid)
            lived = max(0.0, self.monotonic() - started_at)

            if exit_code == 0:
                return 0

            if exit_code == GATEWAY_SERVICE_RESTART_EXIT_CODE:
                recovery_attempt = 0
                self._new_incident(
                    kind="planned", old_pid=child_pid, exit_code=exit_code, attempt=0,
                )
                process, started_at = self._start_child(attempt=0)
                continue

            if exit_code == GATEWAY_FATAL_CONFIG_EXIT_CODE:
                if self._marker is None:
                    self._new_incident(
                        kind="crash", old_pid=child_pid, exit_code=exit_code,
                    )
                return self._fail(exit_code)

            if lived >= self.stable_seconds:
                recovery_attempt = 0
                self._marker = None

            if self._marker is None or self._marker.get("kind") != "crash":
                self._new_incident(
                    kind="crash", old_pid=child_pid, exit_code=exit_code,
                )
            else:
                self._write_marker(last_exit_code=exit_code, old_pid=child_pid)

            if recovery_attempt >= len(self.backoffs):
                return self._fail(exit_code)

            recovery_attempt += 1
            delay = self.backoffs[recovery_attempt - 1]
            self._write_marker(
                state="waiting",
                attempt=recovery_attempt,
                next_attempt_at=self.wall_time() + delay,
                new_pid=None,
            )
            self.sleep(delay)
            process, started_at = self._start_child(attempt=recovery_attempt)


def _configure_logging(home: Path) -> None:
    log_dir = Path(home) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(log_dir / "gateway-supervisor.log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hermes Windows gateway supervisor")
    parser.add_argument("--profile", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    from hermes_cli.config import get_hermes_home

    home = Path(get_hermes_home())
    _configure_logging(home)
    with supervisor_lock(home) as acquired:
        if not acquired:
            logger.error("A gateway supervisor already owns %s", home)
            return 73
        supervisor = GatewayWindowsSupervisor(
            home=home,
            python_exe=sys.executable,
            profile=args.profile,
        )
        return supervisor.run()


if __name__ == "__main__":
    raise SystemExit(main())
