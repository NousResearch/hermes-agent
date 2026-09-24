"""Profile-scoped opt-in and polling for idle Ollama release."""

from __future__ import annotations

import contextvars
import logging
import math
import platform
import subprocess
import sys
import threading
from functools import wraps
from pathlib import Path

from agent.ollama_idle_release import ActivityGate, IdleOllamaPolicy, _local_root

logger = logging.getLogger(__name__)
_POLL_SECONDS = 15


def release_enabled(home: Path) -> bool:
    from hermes_cli.config_effective import load_user_config_effective

    try:
        cfg = load_user_config_effective(home / "config.yaml", fail_closed=True)
        section = cfg.get("local_runtime")
        return isinstance(section, dict) and section.get("ollama_idle_release") is True
    except Exception:
        return False


def gpu_memory_pressure() -> bool | None:
    """Require less than 5% headroom; unknown hardware never implies pressure.

    Without a reliable model-to-card mapping, require every NVIDIA card to be
    pressured. Apple Silicon shares physical memory with the OS. Other devices
    are deliberately unknown until a reliable pressure source is available.
    """
    from hermes_cli.local_runtime.hardware import _nvidia_smi_path

    try:
        executable = _nvidia_smi_path()
        if executable:
            result = subprocess.run(
                [
                    executable,
                    "--query-gpu=memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return None
            cards = [
                tuple(float(v) for v in line.split(","))
                for line in result.stdout.splitlines()
            ]
            if not cards or any(len(card) != 2 for card in cards):
                return None
            if any(
                not (
                    math.isfinite(total)
                    and math.isfinite(free)
                    and total > 0
                    and 0 <= free <= total
                )
                for total, free in cards
            ):
                return None
            return all(free / total < 0.05 for total, free in cards)
        if sys.platform == "darwin" and platform.machine() == "arm64":
            import psutil

            memory = psutil.virtual_memory()
            if memory.total > 0 and 0 <= memory.available <= memory.total:
                return memory.available / memory.total < 0.05
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return None


class IdleReleaseRuntime:
    def __init__(self, *, start_thread: bool = True):
        self._lock = threading.Lock()
        self._entries: dict[
            tuple[str, str, str], tuple[Path, IdleOllamaPolicy, contextvars.Context]
        ] = {}
        self._thread: threading.Thread | None = None
        self._start_thread = start_thread
        self._previous_pressure = False
        self._confirmed_pressure = False

    def register(self, agent, home: Path, gate: ActivityGate) -> None:
        base_url = str(getattr(agent, "base_url", "") or "")
        model = str(getattr(agent, "model", "") or "")
        if not model or _local_root(base_url) is None or not release_enabled(home):
            return
        key = (str(home), base_url, model)
        with self._lock:
            existing = self._entries.get(key)
            api_key = str(getattr(agent, "api_key", "") or "")
            if existing is None or existing[1].api_key != api_key:
                policy = IdleOllamaPolicy(
                    base_url=base_url,
                    model=model,
                    enabled=True,
                    gate=gate,
                    pressure=lambda: self._confirmed_pressure,
                    api_key=api_key,
                )
                self._entries[key] = (home, policy, contextvars.copy_context())
            if self._start_thread and (
                self._thread is None or not self._thread.is_alive()
            ):
                self._thread = threading.Thread(
                    target=self._run, daemon=True, name="ollama-idle-release"
                )
                self._thread.start()

    def poll(self) -> None:
        with self._lock:
            entries = list(self._entries.items())
        # Re-read in the captured profile context before ANY hardware/provider I/O.
        active = []
        for key, (home, policy, context) in entries:
            if context.copy().run(release_enabled, home):
                active.append((policy, context))
            else:
                with self._lock:
                    self._entries.pop(key, None)
        if not active:
            self._previous_pressure = self._confirmed_pressure = False
            return
        pressure = gpu_memory_pressure() is True
        self._confirmed_pressure = pressure and self._previous_pressure
        self._previous_pressure = pressure
        for policy, context in active:
            context.copy().run(policy.tick)

    def _run(self) -> None:
        delay = threading.Event()
        while not delay.wait(_POLL_SECONDS):
            try:
                self.poll()
            except Exception:
                logger.debug("Idle release poll skipped", exc_info=True)
            with self._lock:
                if not self._entries:
                    self._thread = None
                    return


RUNTIME = IdleReleaseRuntime()


def guard_idle_release(method):
    """Keep complete turns visible across profiles, even when their policy is off.

    An opted-in profile must not release a shared model during another profile's
    turn. Recording activity does not start monitoring or alter provider requests.
    """

    @wraps(method)
    def guarded(agent, *args, **kwargs):
        from hermes_constants import get_default_hermes_root, get_hermes_home

        home = get_hermes_home()
        db_path = getattr(getattr(agent, "_session_db", None), "db_path", None)
        if isinstance(db_path, (str, Path)) and db_path:
            home = Path(db_path).parent
        gate = ActivityGate(get_default_hermes_root() / "runtime" / "model-activity")
        with gate.turn():
            RUNTIME.register(agent, home, gate)
            return method(agent, *args, **kwargs)

    return guarded
