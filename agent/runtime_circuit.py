"""Reset-aware circuit state for whole-agent subscription runtimes."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from utils import atomic_json_write


def runtime_target_key(agent: Any, target: dict[str, Any] | None = None) -> tuple[str, str, str]:
    source = target or {}
    return (
        str(source.get("runtime", getattr(agent, "runtime", "hermes")) or "hermes"),
        str(source.get("provider", getattr(agent, "provider", "")) or "").strip().lower(),
        str(source.get("model", getattr(agent, "model", "")) or "").strip(),
    )


def _account_key(agent: Any) -> str:
    attestation = getattr(agent, "_claude_max_attestation", None)
    return str(getattr(attestation, "account_key", "") or "profile-default")


def _state_path() -> Path:
    return get_hermes_home() / "state" / "runtime-circuits.json"


def _persistent_key(agent: Any, target: dict[str, Any] | None = None) -> str:
    return json.dumps([_account_key(agent), *runtime_target_key(agent, target)], separators=(",", ":"))


def _load_persistent() -> dict[str, float]:
    try:
        payload = json.loads(_state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    result: dict[str, float] = {}
    for key, value in payload.items():
        try:
            result[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return result


@contextmanager
def _persistent_lock():
    path = _state_path().with_suffix(".lock")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        os.chmod(path, 0o600)
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass
        handle.close()


def _store_persistent(circuits: dict[str, float]) -> None:
    path = _state_path()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    now = time.time()
    live = {key: until for key, until in circuits.items() if until > now}
    atomic_json_write(path, live, mode=0o600, indent=2, sort_keys=True)


def open_runtime_circuit(
    agent: Any,
    *,
    reset_at: int | float | None,
    fallback_seconds: int = 60,
) -> float:
    until = float(reset_at) if reset_at else time.time() + fallback_seconds
    circuits = getattr(agent, "_runtime_circuits", None)
    if circuits is None:
        circuits = {}
        agent._runtime_circuits = circuits
    circuits[runtime_target_key(agent)] = until
    with _persistent_lock():
        persistent = _load_persistent()
        persistent[_persistent_key(agent)] = until
        _store_persistent(persistent)
    return until


def runtime_circuit_open_until(
    agent: Any, target: dict[str, Any] | None = None
) -> float | None:
    circuits = getattr(agent, "_runtime_circuits", None) or {}
    key = runtime_target_key(agent, target)
    until = float(circuits.get(key) or 0)
    if until <= time.time():
        persistent = _load_persistent()
        until = float(persistent.get(_persistent_key(agent, target)) or 0)
        if until > time.time():
            circuits = getattr(agent, "_runtime_circuits", None)
            if circuits is None:
                circuits = {}
                agent._runtime_circuits = circuits
            circuits[key] = until
    if until <= time.time():
        circuits.pop(key, None)
        return None
    return until


__all__ = [
    "open_runtime_circuit",
    "runtime_circuit_open_until",
    "runtime_target_key",
]
