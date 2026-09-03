"""Persistent Agent Computers + Human Takeover (BWM-796).

Reuses the existing Hermes Chromium/CDP launch shape. Does not build a
second browser runtime. AgentComputer and BrowserIdentity stay separate.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from .adapter import HermesChromiumRuntime, InMemoryRuntime, new_identity_profile_dir
from .contract import AgentComputerContract
from .errors import AgentComputerError
from .service import AgentComputerService
from .store import AgentComputerStore

_SERVICE: AgentComputerService | None = None
_LOCK = threading.Lock()


def default_data_root() -> Path:
    override = os.environ.get("HERMES_AGENT_COMPUTER_ROOT", "").strip()
    if override:
        return Path(override)
    try:
        from hermes_constants import get_default_hermes_root

        return get_default_hermes_root() / "agent-computers"
    except Exception:
        return Path.home() / ".hermes" / "agent-computers"


def resolve_runtime_name(
    env: str | None = None,
    config: dict | None = None,
) -> str:
    """Choose memory vs chromium.

    Order: explicit env override (tests/operators) → config.yaml
    ``agent_computer.runtime`` → memory. Not a user-facing HERMES_* setting.
    """
    env_val = os.environ.get("HERMES_AGENT_COMPUTER_RUNTIME", "") if env is None else env
    env_val = (env_val or "").strip()
    if env_val:
        return env_val
    if config is None:
        try:
            from hermes_cli.config import load_config

            config = load_config()
        except Exception:
            return "memory"
    section = (config or {}).get("agent_computer") or {}
    name = str(section.get("runtime") or "memory").strip()
    return name or "memory"


def build_service(
    *,
    data_root: str | Path | None = None,
    runtime=None,
) -> AgentComputerService:
    root = Path(data_root or default_data_root())
    root.mkdir(parents=True, exist_ok=True)
    if runtime is None:
        if resolve_runtime_name() == "chromium":
            runtime = HermesChromiumRuntime()
        else:
            runtime = InMemoryRuntime()
    store = AgentComputerStore(root / "state.db")
    return AgentComputerService(store, runtime, data_root=root)


def get_service() -> AgentComputerService:
    global _SERVICE
    with _LOCK:
        if _SERVICE is None:
            _SERVICE = build_service()
        return _SERVICE


def get_contract() -> AgentComputerContract:
    return AgentComputerContract(get_service())


def reset_service_for_tests() -> None:
    global _SERVICE
    with _LOCK:
        _SERVICE = None


def release_owner_for_transport_if_active(transport: object) -> int:
    """WS teardown hook. No-ops until a computer service has been created."""
    svc = _SERVICE
    if svc is None:
        return 0
    return svc.release_owner_for_transport(transport)


__all__ = [
    "AgentComputerContract",
    "AgentComputerError",
    "AgentComputerService",
    "AgentComputerStore",
    "HermesChromiumRuntime",
    "InMemoryRuntime",
    "build_service",
    "default_data_root",
    "get_contract",
    "get_service",
    "new_identity_profile_dir",
    "release_owner_for_transport_if_active",
    "reset_service_for_tests",
    "resolve_runtime_name",
]
