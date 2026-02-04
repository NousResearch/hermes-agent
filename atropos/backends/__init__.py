from __future__ import annotations

from typing import Any

from .base import ToolBackend
from .modal_backend import ModalBackendConfig, ModalToolBackend
from .nomad_backend import NomadBackendConfig, NomadToolBackend


def create_tool_backend(cfg: Any) -> ToolBackend:
    mode = str(getattr(cfg, "tool_pool_mode", "nomad")).strip().lower()
    if mode == "nomad":
        return NomadToolBackend(NomadBackendConfig.from_agent_env_config(cfg))
    if mode == "modal":
        return ModalToolBackend(ModalBackendConfig.from_agent_env_config(cfg))
    raise ValueError(f"Unknown tool_pool_mode: {mode}")


__all__ = [
    "ToolBackend",
    "create_tool_backend",
    "NomadBackendConfig",
    "NomadToolBackend",
    "ModalBackendConfig",
    "ModalToolBackend",
]

