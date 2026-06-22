"""External memory-provider adapter status helpers for evals.

The local deterministic evals intentionally do not depend on these providers.
This module reports opt-in adapter readiness without making network calls.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any


_ADAPTERS = {
    "mem0": {"module": "mem0", "env": "MEM0_API_KEY"},
    "zep_graphiti": {"module": "graphiti_core", "env": "ZEP_API_KEY"},
    "letta": {"module": "letta", "env": "LETTA_API_KEY"},
    "langmem": {"module": "langmem", "env": ""},
    "cognee": {"module": "cognee", "env": ""},
    "supermemory": {"module": "supermemory", "env": "SUPERMEMORY_API_KEY"},
}


def external_adapter_status() -> dict[str, dict[str, Any]]:
    status: dict[str, dict[str, Any]] = {}
    for name, config in _ADAPTERS.items():
        module_name = str(config["module"])
        env_name = str(config["env"])
        module_available = importlib.util.find_spec(module_name) is not None
        env_available = bool(os.getenv(env_name)) if env_name else True
        status[name] = {
            "available": bool(module_available and env_available),
            "module": module_name,
            "module_available": module_available,
            "required_env": env_name,
            "env_available": env_available,
        }
    return status
