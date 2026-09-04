"""Optional local zero-tier — port of local/local-zero-tier.ts.

Strictly opt-in (``model_router.local_zero.enabled: true``). When enabled,
configured local inference endpoints (LM Studio / Ollama) are probed with a
short-timeout HTTP GET; trivial / low-intensity turns may then route to the
configured local model. Every failure mode (network down, timeout, import
error) degrades to "not decided" — this stage never breaks routing.
"""
from __future__ import annotations

import socket
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class LocalZeroConfig:
    enabled: bool = False
    endpoints: tuple = ()
    model: str = ""
    timeout_ms: int = 1500


def ping_local_services(config: LocalZeroConfig) -> bool:
    """True when any configured endpoint answers within the timeout."""
    if not config.enabled or not config.endpoints:
        return False
    timeout = max(0.1, config.timeout_ms / 1000.0)
    for endpoint in config.endpoints:
        try:
            request = urllib.request.Request(str(endpoint), method="GET")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status < 500:
                    return True
        except Exception:
            continue
    return False
