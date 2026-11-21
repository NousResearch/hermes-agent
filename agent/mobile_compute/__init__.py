"""Mobile Compute Node — isolated compute substrate for mobile devices (iPhone 16e).

Architecture:
    Hermes (Windows)                    iPhone 16e (iOS)
    ─────────────────                   ──────────────────
    agent.mobile_compute.client         mobile_compute/ (Swift)
         │                                    │
         ├─ MobileComputeClient              ├─ /health (GET)
         │   (async HTTP, mTLS, retry)        ├─ /capabilities (GET)
         │                                    ├─ /compute (POST)
         ├─ MobileComputeServer               │   (task dispatch)
         │   (aiohttp bridge, optional)       └─ /tasks (GET)
         │
         └─ MobileComputeBridge
             (config, auth, lifecycle)

Design constraints:
- No LLM calls from this module (model routing stays in tools/model_routing.yaml).
- No Core ML integration yet (phase 0 = network + scaffold only).
- No changes to existing Hermes/JARVIS code paths.
- Pure-Python stdlib + aiohttp (already a core dep via httpx[socks]).
- Isolated: all state under ~/.hermes/mobile_compute/ (or $HERMES_HOME/mobile_compute/).

"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agent.mobile_compute.client import MobileComputeClient
from agent.mobile_compute.config import MobileComputeConfig, MobileComputeClientConfig
from agent.mobile_compute.bridge import MobileComputeBridge, MobileComputeBridgeStatus
from agent.mobile_compute.server import MobileComputeServer, MobileComputeServerConfig

__all__ = [
    "MobileComputeClient",
    "MobileComputeClientConfig",
    "MobileComputeConfig",
    "MobileComputeBridge",
    "MobileComputeBridgeStatus",
    "MobileComputeServer",
    "MobileComputeServerConfig",
]

logger = logging.getLogger("agent.mobile_compute")