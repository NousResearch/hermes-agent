"""Driver-agnostic out-of-band notice type shared by the agent, gateway, TUI and CLI."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class AgentNotice:
    """Out-of-band notice (``AIAgent.notice_callback`` / ``notice_clear_callback``); each driver
    renders its own way."""

    text: str
    level: str = "info"            # info | warn | error | success
    kind: str = "sticky"           # sticky | ttl
    ttl_ms: Optional[int] = None   # honored only when kind == "ttl"
    key: Optional[str] = None      # dedupe / fired-once-latch / clear key
    id: Optional[str] = None
