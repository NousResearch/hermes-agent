"""Pure contracts shared by the plugin runtime and host adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class RegisteredApprovalTransport:
    """Plugin-owned approval transport retained by one profile's manager."""

    name: str
    present: Callable[..., Any]
    plugin_id: str
    profile_home: str
