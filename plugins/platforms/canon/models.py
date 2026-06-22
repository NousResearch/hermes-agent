"""Shared data models and errors for the Canon platform plugin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


class CanonApiError(Exception):
    """HTTP error from Canon's agent API."""

    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body.strip()
        super().__init__(f"Canon API returned {status_code}: {self.body[:500]}")

    @property
    def retryable(self) -> bool:
        return self.status_code in {408, 425, 429} or self.status_code >= 500


@dataclass
class CanonStreamFrame:
    event: str
    data: Any
    event_id: Optional[str] = None


@dataclass
class CanonResolvedAgent:
    api_key: str = ""
    profile: Optional[str] = None
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    base_url: str = ""
    stream_url: str = ""
