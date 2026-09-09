"""
Hermes Core Session Package.
Manages session creation, archiving, history, and metadata.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional


class SessionManager:
    def create_session(self, prefix: str = "session") -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}"


__all__ = ["SessionManager"]
