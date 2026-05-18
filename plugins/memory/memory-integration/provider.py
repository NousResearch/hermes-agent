"""MemoryProvider implementation for the read-only memory-integration skeleton."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from agent.memory_provider import MemoryProvider

from .status import build_status

STATUS_SCHEMA = {
    "name": "memory_integration_status",
    "description": "Report read-only memory-integration vault and sidecar configuration status.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}


class MemoryIntegrationProvider(MemoryProvider):
    """Minimal read-only MemoryProvider for the vault memory integration."""

    def __init__(self) -> None:
        self._initialized = False
        self._hermes_home: Path | None = None

    @property
    def name(self) -> str:
        return "memory-integration"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        self._initialized = True
        hermes_home = kwargs.get("hermes_home")
        self._hermes_home = Path(hermes_home) if hermes_home else None

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [STATUS_SCHEMA]

    def status(
        self,
        *,
        config: Mapping[str, Any] | None = None,
        hermes_home: str | Path | None = None,
    ) -> dict[str, Any]:
        return build_status(
            config=config,
            hermes_home=hermes_home or self._hermes_home,
            initialized=self._initialized,
        )

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs: Any) -> str:
        if tool_name != "memory_integration_status":
            return json.dumps({"ok": False, "error": "unknown_tool", "tool": tool_name})
        result = self.status(config=kwargs.get("config"), hermes_home=kwargs.get("hermes_home"))
        return json.dumps(result, sort_keys=True)
