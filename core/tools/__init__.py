"""
Hermes Core Universal Tool Interface Package.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass
class ToolSchema:
    name: str
    description: str
    parameters: Dict[str, Any]
    required_permissions: list[str] = field(default_factory=list)


class ToolManager:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolSchema] = {}

    def register_tool(self, schema: ToolSchema) -> None:
        self._tools[schema.name] = schema

    def get_tool(self, name: str) -> Optional[ToolSchema]:
        return self._tools.get(name)


__all__ = ["ToolSchema", "ToolManager"]
