"""Hindsight adapter for dynamic memory writes."""

from __future__ import annotations

import json

from knowledge.adapters.base import WriteResult
from knowledge.types import KnowledgeWriteRequest

DEFAULT_CONTEXT = "Hermes knowledge router"


class HindsightMemoryAdapter:
    @property
    def name(self) -> str:
        return "hindsight"

    def write(self, request: KnowledgeWriteRequest) -> WriteResult:
        from plugins.memory.hindsight import HindsightMemoryProvider

        provider = HindsightMemoryProvider()
        provider.initialize("knowledge-router", platform="tool")
        try:
            result_text = provider.handle_tool_call(
                "hindsight_retain",
                {
                    "content": request.content,
                    "context": request.context or DEFAULT_CONTEXT,
                    "tags": list(request.tags),
                },
            )
            parsed = json.loads(result_text) if isinstance(result_text, str) else {}
            if isinstance(parsed, dict) and parsed.get("error"):
                return WriteResult(success=False, backend=self.name, action="retain", error=str(parsed["error"]), data=parsed)
            return WriteResult(success=True, backend=self.name, action="retain", data=parsed if isinstance(parsed, dict) else {})
        finally:
            provider.shutdown()
