"""
Hermes Core Context Management Package.
Manages context sources, token budgets, and context injection.
"""

from typing import Any, Dict, List


class ContextManager:
    def assemble_context(
        self,
        user_prompt: str,
        session_id: str,
        memory_items: List[str],
        project_rules: List[str],
    ) -> Dict[str, Any]:
        return {
            "prompt": user_prompt,
            "session_id": session_id,
            "memory": memory_items,
            "rules": project_rules,
        }


__all__ = ["ContextManager"]
