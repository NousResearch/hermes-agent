"""
UE5 Natural Language to Python Translator.
"""

import re
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class UE5Command:
    """Translated UE5 command."""
    code: str
    original: str
    operation: str
    parameters: Dict[str, Any]


class UE5CommandTranslator:
    """
    Translate natural language to UE5 Python API calls.

    Uses pattern matching for common commands and LLM for complex ones.
    """

    COMMAND_PATTERNS = {
        r"spawn (\d+) (.+) at (\d+),\s*(\d+),\s*(\d+)": "spawn_actors",
        r"create material (.+) with color (.+)": "create_material",
        r"import assets? from (.+)": "import_assets",
        r"build lighting": "build_lighting",
        r"take screenshot(?: to (.+))?": "take_screenshot",
        r"select all (.+)": "select_actors",
        r"delete selected": "delete_selected",
        r"move selected to (\d+),\s*(\d+),\s*(\d+)": "move_actors",
    }

    def __init__(self, llm=None):
        self.llm = llm
        self._compiled_patterns = [
            (re.compile(p, re.IGNORECASE), op)
            for p, op in self.COMMAND_PATTERNS.items()
        ]

    def translate(self, natural_command: str) -> UE5Command:
        """
        Translate natural language to UE5 Python code.

        Args:
            natural_command: Natural language command

        Returns:
            UE5Command with executable code
        """
        # Try pattern matching first
        for pattern, operation in self._compiled_patterns:
            match = pattern.search(natural_command)
            if match:
                return self._handle_pattern_match(match, operation, natural_command)

        # Fall back to LLM translation
        return self._translate_with_llm(natural_command)

    def _handle_pattern_match(
        self,
        match: re.Match,
        operation: str,
        original: str
    ) -> UE5Command:
        """Handle a matched pattern."""
        groups = match.groups()

        if operation == "spawn_actors":
            count, actor_type, x, y, z = groups
            code = f"""
for i in range({count}):
    offset = i * 200
    loc = unreal.Vector({x} + offset, {y}, {z})
    actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
        unreal.load_class(None, "{actor_type}"),
        loc
    )
"""
            return UE5Command(code, original, operation, {
                "count": int(count),
                "actor_type": actor_type,
                "location": (float(x), float(y), float(z))
            })

        elif operation == "build_lighting":
            return UE5Command(
                "unreal.EditorLevelLibrary.build_lighting()",
                original, operation, {}
            )

        elif operation == "take_screenshot":
            path = groups[0] or "/tmp/screenshot.png"
            return UE5Command(
                f'unreal.AutomationLibrary.take_high_res_screenshot(1920, 1080, "{path}")',
                original, operation, {"path": path}
            )

        # Default
        logger.warning(f"No handler for UE5 operation '{operation}': {original}")
        return UE5Command(f"// Unrecognized: {original}", original, operation, {})

    def _translate_with_llm(self, command: str) -> UE5Command:
        """Translate using LLM for complex commands."""
        # Placeholder - would use actual LLM
        logger.warning(f"LLM translation not implemented for: {command}")

        # Attempt keyword-based categorization
        keywords = {"spawn": "spawn", "create": "create", "make": "create", "build": "create", "add": "create",
                    "move": "transform", "rotate": "transform", "scale": "transform", "translate": "transform",
                    "delete": "delete", "remove": "delete", "destroy": "delete",
                    "select": "select", "pick": "select", "highlight": "select",
                    "import": "import", "export": "export", "save": "export",
                    "light": "lighting", "build lighting": "lighting",
                    "screenshot": "screenshot", "capture": "screenshot"}
        cmd_lower = command.lower()
        operation = "unknown"
        for kw, op in keywords.items():
            if kw in cmd_lower:
                operation = op
                break
        return UE5Command(
            f"// Complex command ({operation}): {command}",
            command,
            operation,
            {}
        )

    async def translate_async(self, natural_command: str) -> UE5Command:
        """Async version of translate."""
        return self.translate(natural_command)
