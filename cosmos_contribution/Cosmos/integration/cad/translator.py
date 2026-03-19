"""
CAD Natural Language Translator.

Translates natural language to CadQuery code.
"""

import re
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CADCommand:
    """Translated CAD command."""
    code: str
    original: str
    operation: str


class CADTranslator:
    """
    Translate natural language to CadQuery operations.

    Uses pattern matching for common shapes and LLM for complex designs.
    """

    PATTERNS = {
        r"(?:create|make|build) (?:a )?box (\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)": "box",
        r"(?:create|make|build) (?:a )?cylinder (?:with )?(?:radius )?(\d+(?:\.\d+)?)\s*(?:,|and)?\s*(?:height )?(\d+(?:\.\d+)?)": "cylinder",
        r"(?:create|make|build) (?:a )?sphere (?:with )?(?:radius )?(\d+(?:\.\d+)?)": "sphere",
        r"fillet (?:all )?edges? (?:with )?(?:radius )?(\d+(?:\.\d+)?)": "fillet",
        r"chamfer (?:all )?edges? (\d+(?:\.\d+)?)": "chamfer",
        r"export (?:to )?stl(?: as (.+))?": "export_stl",
        r"export (?:to )?step(?: as (.+))?": "export_step",
    }

    def __init__(self, llm=None):
        self.llm = llm
        self._compiled = [
            (re.compile(p, re.IGNORECASE), op)
            for p, op in self.PATTERNS.items()
        ]

    def translate(self, natural_command: str) -> CADCommand:
        """
        Translate natural language to CadQuery code.

        Args:
            natural_command: Natural language description

        Returns:
            CADCommand with executable code
        """
        for pattern, operation in self._compiled:
            match = pattern.search(natural_command)
            if match:
                return self._handle_match(match, operation, natural_command)

        # Complex design - would use LLM
        return self._translate_complex(natural_command)

    def _handle_match(
        self,
        match: re.Match,
        operation: str,
        original: str
    ) -> CADCommand:
        """Handle a matched pattern."""
        groups = match.groups()

        if operation == "box":
            l, w, h = map(float, groups[:3])
            code = f"result = cq.Workplane('XY').box({l}, {w}, {h})"
            return CADCommand(code, original, operation)

        elif operation == "cylinder":
            r, h = map(float, groups[:2])
            code = f"result = cq.Workplane('XY').cylinder({h}, {r})"
            return CADCommand(code, original, operation)

        elif operation == "sphere":
            r = float(groups[0])
            code = f"result = cq.Workplane('XY').sphere({r})"
            return CADCommand(code, original, operation)

        elif operation == "fillet":
            r = float(groups[0])
            code = f"result = result.edges().fillet({r})"
            return CADCommand(code, original, operation)

        elif operation == "chamfer":
            s = float(groups[0])
            code = f"result = result.edges().chamfer({s})"
            return CADCommand(code, original, operation)

        elif operation == "export_stl":
            filename = groups[0] or "output.stl"
            code = f"cq.exporters.export(result, '{filename}')"
            return CADCommand(code, original, operation)

        elif operation == "export_step":
            filename = groups[0] or "output.step"
            code = f"cq.exporters.export(result, '{filename}', exportType='STEP')"
            return CADCommand(code, original, operation)

        logger.warning(f"No pattern matched for CAD command: {original}")
        return CADCommand(f"// Unrecognized command: {original}", original, "comment")

    def _translate_complex(self, command: str) -> CADCommand:
        """Translate complex designs using LLM."""
        # Placeholder for LLM-based translation
        logger.warning(f"Complex design translation not implemented: {command}")

        # Attempt keyword-based categorization
        keywords = {"create": "create", "make": "create", "build": "create", "add": "create",
                    "move": "transform", "rotate": "transform", "scale": "transform", "translate": "transform",
                    "delete": "delete", "remove": "delete", "erase": "delete",
                    "select": "select", "pick": "select", "highlight": "select",
                    "export": "export", "save": "export", "output": "export"}
        cmd_lower = command.lower()
        operation = "unknown"
        for kw, op in keywords.items():
            if kw in cmd_lower:
                operation = op
                break
        return CADCommand(
            f"// Complex command ({operation}): {command}",
            command,
            operation
        )

    async def translate_async(self, command: str) -> CADCommand:
        """Async version of translate."""
        return self.translate(command)
