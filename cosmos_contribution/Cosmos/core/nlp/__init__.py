"""
Cosmos Natural Language Task System.

Enables natural language commands like "Hey Farn, do this..." with intelligent
routing to appropriate handlers including:
- Code generation (evolution loop)
- Crypto trading (Bankr)
- Browser automation
- Communication
- Research
"""

from .intent_parser import IntentParser, Intent
from .task_classifier import TaskClassifier, TaskType
from .command_router import CommandRouter
from .entity_extractor import EntityExtractor

__all__ = [
    'IntentParser',
    'Intent',
    'TaskClassifier',
    'TaskType',
    'CommandRouter',
    'EntityExtractor',
]

# Global router instance
_router: "CommandRouter" = None


async def process_command(command: str) -> dict:
    """
    Process a natural language command.

    This is the main entry point for NL processing.

    Args:
        command: Natural language command (with or without wake word)

    Returns:
        Result dict with response and dict actions taken
    """
    global _router
    if _router is None:
        _router = CommandRouter()

    return await _router.execute(command)


def get_router() -> "CommandRouter":
    """Get the global command router."""
    global _router
    if _router is None:
        _router = CommandRouter()
    return _router
