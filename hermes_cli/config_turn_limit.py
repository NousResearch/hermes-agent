"""Formatting for resolved foreground turn budgets."""

from hermes_cli.config import TURN_LIMIT_UNLIMITED


def turn_limit_label(limit: int) -> str:
    """How a resolved turn limit reads on screen."""
    return "unlimited" if limit == TURN_LIMIT_UNLIMITED else str(limit)
