"""Shared CLI hints for Hermes kanban argument parsing."""

from __future__ import annotations

BOARD_FLAG_PLACEMENT_HINT = (
    "--board is a global kanban option; put it before the subcommand, e.g. "
    "`hermes kanban --board incoming-knowledge list`."
)


def append_board_flag_placement_hint(message: str) -> str:
    """Append the --board placement hint to matching argparse errors."""
    if (
        "--board" in message
        and "unrecognized arguments" in message
        and BOARD_FLAG_PLACEMENT_HINT not in message
    ):
        return f"{message}\n{BOARD_FLAG_PLACEMENT_HINT}"
    return message


def looks_like_misplaced_kanban_board_error(message: str, argv: list[str]) -> bool:
    """Return True when a top-level argparse error lost kanban context."""
    return (
        "kanban" in argv
        and "--board" in message
        and "unrecognized arguments" in message
        and BOARD_FLAG_PLACEMENT_HINT not in message
    )
