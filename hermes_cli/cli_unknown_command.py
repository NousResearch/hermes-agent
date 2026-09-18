"""Copy for an unknown slash command: says nothing was sent and suggests a near-miss."""

from __future__ import annotations

import difflib
from collections.abc import Iterable


def unknown_command_lines(typed: str, known: Iterable[str]) -> tuple[str, str]:
    """(lead line, pointer line) for a slash token with no handler.

    ``known`` holds names WITH the leading slash (``hermes_cli.commands.COMMANDS`` keys plus skill
    commands). A close match (typo) becomes ``Did you mean /model?``; prefix expansion already ran
    before this, so only fuzzy matches are considered here.
    """
    base = typed.split()[0] if typed.split() else typed
    close = difflib.get_close_matches(base, list(known), n=1, cutoff=0.6)
    hint = f" Did you mean {close[0]}?" if close else ""
    return (f"Unknown command {base} — nothing was sent.{hint}", "Type /help for the full list.")
