"""Recover pre-prune tool bodies for Anchor Index extraction only."""

import re
from collections import Counter
from typing import Any

# No single anchor is this long, and the Anchor Index's file pattern backtracks quadratically over such a run
# (a 100,000-char dotted run takes seconds), so a recovered body drops it before indexing.
_LONG_IDENTIFIER_RUN = re.compile(r"(?<![\w./-])[\w./-]{512,}")


def with_pre_prune_tool_bodies(
    turns: list[dict[str, Any]], source: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Substitute unambiguous string bodies without changing prompts or returned stubs."""
    rows = [row for row in source or () if row.get("role") == "tool" and row.get("tool_call_id")]
    counts = Counter(row["tool_call_id"] for row in rows)
    bodies = {
        row["tool_call_id"]: _LONG_IDENTIFIER_RUN.sub(" ", row["content"]) for row in rows
        if counts[row["tool_call_id"]] == 1 and isinstance(row.get("content"), str)
    }
    return [
        {**row, "content": bodies[row["tool_call_id"]]}
        if row.get("role") == "tool" and row.get("tool_call_id") in bodies
        and isinstance(row.get("content"), str) else row
        for row in turns
    ]
