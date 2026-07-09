"""Helpers for noninteractive chat query input sources."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TextIO


class QueryInputError(ValueError):
    """Raised when noninteractive query input is invalid."""


def normalize_query_text(text: str) -> str:
    """Normalize line endings while preserving the caller's text otherwise."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def resolve_query_input(
    *,
    query: str | None = None,
    q: str | None = None,
    slash: str | None = None,
    query_file: str | None = None,
    stdin_query: bool = False,
    stdin: TextIO | None = None,
) -> tuple[str | None, str | None]:
    """Resolve mutually-exclusive noninteractive query sources.

    Returns ``(query_text, source)`` where source is one of ``query``,
    ``slash``, ``query_file``, ``stdin_query``, or ``None``.
    """
    provided = []
    inline = query if query is not None else q
    if inline is not None:
        provided.append("query")
    if slash is not None:
        provided.append("slash")
    if query_file:
        provided.append("query_file")
    if stdin_query:
        provided.append("stdin_query")

    if len(provided) > 1:
        raise QueryInputError(
            "--query/-q, --slash, --query-file, and --stdin-query are mutually exclusive"
        )

    if slash is not None:
        return slash, "slash"
    if inline is not None:
        return inline, "query"
    if query_file:
        path = Path(query_file).expanduser()
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise QueryInputError(f"Could not read query file {path}: {exc}") from exc
        text = normalize_query_text(text)
        if text == "":
            raise QueryInputError(f"Query file is empty: {path}")
        return text, "query_file"
    if stdin_query:
        stream = stdin if stdin is not None else sys.stdin
        text = normalize_query_text(stream.read())
        if text == "":
            raise QueryInputError("stdin query is empty")
        return text, "stdin_query"
    return None, None
