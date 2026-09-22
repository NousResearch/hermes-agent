"""Dependency-light helpers shared by .env load and edit paths."""

from __future__ import annotations


def sanitize_env_lines(lines: list[str]) -> list[str]:
    """Normalize .env line endings/whitespace without changing assignment semantics.

    Content after the first ``=`` is opaque value data: a known variable name embedded in a value
    must never be reinterpreted as another assignment, so concatenated lines stay on one line.
    """
    sanitized: list[str] = []
    for line in lines:
        raw = line.rstrip("\r\n")
        stripped = raw.strip()
        # Blank lines and comments are preserved verbatim.
        sanitized.append((raw if not stripped or stripped.startswith("#") else stripped) + "\n")
    return sanitized
