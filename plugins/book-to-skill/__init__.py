"""Hermes plugin bridge for virgiliojr94/book-to-skill."""

from __future__ import annotations

from .cli import book_to_skill_command, register_cli


def register(ctx) -> None:
    """Register ``hermes book-to-skill`` — no core tools; skill slash comes from ~/.hermes/skills/."""
    ctx.register_cli_command(
        name="book-to-skill",
        help="Install and manage the book-to-skill document converter skill",
        setup_fn=register_cli,
        handler_fn=book_to_skill_command,
        description=(
            "Clone https://github.com/virgiliojr94/book-to-skill, link it into "
            f"your Hermes skills directory, and run extractor dependency checks. "
            "After install, use /book-to-skill in chat to convert PDFs, EPUBs, and "
            "other documents into structured skills under ~/.hermes/skills/<slug>/."
        ),
    )
