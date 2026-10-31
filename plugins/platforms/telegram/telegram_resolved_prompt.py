"""Resolved-prompt edits that keep the request visible, kept out of the adapter facade.

Replacing an approval or slash-confirm prompt with only the decision ("✅ Approved once by Ana")
erases what was approved, so a shared chat loses its record. The edit keeps the prompt's own
rendering (``Message.text_html`` rebuilds it from the entities, code block included) and appends
the decision.
"""

from __future__ import annotations

import contextlib
import html
from collections.abc import Awaitable, Callable
from typing import Any


async def edit_keeping_prompt(query: Any, decision: str, fallback: Callable[[Any, str], Awaitable[None]]) -> None:
    """Append *decision* to the prompt and drop its buttons; *fallback* (decision only) runs when the
    prompt has no readable body or the appended line pushes it past Telegram's length cap."""
    body = getattr(query.message, "text_html", None) if query.message else None
    if isinstance(body, str) and body:
        with contextlib.suppress(Exception):
            await query.edit_message_text(text=f"{body}\n\n— {html.escape(decision)}", parse_mode="HTML", reply_markup=None)
            return
    await fallback(query, decision)
