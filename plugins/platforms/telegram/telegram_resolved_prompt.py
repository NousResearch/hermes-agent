"""Resolved-prompt edits that keep the request visible, kept out of the adapter facade.

Replacing an approval or slash-confirm prompt with only the decision ("✅ Approved once by Ana")
erases what was approved, so a shared chat loses its record. The edit keeps the prompt's own
rendering (``Message.text_html`` rebuilds it from the entities, code block included), drops the
"if you don't answer within N minutes" line that no longer applies, and appends the decision.
"""

from __future__ import annotations

import contextlib
import html
from typing import Any


async def edit_keeping_prompt(adapter: Any, query: Any, decision: str) -> None:
    """Append *decision* to the prompt and drop its buttons; the adapter's decision-only edit runs when
    the prompt has no readable body or the appended line pushes it past Telegram's length cap."""
    body = getattr(query.message, "text_html", None) if query.message else None
    if isinstance(body, str) and body:
        body = body.replace(adapter._ea_deadline_line(), "")  # same escaping as text_html
        with contextlib.suppress(Exception):
            await query.edit_message_text(text=f"{body}\n\n— {html.escape(decision)}", parse_mode="HTML", reply_markup=None)
            return
    await adapter._edit_md_quiet(query, decision)
