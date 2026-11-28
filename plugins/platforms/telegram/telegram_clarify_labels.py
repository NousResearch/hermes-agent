"""Presentation-only captions for Telegram clarify keyboards; original choices stay in the body."""

from __future__ import annotations

import html
import unicodedata
from typing import Any

from wcwidth import iter_graphemes, wcswidth

AUTO_COLUMNS = 28
TEXT_COLUMNS = 60
CAPTION_CHARS = 64


def choice_caption(choice: Any, index: int, mode: Any) -> str:
    """Use numbers by default, text on request, or text only for short choices in auto mode.

    Columns approximate mobile width, not Telegram's proportional font. Keep both a column and
    code-point budget and truncate at grapheme boundaries so emoji/combining sequences stay whole.
    """
    number = str(index + 1)
    if mode not in ("text", "auto"):
        return number
    # A caption is one line. Keep joiners and emoji tags, not bidi/control formatting.
    clean = "".join(" " if unicodedata.category(c) in ("Cc", "Cs", "Cf")
                    and c not in ("\u200c", "\u200d") and not "\U000e0020" <= c <= "\U000e007f" else c
                    for c in str(choice))
    label = " ".join(clean.split())
    width = wcswidth(label)
    if width <= 0 or (mode == "auto" and width > AUTO_COLUMNS):
        return number
    if width <= TEXT_COLUMNS and len(label) <= CAPTION_CHARS:
        return label
    caption = ""
    for cluster in iter_graphemes(label):
        candidate = caption + cluster
        if len(candidate) >= CAPTION_CHARS or wcswidth(candidate) >= TEXT_COLUMNS:
            break
        caption = candidate
    return caption.rstrip() + "…" if wcswidth(caption) > 0 else number


def clarify_prompt(adapter: Any, question: str, choices: list | None, clarify_id: str) -> tuple:
    """Build the original full numbered body and a caption-configurable keyboard."""
    from plugins.platforms.telegram import adapter as tg
    from plugins.platforms.telegram.telegram_clarify_multi import multi_prompt_extras

    text = f"❓ {html.escape(question)}"
    keyboard = None
    if choices:
        # Mobile clients can truncate even bounded captions: the full numbered list is authoritative.
        text += "\n\n" + "\n".join(f"{i + 1}. {html.escape(str(c))}" for i, c in enumerate(choices))
        if multi := multi_prompt_extras(adapter, clarify_id, len(choices)):
            text, keyboard = text + multi[0], multi[1]
        else:
            mode = adapter.config.extra.get("clarify_button_labels", "numbers")
            rows = [[tg.InlineKeyboardButton(choice_caption(c, i, mode), callback_data=f"cl:{clarify_id}:{i}")]
                    for i, c in enumerate(choices)]
            rows.append([tg.InlineKeyboardButton(tg.t("platform.telegram.prompt.other"), callback_data=f"cl:{clarify_id}:other")])
            keyboard = tg.InlineKeyboardMarkup(rows)
    return text, keyboard
