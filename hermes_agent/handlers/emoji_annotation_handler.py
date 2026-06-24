"""
EmojiAnnotationHandler
======================

Annotates a message payload with a leading emoji that reflects the
message's *sentiment* or *category* tag.

Annotation rules
----------------
The handler inspects ``message["category"]`` (a lowercase string) and
prepends the matching emoji to ``message["text"]``.  If the category is
absent or unrecognised the message is returned unchanged.

Supported categories
~~~~~~~~~~~~~~~~~~~~
    positive  → 😊
    negative  → 😞
    question  → ❓
    warning   → ⚠️
    info      → ℹ️
    success   → ✅
    error     → ❌

The mapping is intentionally kept as a plain ``dict`` so callers can
supply a *custom* mapping via the constructor.
"""

from __future__ import annotations

from typing import Dict, MutableMapping, Optional

# ---------------------------------------------------------------------------
# Default category → emoji mapping
# ---------------------------------------------------------------------------

DEFAULT_EMOJI_MAP: Dict[str, str] = {
    "positive": "😊",
    "negative": "😞",
    "question": "❓",
    "warning": "⚠️",
    "info": "ℹ️",
    "success": "✅",
    "error": "❌",
}


class EmojiAnnotationHandler:
    """Prepend a category emoji to ``message["text"]``.

    Parameters
    ----------
    emoji_map:
        Optional override for the default category-to-emoji mapping.
        When provided it *replaces* (not merges with) the default map.
    separator:
        String inserted between the emoji and the original text.
        Defaults to a single space ``" "``.
    """

    def __init__(
        self,
        emoji_map: Optional[Dict[str, str]] = None,
        separator: str = " ",
    ) -> None:
        self._emoji_map: Dict[str, str] = (
            emoji_map if emoji_map is not None else dict(DEFAULT_EMOJI_MAP)
        )
        self._separator = separator

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle(self, message: MutableMapping[str, str]) -> MutableMapping[str, str]:
        """Annotate *message* in-place and return it.

        Parameters
        ----------
        message:
            A mutable mapping that **must** contain a ``"text"`` key.
            An optional ``"category"`` key drives emoji selection.

        Returns
        -------
        The same mapping object, potentially with ``"text"`` modified.

        Raises
        ------
        KeyError
            If ``"text"`` is missing from *message*.
        """
        if "text" not in message:
            raise KeyError("message must contain a 'text' key")

        category: str = message.get("category", "").strip().lower()
        emoji: Optional[str] = self._emoji_map.get(category)

        if emoji:
            message["text"] = f"{emoji}{self._separator}{message['text']}"

        return message

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def annotate(self, text: str, category: str) -> str:
        """Return *text* annotated with the emoji for *category*.

        Unlike :meth:`handle` this helper works on plain strings and does
        **not** mutate any mapping.  Returns *text* unchanged when
        *category* is unrecognised.
        """
        emoji = self._emoji_map.get(category.strip().lower())
        if emoji:
            return f"{emoji}{self._separator}{text}"
        return text

    @property
    def emoji_map(self) -> Dict[str, str]:
        """Read-only view of the active category → emoji mapping."""
        return dict(self._emoji_map)
