"""Token-free detection of user *reactions* to the agent.

Currently the only reaction is ``vibe`` — an expression of affection or
gratitude toward the agent (``ily``, ``<3``, ``love you``, ``good bot``, a heart
emoji, …). Detection is a curated regex/lexicon: **no model call, no tokens**.

This is the single source of truth shared by every surface — the CLI pet, the
TUI heart, and the desktop floating hearts all react off the same signal,
delivered via ``AIAgent.reaction_callback`` (wired per interactive host).

Generalized on purpose: :func:`detect_reaction` returns a reaction *kind*
string, so new kinds (other emoji reactions, etc.) can be added here without
touching any caller. We match affection specifically — not general positive
sentiment — so "this is great" does NOT fire, but "good bot" / "❤️" do.
"""

from __future__ import annotations

import re

#: The affection/gratitude reaction — the only kind today.
VIBE = "vibe"

# Curated affection lexicon. Kept deliberately narrow: gratitude + love aimed at
# the agent, heart emoji, and ``<3`` (but not the broken heart ``</3``).
_VIBE_RE = re.compile(
    "|".join(
        (
            r"\bgood\s*bot\b",
            r"\bi\s*(?:love|luv)\s*(?:you|u|ya)\b",
            r"\b(?:love|luv)\s*(?:you|u|ya)\b",
            r"\bily(?:sm)?\b",
            r"\bthank\s*(?:you|u)\b",
            r"\b(?:thanks|thx|tysm|ty)\b",
            r"<3+",  # <3, <33 … but not </3
            # Hearts + affection faces (❤ ♥ 🥰 😍 😘 💕 💖 💗 💞 💛 💜 💚 💙 💓 💘 💝 🩷).
            r"[\u2764\u2665"
            r"\U0001F970\U0001F60D\U0001F618"
            r"\U0001F495\U0001F496\U0001F497\U0001F49E"
            r"\U0001F49B\U0001F49C\U0001F49A\U0001F499"
            r"\U0001F493\U0001F498\U0001F49D\U0001FA77]",
        )
    ),
    re.IGNORECASE,
)


def detect_reaction(text: str | None) -> str | None:
    """Return the reaction kind for *text* (currently :data:`VIBE`), or ``None``.

    Pure, token-free, and safe to call on every user turn.
    """
    if not text:
        return None

    return VIBE if _VIBE_RE.search(text) else None
