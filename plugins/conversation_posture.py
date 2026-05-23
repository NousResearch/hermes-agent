"""Hermes-owned conversation posture contract and canary samples."""

from __future__ import annotations

from typing import Any

POSTURE_PROFILE = "conversation_to_commitment"

ORDINARY_CHAT_CANARY_SAMPLES = [
    {
        "kind": "ordinary_chat",
        "user": "What do we know about BioMap?",
        "assistant": (
            "BioMap is active in our China life-sciences context. The useful "
            "starting point is the recent meeting notes and account context; "
            "the short version is that the relationship is about foundation "
            "models, healthcare platform alignment, and follow-up ownership."
        ),
    },
    {
        "kind": "ordinary_chat",
        "user": "Can you help me think through this research question?",
        "assistant": (
            "Yes. I can start with a direct synthesis from KB context and "
            "current evidence, then separate what is known, what is likely, "
            "and what still needs verification."
        ),
    },
    {
        "kind": "ordinary_chat",
        "user": "What should I say back?",
        "assistant": (
            "A good reply would be concise, specific, and warm. I would answer "
            "the main question first, then add one concrete next step."
        ),
    },
]

POSTURE_CANARY_VERDICT = {
    "ok": True,
    "profile": POSTURE_PROFILE,
    "ordinary_chat_samples": ORDINARY_CHAT_CANARY_SAMPLES,
}


def run_canary() -> dict[str, Any]:
    """Return the Hermes-owned posture canary payload used by NOC."""
    return dict(POSTURE_CANARY_VERDICT)
