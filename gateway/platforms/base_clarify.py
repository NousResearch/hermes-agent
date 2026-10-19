"""Plain-text rendering of a clarify prompt (split from :mod:`gateway.platforms.base`).

``BasePlatformAdapter.send_clarify`` sends this text on surfaces without a native card, and the
gateway's origin relay (``TurnRunner._relay_clarify_question``) posts the same text to the chat
that ORIGINATED a session driven elsewhere, so both views of one question stay identical — the
relay is only useful if the topic shows what the card actually offers (#103209).

No imports from ``gateway.platforms.base`` or ``gateway.run`` — both import this module.
"""
from __future__ import annotations

from agent.i18n import t


def format_clarify_prompt(question: str, choices: list[str] | None,
                          multi_select: bool = False) -> str:
    """``❓ question``, the numbered choices, then the reply hint (or just the question)."""
    question_line = t("gateway.clarify.question", question=question)
    if not choices:
        return question_line
    hint = t("gateway.clarify.hint_multi") if multi_select else t("gateway.clarify.hint_single")
    numbered = [t("gateway.clarify.option_line", index=i, choice=choice)
                for i, choice in enumerate(choices, start=1)]
    return "\n".join([question_line, "", *numbered, "", hint])
