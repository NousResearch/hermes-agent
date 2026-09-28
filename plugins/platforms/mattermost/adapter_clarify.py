"""Pure helpers for the clarify-picker card layout on Mattermost.

Lives in a sibling file (per ``AGENTS.md`` god-file split rule) so the layout
can be unit-tested without spinning up the adapter. The mattermost adapter
imports ``format_clarify_picker_body`` from here.

The picker protocol (resolve handler + seeded reactions) lives in
``adapter.py``; only display formatting lives here.
"""
from typing import List, Sequence


def numeric_labels_for(choices: Sequence[str]) -> List[str]:
    """Return ``["1", "2", ...]`` parallel to ``choices``.

    Index-based, not name-based: ``keycap_ten`` → "10", not "ten". The numeric
    label is what the user types to reply via text (the mattermost resolve path
    also accepts ``"1"``, ``"10"`` etc. via the picker regex).
    """
    return [str(i + 1) for i in range(len(choices))]


def _format_options_row(choices: Sequence[str], responses: Sequence[str]) -> str:
    """One row of ``:one: foo  :two: bar`` joined with two spaces."""
    return "  ".join(
        f":{emoji}: {label}"
        for emoji, label in zip(choices, responses)
    )


def _format_text_fallback(
    question: str,
    choices: Sequence[str],
    responses: Sequence[str],
) -> str:
    """Legacy numbered list used when ``send_clarify`` can't render the picker.

    Preserved as a fallback for the >12-choice cap or any malformed input —
    matches the shape the adapter previously rendered inline.
    """
    lines = [f"❓ {question}", ""]
    for i, label in enumerate(responses or choices or [], 1):
        lines.append(f"{i}. {label}")
    return "\n".join(lines)


def format_clarify_picker_body(
    question: str,
    choices: Sequence[str],
    responses: Sequence[str],
    *,
    include_typeable_hint: bool = True,
) -> str:
    """Build the bot's clarify-picker card body.

    Layout (replaces the legacy plain inline list — see the plan at
    ``.hermes/plans/2026-09-28-mattermost-clarify-emoji-reactions.md``)::

        ❓ <question>

        :one: choice A   :two: choice B   :three: choice C

        (Tap an emoji above to answer, or reply with 1 / 2 / 3.)

    ``choices`` is the mattermost emoji short-name list (e.g. ``["one", "two",
    "three"]``); ``responses`` is the human labels in the same order. Falls
    back to the legacy numbered list when the input is empty or mismatched
    — preserves the existing ``send_clarify`` fallback contract.
    """
    if (
        not choices
        or not responses
        or len(choices) != len(responses)
    ):
        return _format_text_fallback(question, choices, responses)

    options_row = _format_options_row(choices, responses)
    if include_typeable_hint and len(choices) >= 2:
        # Single-choice prompts are unambiguous (no enumeration needed); skip
        # the typeable hint line to keep the card tight.
        nums = " / ".join(numeric_labels_for(choices))
        hint = f"(Tap an emoji above to answer, or reply with {nums}.)"
        return f"❓ {question}\n\n{options_row}\n\n{hint}"
    return f"❓ {question}\n\n{options_row}"
