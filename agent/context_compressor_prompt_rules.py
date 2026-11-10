"""Rule text and helpers for the compaction summarizer prompt.

Epistemic stance: compression sheds qualifier words first ("probably", "I suspect", "unconfirmed"),
which either hardens an untested hypothesis into a fact the resumed agent acts on, or drops it and
loses the investigative thread. Asking for stance as an explicit labelled field rather than an inline aside
keeps it through compression (arXiv:2608.06953). Both strings are interpolated into the summarizer
prompt only (never the system prompt or the persisted handoff prefix), so the cached prefix is
untouched; f-string interpolation does not re-parse them, so braces in them are safe.
"""

import re
from typing import Any

UNVERIFIED_SECTION_HEADING = "## Unverified / Working Hypotheses"

#: Acceptable model spellings of the section. Anything matching is rewritten to the canonical heading so a
#: downstream reader (a validator, a resumed agent, a telemetry query) sees one string.
_HEADING_VARIANT_RE = re.compile(
    r"^#{2,3}[ \t]*(?:Unverified(?:[ \t]*[/&,][ \t]*Working)?(?:[ \t]+Hypotheses)?"
    r"|Unverified[ \t]+Claims|Working[ \t]+Hypotheses)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_ANY_H2_RE = re.compile(r"^##[ \t]+\S", re.MULTILINE)
_HYPOTHESIS_LINE_RE = re.compile(r"^[ \t]*(?:[-*][ \t]+)?UNVERIFIED[ \t]*:", re.IGNORECASE | re.MULTILINE)
_NONE_LINE_RE = re.compile(r"^[ \t]*(?:[-*][ \t]+)?None\.?[ \t]*$", re.IGNORECASE | re.MULTILINE)

EPISTEMIC_STANCE_RULE = (
    "Never restate an uncertain claim (suspicion, hypothesis, unverified inference) as fact: list "
    f'each under {UNVERIFIED_SECTION_HEADING} as "UNVERIFIED: <claim> — <basis>". '
    "Drop a hypothesis once a later turn verified or refuted it, and say which."
)

UNVERIFIED_HYPOTHESES_SECTION = f"""

{UNVERIFIED_SECTION_HEADING}
[UNVERIFIED: <claim> — <basis so far>, one per line. If none, write "None."]"""


def _today_for_prompt() -> str:
    """Date-only (user tz) for temporal anchoring; "" when the clock fails. Cache-safe: the summary is outside the prefix."""
    try:
        # Date-only granularity matches system_prompt.py:337 (PR #20451) and the user's configured timezone
        # via hermes_time.now(). The compaction summary is a mid-conversation message that is NOT part of
        # the cached prefix, so a date here never affects prompt-cache stability. Resolved defensively — a
        # clock failure must never block compaction.
        from hermes_time import now as _hermes_now
        return _hermes_now().strftime("%Y-%m-%d")
    except Exception:  # pragma: no cover - clock resolution is best-effort
        return ""


def normalize_stance_section(summary: str) -> tuple[str, dict[str, Any]]:
    """Canonicalize a variant stance heading and report the section's shape.

    The summarizer is asked for one heading and models spell it several ways ("## Unverified Hypotheses",
    "## Working Hypotheses"). Normalizing before anything reads the summary keeps a validator, a resumed
    agent, and a telemetry query agreeing on one string. The report carries counts only; the claims stay
    in the summary and are never copied into logs.

    ``present`` is False when the model dropped the section entirely, which is the failure this measures:
    the rule is a prompt instruction, and nothing else enforces it.
    """
    text = _HEADING_VARIANT_RE.sub(UNVERIFIED_SECTION_HEADING, summary or "")
    start = text.find(UNVERIFIED_SECTION_HEADING)
    if start < 0:
        return text, {"present": False, "hypothesis_lines": 0, "explicit_none": False}
    after = start + len(UNVERIFIED_SECTION_HEADING)
    next_heading = _ANY_H2_RE.search(text, after)
    section = text[after:next_heading.start() if next_heading else len(text)]
    return text, {
        "present": True,
        "hypothesis_lines": len(_HYPOTHESIS_LINE_RE.findall(section)),
        "explicit_none": bool(_NONE_LINE_RE.search(section)),
    }
