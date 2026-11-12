"""Rule text and helpers for the compaction summarizer prompt.

Epistemic stance: compression sheds qualifier words first ("probably", "I suspect", "unconfirmed"),
which either hardens an untested hypothesis into a fact the resumed agent acts on, or drops it and
loses the investigative thread. Asking for stance as an explicit labelled field rather than an inline aside
keeps it through compression (arXiv:2608.06953). Both strings are interpolated into the summarizer
prompt only (never the system prompt or the persisted handoff prefix), so the cached prefix is
untouched; f-string interpolation does not re-parse them, so braces in them are safe.
"""

UNVERIFIED_SECTION_HEADING = "## Unverified / Working Hypotheses"

EPISTEMIC_STANCE_RULE = (
    "Never restate an uncertain claim (suspicion, hypothesis, unverified inference) as fact: list "
    f'each under {UNVERIFIED_SECTION_HEADING} as "UNVERIFIED: <claim> — <basis>". '
)

UNVERIFIED_HYPOTHESES_SECTION = f"""

{UNVERIFIED_SECTION_HEADING}
[UNVERIFIED: <claim> — <basis so far>, one per line. If none, write "None."]"""


def _today_for_prompt() -> str:
    """Date-only (user tz) for temporal anchoring; "" when the clock fails. Cache-safe: the summary is outside the prefix."""
    try:
        # Date-only granularity matches system_prompt.py:337 (PR #20451) and the user's configured timezone
        # via hermes_agent.hermes_time.now(). The compaction summary is a mid-conversation message that is NOT part of
        # the cached prefix, so a date here never affects prompt-cache stability. Resolved defensively — a
        # clock failure must never block compaction.
        from hermes_agent.hermes_time import now as _hermes_now
        return _hermes_now().strftime("%Y-%m-%d")
    except Exception:  # pragma: no cover - clock resolution is best-effort
        return ""
