#!/usr/bin/env python3
"""``/upskill`` — sweep the current session's work and propose reusable skills.

Where ``/learn`` is open-ended (the user names a source to distill),
``/upskill`` is the automatic inverse: it reviews what the agent ACTUALLY did
in this session (its tool-call history, workflows, repeated procedures) and
proposes candidate reusable skills — including seemingly trivial ones the user
might not think to save (e.g. "connect to the AP7632i over serial/SSH", "run
the daily typecheck+tests"). The user confirms each candidate, and approved
ones are saved via ``skill_manage``.

Like ``/learn``, there is no separate distillation engine and no model-tool
footprint: the agent does the sweep with the tools it already has
(``session_search``, its own in-context history, ``skills_list`` /
``skill_view`` for dedupe, ``skill_manage`` to save). This module builds the
ONE prompt that drives that sweep, and every surface (CLI ``/upskill``,
gateway ``/upskill``) feeds the result to the agent as a normal turn.
"""

from __future__ import annotations

# Reuse the same HARDLINE authoring + source-hygiene standards as /learn, so a
# skill saved by /upskill is identical in quality to one saved by /learn.
# Import from agent.skill_standards (the shared source of truth), NOT from
# learn_prompt — a refactor of /learn's prompt module must not break /upskill.
from agent.skill_standards import (
    AUTHORING_STANDARDS,
    SOURCE_HYGIENE,
)


def build_upskill_prompt(scope_hint: str = "") -> str:
    """Build the agent prompt for an automatic ``/upskill`` sweep.

    Args:
        scope_hint: optional user emphasis to narrow the sweep (e.g.
            "focus on the WiNG console workflow", "skip the git stuff"). May
            be empty for a full-session sweep.

    Returns:
        A complete instruction the agent runs as a normal turn. The agent
        surveys the session, proposes candidates, waits for approval, then
        saves approved skills via ``skill_manage``.
    """
    hint = (scope_hint or "").strip()

    scope_line = ""
    if hint:
        scope_line = (
            f"\nUSER SCOPE EMPHASIS: {hint}\n"
            "Treat this as load-bearing — focus the sweep on it, and only it. "
            "Do not propose candidates outside this scope.\n"
        )

    return (
        "[/upskill] The user wants you to review this session's work and turn "
        "the genuinely reusable bits into persistent skills. This is NOT an "
        "open-ended 'learn from X' — you are sweeping what actually happened "
        "in the current session and PROPOSING candidate skills for approval.\n\n"
        f"{scope_line}"
        "Do this:\n"
        "1. Survey the current session: your in-context history plus "
        "`session_search` for this session's messages and tool calls. Identify "
        "repeated procedures, non-trivial workflows, and reusable techniques — "
        "including 'small' or 'trivial' ones that are still multi-step and "
        "would otherwise be rediscovered each session (e.g. a device "
        "connect-and-configure sequence, a verification loop, a build-push "
        "ritual).\n"
        "2. Cluster the findings into candidate skills. For each candidate, "
        "judge: is it (a) a genuinely reusable procedure with stable steps, not "
        "a one-off task; (b) specific enough to be a real skill rather than "
        "'general agent behaviour'; (c) likely to recur? Drop candidates that "
        "fail any of these — do NOT propose noise.\n"
        "3. DEDUPE against skills that already exist (`skills_list`, then "
        "`skill_view` any plausible match). If a candidate overlaps an existing "
        "skill, do not propose a new one — either skip it or propose it as an "
        "extension of the existing skill (say so explicitly). Never propose a "
        "near-duplicate.\n"
        "4. PROPOSE the candidates to the user as a concise numbered list, "
        "each with: a proposed skill name, a one-line description, and one "
        "sentence on why it's reusable. Do NOT save anything yet. Ask the user "
        "to approve (all / by number / none).\n"
        "5. Only after approval, save each approved candidate with "
        "`skill_manage` action=\"create\" (or patch/extend an existing skill if "
        "that is what was approved), obeying the authoring standards "
        "appended below. Pick a sensible category. If a candidate needs a "
        "non-trivial script, add it under the skill's `scripts/` with "
        "`skill_manage` write_file and reference it by relative path.\n"
        "6. Report back: how many candidates you found, which were approved and "
        "saved (names), which were skipped/deduped and why.\n\n"
        f"{SOURCE_HYGIENE}\n\n"
        f"{AUTHORING_STANDARDS}\n\n"
        "If, after the sweep, there is genuinely nothing worth saving, say so "
        "clearly rather than inventing a low-value skill.\n"
    )
