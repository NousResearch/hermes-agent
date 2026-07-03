#!/usr/bin/env python3
"""retro_envelope.py — the hindsight oneshot's invocation contract (single source).

After a SUCCESSFUL full-route run, one oneshot judges "was there a more optimal path?"
against the consolidated journey record (journey.py's COMPACT render). The judgment is
ADVISORY ONLY: it can never un-succeed the run, and it triggers no execution — it is
ink on the record, not control flow. Its claims are validated and tier-stamped by CODE
(journey.validate_hindsight / journey.stamp_tiers): every avoidable-branch claim must
cite a node key and an enabling-evidence fp exactly as rendered, and pure positional
logic then classifies it genuinely-avoidable / blind-spot / honest-exploration.

The dual-channel rule (artifact + echoed JSON) matches every other oneshot contract
here: a timeout can kill stdout after the file write already landed.

Stdlib-only; no env reads, no imports.
"""

SCHEMA_EXAMPLE = (
    '{"schema": 1, "optimality": "near-optimal"|"acceptable"|"sloppy", '
    '"hindsight_path": [{"method": "<step of the shorter path, if one existed>", '
    '"why_available_earlier": "<one line>"}], '
    '"avoidable_branches": [{"node": "<S-key where a better option existed>", '
    '"option": "<the better method>", '
    '"enabling_evidence_fp": "<fp of the evidence that already justified it, exactly '
    'as rendered>", "why": "<one line>"}], '
    '"unavoidable_branches": [{"node": "<S-key>", "why_necessary": "<one line: why '
    'this dead end had to be run to be learned>"}], '
    '"promoted_learnings": ["<OPTIONAL: a self-contained, reusable fact a FUTURE run '
    'on a similar intent should start with>"]}'
)


def hindsight_prompt(journey_text, out_path):
    """The post-success retrospective oneshot: judge the recorded decisions ONLY from
    the record — the journey renders in decision order, so everything above a node is
    exactly what the system knew there."""
    return (
        "You are a HINDSIGHT JUDGE reviewing a SUCCESSFUL autonomous run. The task is "
        "already solved — do NOT redo it, execute anything, or modify any state. Judge "
        "ONE question: knowing only what the record shows, was there a more optimal "
        "path to the same success?\n\n"
        "THE RECORD (a chain of decision nodes; each node shows the evidence that was "
        "NEW there, the options as recorded at the time, and what was chosen — so "
        "everything rendered ABOVE a node is exactly what was known at it):\n\n"
        f"{journey_text.rstrip()}\n\n"
        "Emit ONE JSON object matching this schema exactly:\n"
        f"{SCHEMA_EXAMPLE}\n"
        "Rules:\n"
        "- CITE OR IT DIDN'T HAPPEN: every branch claim names a node key (S0, S1, ...) "
        "from the record, and every avoidable-branch claim names the enabling-evidence "
        "fp EXACTLY as rendered (the [kind·fp ...] tags). Uncited claims are "
        "rejected.\n"
        '- When the better option WAS on the record, copy its method label into "option" '
        "EXACTLY as rendered — a paraphrased label reads as an option nobody recorded "
        "and downgrades your claim.\n"
        '- A branch is "avoidable" ONLY if the evidence justifying the better option '
        "was already rendered at or above the node where the choice was made. A dead "
        "end whose disproof REQUIRED running it is the system working as designed — "
        'list it under "unavoidable_branches", never as sloppiness.\n'
        '- "optimality" grades the whole run: near-optimal (no avoidable branches), '
        "acceptable (minor avoidable cost), sloppy (the success path was reachable "
        "much earlier from evidence already in hand).\n"
        '- "hindsight_path" is the shorter route IF one existed — omit or leave empty '
        "when the taken path was already near-optimal.\n"
        '- When a shorter path existed, also restate it as one self-contained '
        '"promoted_learning" naming the route and the evidence that justified it.\n'
        '- "promoted_learnings" (optional, max 5) are for FUTURE runs: self-contained '
        "facts someone with no other context could act on — name the systems involved "
        "and the mechanism, not just the outcome.\n"
        "- Judge only from the record above. Never invent evidence, options, or nodes.\n"
        f"Write the JSON to {out_path} AND echo the same JSON as your final message."
    )


def retry_suffix(violations):
    """Appended when a previous emission failed validation (same shape as the
    task-decomposer envelope's retry channel)."""
    return ("\n\nYour previous hindsight FAILED validation:\n"
            + "\n".join(f"- {v}" for v in violations)
            + "\nOutput ONLY the corrected JSON object (write the file AND echo it), "
              "nothing else.")
