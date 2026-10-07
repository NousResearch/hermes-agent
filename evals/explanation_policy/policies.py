"""Explanation-policy matrix for the adaptive-explanation eval (#93382).

Each policy is a name -> spec mapping. A spec has:
  select: (Signals) -> Modality, the policy under test
  label:  human-readable description, used by report.py

`fixed_markdown` is the control arm from #93382's evaluation contract: one
concise answer, no modality selection. Every other policy picks a modality
from the catalog and renders it as Markdown too, so the arms differ only in
*what* gets explained -- never in transport. Nothing here touches the
response envelope, so the harness has no dependency on #7191/#61095/#74334.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict

# Reference model for PUBLISHED scorecards. The runner does not pin this: by
# default it lets the configured auxiliary route answer, like the other evals/
# harnesses, and records whatever replied on each scorecard row. What the
# evaluation contract actually requires is that the model be held constant
# across policies within a run -- "the same model and source content" -- which
# holds either way. Pass --model to force this one.
EVAL_MODEL = "anthropic/claude-opus-5"


# Only the signal values some policy actually branches on are defined. #93382
# lists a wider vocabulary; adding a member no policy reads would be surface
# with no consumer, and the scorecard could not say anything about it.
class Intent(str, Enum):
    ANSWER = "answer"
    DECIDE = "decide"
    LEARN = "learn"
    PRACTICE = "practice"
    OPERATE = "operate"


class Structure(str, Enum):
    COMPARISON = "comparison"
    CAUSAL = "causal"


class Modality(str, Enum):
    CONCISE = "concise"
    COMPARISON_TABLE = "comparison_table"
    CAUSAL_CHAIN = "causal_chain"
    WORKED_EXAMPLE = "worked_example"
    PREDICTION_FIRST = "prediction_first"
    RETRIEVAL_CHECK = "retrieval_check"


@dataclass(frozen=True)
class Signals:
    """The bounded, observable inputs a policy may read (#93382).

    `knowledge` is user-declared or evidenced in the current task, never
    inferred into a durable profile: it is re-supplied per run, stored never.
    """

    intent: Intent
    structure: Structure
    risk: str = "low"
    knowledge: str = "unknown"  # unknown | novice | practitioner


def _always_concise(_: Signals) -> Modality:
    return Modality.CONCISE


def _smallest_useful(s: Signals) -> Modality:
    """Smallest modality that fits intent + structure.

    A lookup rather than a model call on purpose: the policy under test has to
    be reproducible across runs, or a scorecard difference cannot be
    attributed to it. The `practitioner` branch is the expertise-reversal
    guard -- a reader who declares prior knowledge never gets a worked example.
    """
    if s.intent in (Intent.ANSWER, Intent.OPERATE):
        return Modality.CONCISE
    if s.intent is Intent.DECIDE:
        if s.structure is Structure.COMPARISON:
            return Modality.COMPARISON_TABLE
        return Modality.CAUSAL_CHAIN
    if s.intent in (Intent.LEARN, Intent.PRACTICE):
        if s.knowledge == "practitioner":
            return Modality.RETRIEVAL_CHECK
        if s.structure is Structure.COMPARISON:
            return Modality.COMPARISON_TABLE
        return Modality.WORKED_EXAMPLE
    return Modality.CONCISE


def _prediction_first(s: Signals) -> Modality:
    """Retrieval-practice arm: ask before revealing, else smallest useful.

    Deliberately does NOT consult `knowledge`. Expertise reversal is a result
    about redundant *guidance* -- a worked example a knowledgeable reader has
    to wade through -- not about retrieval, so `_smallest_useful`'s guard does
    not apply here. Practitioners losing on this arm would be a finding, not a
    bug to patch away.
    """
    if s.intent in (Intent.LEARN, Intent.PRACTICE):
        return Modality.PREDICTION_FIRST
    return _smallest_useful(s)


POLICIES: Dict[str, Dict[str, object]] = {
    "fixed_markdown": {
        "select": _always_concise,
        "label": "control: one concise Markdown answer",
    },
    "smallest_useful": {
        "select": _smallest_useful,
        "label": "smallest modality that fits intent + structure",
    },
    "prediction_first": {
        "select": _prediction_first,
        "label": "retrieval practice before reveal, then smallest useful",
    },
}

# How each modality is rendered. Markdown only: no artifact envelope, no
# renderer, no client-side component. An unsupported modality falls back to
# CONCISE, which is the degradation path #93382 asks for.
RENDER_INSTRUCTIONS: Dict[Modality, str] = {
    Modality.CONCISE:
        "At most 120 words of plain prose. No headings, no table, no exercise.",
    Modality.COMPARISON_TABLE:
        "A short Markdown table on the axes that actually discriminate the "
        "concepts, then one sentence naming the axis that matters most.",
    Modality.CAUSAL_CHAIN:
        "A cause -> effect chain of at most 5 links, then one sentence on "
        "which link is load-bearing.",
    Modality.WORKED_EXAMPLE:
        "One fully worked concrete example showing the distinction in action, "
        "then one sentence generalizing from it.",
    Modality.PREDICTION_FIRST:
        "Pose ONE short prediction question and stop. Then, after a '---' "
        "separator, reveal the answer with corrective feedback aimed at the "
        "likely wrong prediction.",
    Modality.RETRIEVAL_CHECK:
        "At most 80 words, then one short retrieval question the reader can "
        "use to self-check.",
}


def select(policy: str, signals: Signals) -> Modality:
    """Apply a named policy, failing closed to CONCISE on an unknown name."""
    spec = POLICIES.get(policy)
    if spec is None:
        return Modality.CONCISE
    fn: Callable[[Signals], Modality] = spec["select"]  # type: ignore[assignment]
    return fn(signals)
