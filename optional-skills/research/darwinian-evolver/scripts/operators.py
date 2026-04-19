"""LLM-driven mutation and crossover operators.

Each operator takes one or two parents and returns a child genome plus a
stable ``prompt_hash`` identifying the exact operator prompt used — the
hash is stored in the lineage table so replay can verify operator
determinism.

The operator library is intentionally small and composable. The Exp3
bandit in ``algorithms.Exp3Bandit`` decides which to invoke; every
operator here is uniform in signature so the bandit can treat them
symmetrically.

References
----------
Meyerson et al. (2023)   "Language Model Crossover: Variation through
                          Few-Shot Prompting." arXiv:2302.12170.
Fernando et al. (2023)   "Promptbreeder: Self-Referential
                          Self-Improvement via Prompt Evolution."
                          arXiv:2309.16797.
Agrawal et al. (2025)    "GEPA: Reflective Prompt Evolution."
                          arXiv:2507.19457.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from llm import LLMClient


# ---------------------------------------------------------------------------
# Operator prompts
# ---------------------------------------------------------------------------

_PARAPHRASE_SYS = (
    "You rewrite text to preserve meaning while varying wording. "
    "Return ONLY the rewritten text — no explanations, no quotes."
)

_STRUCTURAL_EDIT_SYS = (
    "You edit text by making one structural change: add or remove a clause, "
    "reorder for emphasis, or merge two sentences into one. Preserve intent. "
    "Return ONLY the edited text."
)

_COT_INJECT_SYS = (
    "You modify a prompt so that, when followed, the answer uses step-by-step "
    "reasoning. Keep the user's goal intact. Return ONLY the revised prompt."
)

_NOVELTY_SYS = (
    "You rewrite the given text to be as different as possible along the "
    "specified axis while still accomplishing the original goal. Return ONLY "
    "the rewritten text."
)

_META_MUTATOR_SYS = (
    "You are a mutation-strategy designer. Given the current mutation prompt "
    "and recent fitness outcomes, propose a new mutation instruction that "
    "would produce better offspring next generation. Return ONLY the new "
    "mutation instruction."
)

_CRITIQUE_EDIT_SYS = (
    "You are an iterative editor. You will be given a candidate and a short "
    "description of how it failed. First, silently identify one concrete fix. "
    "Then output ONLY the revised candidate — nothing else."
)

_SEMANTIC_CROSSOVER_SYS = (
    "You combine two text variants into one that inherits the strongest "
    "properties of each. Preserve the shared intent. Return ONLY the combined "
    "text — no explanation."
)


def _hash_prompt(*parts: str) -> str:
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Operator records
# ---------------------------------------------------------------------------


@dataclass
class OperatorResult:
    child: str
    prompt_hash: str
    operator: str
    parents: list[str]


# The operator interface: (client, parents, seed, **kwargs) -> OperatorResult
OperatorFn = Callable[..., Awaitable[OperatorResult]]


# ---------------------------------------------------------------------------
# Mutation operators (single parent)
# ---------------------------------------------------------------------------


async def paraphrase(
    client: LLMClient,
    parent: str,
    *,
    seed: Optional[int] = None,
) -> OperatorResult:
    child = await client.complete(
        _PARAPHRASE_SYS, parent, seed=seed, temperature=0.8, operator="paraphrase",
    )
    return OperatorResult(
        child=child or parent,
        prompt_hash=_hash_prompt(_PARAPHRASE_SYS, parent),
        operator="paraphrase",
        parents=[parent],
    )


async def structural_edit(
    client: LLMClient,
    parent: str,
    *,
    seed: Optional[int] = None,
) -> OperatorResult:
    child = await client.complete(
        _STRUCTURAL_EDIT_SYS, parent, seed=seed, temperature=0.85, operator="structural_edit",
    )
    return OperatorResult(
        child=child or parent,
        prompt_hash=_hash_prompt(_STRUCTURAL_EDIT_SYS, parent),
        operator="structural_edit",
        parents=[parent],
    )


async def cot_inject(
    client: LLMClient,
    parent: str,
    *,
    seed: Optional[int] = None,
) -> OperatorResult:
    child = await client.complete(
        _COT_INJECT_SYS, parent, seed=seed, temperature=0.6, operator="cot_inject",
    )
    return OperatorResult(
        child=child or parent,
        prompt_hash=_hash_prompt(_COT_INJECT_SYS, parent),
        operator="cot_inject",
        parents=[parent],
    )


async def novelty_seeking(
    client: LLMClient,
    parent: str,
    *,
    axis: str = "sentence structure",
    seed: Optional[int] = None,
) -> OperatorResult:
    user = f"Axis: {axis}\n\nOriginal:\n{parent}"
    child = await client.complete(
        _NOVELTY_SYS, user, seed=seed, temperature=1.0, operator="novelty",
    )
    return OperatorResult(
        child=child or parent,
        prompt_hash=_hash_prompt(_NOVELTY_SYS, axis, parent),
        operator="novelty",
        parents=[parent],
    )


async def critique_then_edit(
    client: LLMClient,
    parent: str,
    *,
    failure_note: str = "",
    seed: Optional[int] = None,
) -> OperatorResult:
    """GEPA-lite: one-shot critique followed by a targeted edit.

    ``failure_note`` should summarise the observed eval failure; leave it
    empty for exploratory mutation. Chain-of-thought is *internal* to the
    LLM — we ask it to identify the fix silently and only emit the
    revised candidate, so the downstream fitness evaluator sees a clean
    artifact.
    """
    user = f"Candidate:\n{parent}\n\nObserved failure:\n{failure_note or '(none supplied — do exploratory edit)'}"
    child = await client.complete(
        _CRITIQUE_EDIT_SYS, user, seed=seed, temperature=0.6, operator="critique_then_edit",
    )
    return OperatorResult(
        child=child or parent,
        prompt_hash=_hash_prompt(_CRITIQUE_EDIT_SYS, failure_note, parent),
        operator="critique_then_edit",
        parents=[parent],
    )


# ---------------------------------------------------------------------------
# Meta-mutation (PromptBreeder-style)
# ---------------------------------------------------------------------------


async def meta_mutate_operator_prompt(
    client: LLMClient,
    current_prompt: str,
    *,
    recent_deltas: list[float],
    seed: Optional[int] = None,
) -> str:
    """Evolve the mutation prompt itself.

    Returns the proposed new mutation-prompt text. The caller decides
    whether to adopt it (e.g., by running a shadow mini-tournament).
    """
    summary = ", ".join(f"{d:+.3f}" for d in recent_deltas[-8:]) or "no data"
    user = (
        f"Current mutation prompt:\n{current_prompt}\n\n"
        f"Recent offspring fitness deltas: {summary}\n\n"
        "Propose a revised mutation prompt that should yield better offspring."
    )
    out = await client.complete(
        _META_MUTATOR_SYS, user, seed=seed, temperature=0.7, operator="meta_mutator",
    )
    return out or current_prompt


# ---------------------------------------------------------------------------
# Crossover operators (two parents)
# ---------------------------------------------------------------------------


async def semantic_crossover(
    client: LLMClient,
    parent_a: str,
    parent_b: str,
    *,
    seed: Optional[int] = None,
) -> OperatorResult:
    user = f"Variant A:\n{parent_a}\n\nVariant B:\n{parent_b}"
    child = await client.complete(
        _SEMANTIC_CROSSOVER_SYS, user, seed=seed, temperature=0.7, operator="semantic_crossover",
    )
    return OperatorResult(
        child=child or parent_a,
        prompt_hash=_hash_prompt(_SEMANTIC_CROSSOVER_SYS, parent_a, parent_b),
        operator="semantic_crossover",
        parents=[parent_a, parent_b],
    )


def segment_splice(parent_a: str, parent_b: str, *, rng) -> OperatorResult:
    """Sentence-level cut-and-paste. Does not call the LLM.

    Splits each parent on ``". "``; picks a random cut in A and a random
    suffix from B. Cheap O(1) operator that often produces coherent
    output for prompts but can create grammatical clashes — useful
    mostly as a zero-cost diversity injection.
    """
    sents_a = parent_a.split(". ")
    sents_b = parent_b.split(". ")
    if len(sents_a) < 2 or len(sents_b) < 2:
        return OperatorResult(
            child=parent_a,
            prompt_hash=_hash_prompt("segment_splice", parent_a, parent_b),
            operator="segment_splice",
            parents=[parent_a, parent_b],
        )
    cut_a = rng.randint(1, len(sents_a) - 1)
    cut_b = rng.randint(0, len(sents_b) - 1)
    head = ". ".join(sents_a[:cut_a]).rstrip(".")
    tail = ". ".join(sents_b[cut_b:])
    child = f"{head}. {tail}".strip()
    return OperatorResult(
        child=child,
        prompt_hash=_hash_prompt("segment_splice", parent_a, parent_b),
        operator="segment_splice",
        parents=[parent_a, parent_b],
    )


# ---------------------------------------------------------------------------
# Operator registry
# ---------------------------------------------------------------------------


MUTATION_OPERATORS: dict[str, OperatorFn] = {
    "paraphrase":          paraphrase,
    "structural_edit":     structural_edit,
    "cot_inject":          cot_inject,
    "novelty":             novelty_seeking,
    "critique_then_edit":  critique_then_edit,
}


CROSSOVER_OPERATORS: dict[str, OperatorFn] = {
    "semantic_crossover":  semantic_crossover,
}
