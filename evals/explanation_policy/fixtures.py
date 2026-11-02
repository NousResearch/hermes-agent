"""Concept-comparison tasks for the explanation-policy eval.

The first slice is deliberately one task family: two concepts a reader
routinely conflates, the discriminators a correct explanation has to carry,
comprehension items answerable from a good explanation, and one transfer item
that a reader cannot answer by restating the explanation back.

`synthetic_task()` returns a task whose scoring is mechanically checkable, so
a smoke test can exercise the whole pipeline without spending an LLM call.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class Task:
    key: str
    question: str
    concepts: Tuple[str, str]
    discriminators: List[str]
    comprehension: List[str]
    transfer: str
    transfer_gold: str

    @property
    def gold(self) -> str:
        return "\n".join(f"- {d}" for d in self.discriminators)


TASKS: List[Task] = [
    Task(
        key="value_vs_reference",
        question=(
            "What is the difference between passing a payload by value and "
            "passing it by reference in a response protocol?"
        ),
        concepts=("by value", "by reference"),
        discriminators=[
            "by value embeds the payload inside the response; by reference "
            "sends an identifier the client has to resolve separately",
            "by reference introduces a lifetime question the protocol must "
            "answer: how long the referenced artifact stays resolvable",
            "by value makes a response self-contained, so replaying stored "
            "history reproduces exactly what the user originally saw",
            "by reference keeps responses small and lets one artifact be "
            "shared across responses or updated after it was sent",
        ],
        comprehension=[
            "Which of the two makes a response self-contained, and why?",
            "What new question does passing by reference force the protocol "
            "to answer that passing by value does not?",
        ],
        transfer=(
            "A client replays a six-month-old conversation from its own log to "
            "re-render it. Which of the two approaches is more likely to render "
            "incorrectly, and what exactly fails?"
        ),
        transfer_gold=(
            "By reference. The identifier may no longer resolve -- the artifact "
            "can have expired, been deleted, or been mutated since it was sent -- "
            "so the replay either fails to render or renders content that "
            "differs from what the user originally saw. By value is unaffected, "
            "because the payload travels inside the stored response."
        ),
    ),
]


def get_task(key: str) -> Task:
    """Look up a task by key. `synthetic` is always available.

    Exposing the synthetic task through the same lookup is what makes
    `--task synthetic` a real end-to-end smoke run: the whole
    explain -> read -> judge -> scorecard chain against a live provider, for
    a few hundred tokens instead of a few thousand.
    """
    if key == "synthetic":
        return synthetic_task()
    for t in TASKS:
        if t.key == key:
            return t
    known = [t.key for t in TASKS] + ["synthetic"]
    raise KeyError(f"unknown task: {key!r} (have: {known})")


def synthetic_task() -> Task:
    """Deterministic task with literal, greppable gold for smoke tests.

    Every gold string appears verbatim in the discriminators, so a smoke test
    can assert the scoring plumbing end to end with a stub answerer instead of
    a live model.
    """
    return Task(
        key="synthetic",
        question="What is the difference between ALPHA and BETA?",
        concepts=("ALPHA", "BETA"),
        discriminators=["ALPHA is red", "BETA is blue"],
        comprehension=["What colour is ALPHA?", "What colour is BETA?"],
        transfer="Something is blue. Is it ALPHA or BETA?",
        transfer_gold="BETA, because BETA is blue.",
    )
