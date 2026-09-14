"""Limits, and an honest account of what each one actually does.

Every control carries its **enforcement class** as data rather than as a docstring, so
nothing downstream — the Control API, the dashboard, a customer's security review — can
present a trailing measurement as a hard ceiling. Misrepresenting a limit is the failure
mode this module exists to prevent.

The classes, in descending order of strength:

``HARD_PREEMPTIVE``
    The runtime refuses before the work happens. Compiled into the agent's runtime
    configuration and enforced by the runtime's own code paths, each verified at its call
    site (see ``docs/platform/BUDGET_ENFORCEMENT_AUDIT.md``).

``HARD_BOUNDARY``
    NOVA refuses at the tool-call boundary, through the runtime's one policy hook. Hard,
    but it can only stop work that goes through a tool call.

``SOFT_ADVISORY``
    Asks the agent to wrap up. **Not a limit.** The agent may ignore it and nothing
    terminates.

``RECORDED_ONLY``
    Carried to the runtime but nothing NOVA controls enforces it today.

``OBSERVED_ONLY``
    Measured after the fact, never enforced. Lags reality.

This module holds the **vocabulary only**. Which class a given limit falls into is a
property of the *(limit, runtime)* pair rather than of the limit alone — a control is hard
only because some runtime enforces it — so each adapter declares its own register and the
contract exposes it through :meth:`AgentRuntime.limit_facts`. A future runtime that cannot
veto a tool call would classify the same limit differently, and must be able to say so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

HARD_PREEMPTIVE = "hard_preemptive"
HARD_BOUNDARY = "hard_boundary"
SOFT_ADVISORY = "soft_advisory"
RECORDED_ONLY = "recorded_only"
OBSERVED_ONLY = "observed_only"

#: Classes that genuinely stop an agent. Anything outside this set must never be
#: described as a ceiling, a cap, or a limit in customer-facing text.
ENFORCING_CLASSES = frozenset({HARD_PREEMPTIVE, HARD_BOUNDARY})


@dataclass(frozen=True)
class LimitFact:
    """One control: what it does, who enforces it, and where that was verified."""

    key: str
    enforcement: str
    summary: str
    #: The runtime configuration key it compiles to, when it compiles to one.
    compiles_to: str = ""
    #: File and symbol where enforcement was confirmed by reading the runtime's source.
    verified_at: str = ""

    @property
    def enforced(self) -> bool:
        return self.enforcement in ENFORCING_CLASSES

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "enforcement": self.enforcement,
            "enforced": self.enforced,
            "summary": self.summary,
            "compiles_to": self.compiles_to,
            "verified_at": self.verified_at,
        }


def facts_by_enforcement(facts) -> dict[str, list[LimitFact]]:
    """Group a runtime's declared facts by enforcement class."""
    grouped: dict[str, list[LimitFact]] = {}
    for fact in facts:
        grouped.setdefault(fact.enforcement, []).append(fact)
    return grouped
