"""Which stored cost column a usage write goes to.

A provider-reported (``actual``) amount is stored in ``actual_cost_usd`` and any other amount in
``estimated_cost_usd``. The two columns hold disjoint calls, so spend readers add them.
"""

from __future__ import annotations

from typing import Any, Optional


def split_cost_columns(amount: Optional[float], status: Optional[str]) -> dict[str, Optional[float]]:
    """``estimated_cost_usd`` / ``actual_cost_usd`` kwargs for one usage write: a provider-reported
    (``actual``) amount goes in ``actual_cost_usd``, any other in ``estimated_cost_usd``. The other
    column gets None, which keeps its stored value; readers sum the two columns."""
    actual = status == "actual"
    return {"estimated_cost_usd": None if actual else amount, "actual_cost_usd": amount if actual else None}


def note_reported_cost(agent: Any, status: Optional[str], amount: Optional[float]) -> None:
    """Track the provider-reported part of ``agent.session_estimated_cost_usd`` (which stays the whole
    spend) as ``agent.session_actual_cost_usd``, for writers that record a whole agent's spend at once
    (background review forks) and must split the two columns as well."""
    if status == "actual" and amount is not None:
        agent.session_actual_cost_usd = (getattr(agent, "session_actual_cost_usd", 0.0) or 0.0) + amount


def add_moa_advisor_cost(status: Optional[str], delta: Optional[float],
                         advisor_cost: float) -> tuple[Optional[float], Optional[float]]:
    """``(delta, separate_estimate)`` for a MoA turn's advisor cost. Advisors are priced by core at each
    advisor's own rate, so they are an estimate even when the aggregator's provider reports its own
    call: then they stay out of the aggregator's ``actual`` delta and are written apart as an
    estimate. Otherwise they join the aggregator's (estimated) delta, as before."""
    if status == "actual":
        return delta, advisor_cost
    return (delta or 0.0) + advisor_cost, None
