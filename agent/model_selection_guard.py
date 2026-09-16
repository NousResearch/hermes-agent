"""Common managed-route guard (design §5, §12): the one seam every adapter
calls to validate a persisted routing receipt against the actual constructed
route before content is transmitted.

Pure validation — no provider clients, no credentials, no network. Adapters
(delegation, Kanban worker bootstrap, MoA) call `validate_actual_route`
immediately before their first real inference with the constructed
provider/model/endpoint/reasoning; a mismatch raises RoutingBlocked and the
call site must stop before sending task content (§4 step 7, §5: "no silent
escape through parent inheritance").
"""
from __future__ import annotations

from typing import Optional

from agent.model_selection_types import RoutingBlocked


def validate_actual_route(
    decision: dict,
    *,
    actual_provider: str,
    actual_model: str,
    actual_endpoint: Optional[str],
    actual_reasoning: Optional[str],
) -> None:
    """Raise RoutingBlocked if the actually-constructed route/reasoning does
    not match the receipted decision. Called at the last possible moment
    before content transmission (§12 "Claim/start/crash sequence" step 5).

    ``actual_endpoint``/``actual_reasoning`` are REQUIRED keyword args (no
    default) so a call site cannot omit them and silently skip that half of
    the check; pass ``None`` explicitly when the caller genuinely has no
    endpoint/reasoning concept -- the check still enforces whenever the
    receipted decision specifies one.
    """
    selected = decision["selected"]
    if actual_provider != selected["provider"] or actual_model != selected["model"]:
        raise RoutingBlocked(
            "stale_or_revoked_decision",
            f"constructed route provider={actual_provider!r} model={actual_model!r} "
            f"does not match receipted selection provider={selected['provider']!r} "
            f"model={selected['model']!r}",
        )
    if selected.get("endpoint") and actual_endpoint != selected["endpoint"]:
        raise RoutingBlocked(
            "stale_or_revoked_decision",
            f"constructed endpoint {actual_endpoint!r} does not match receipted "
            f"endpoint {selected['endpoint']!r}",
        )
    expected_reasoning = decision["requirements"].get("reasoning")
    if expected_reasoning and actual_reasoning != expected_reasoning:
        raise RoutingBlocked(
            "reasoning_unsupported",
            f"constructed reasoning {actual_reasoning!r} does not match receipted "
            f"reasoning {expected_reasoning!r}",
        )


def managed_child_kwargs(decision: dict) -> dict:
    """The exact resolved route/reasoning to hand a real child constructor
    (delegate_tool._build_child_agent, Kanban worker argv, MoA slot client).
    Disables inherited fallback: callers pass these as authoritative
    overrides, never merged with a parent/profile default (§5)."""
    selected = decision["selected"]
    return {
        "provider": selected["provider"],
        "model": selected["model"],
        "endpoint": selected.get("endpoint"),
        "reasoning_effort": decision["requirements"].get("reasoning"),
    }
