"""What a route taught this session about itself, and the reset when the session changes route.

Each flag below is set after one endpoint/model rejected something (a reasoning disable, a
reasoning effort, the ``stream_options`` usage extension). It is a fact about that route, not
about the session: a ``/model`` switch, a fallback activation, a credential rotation that moves
the endpoint, or a primary restore starts the new route without the old route's verdicts, as the
Codex replay verdict already did (#61552). The primary's own verdicts come back with it.
"""
from __future__ import annotations

import copy
from typing import Any

ROUTE_REJECTION_FIELDS = (
    "_reasoning_disable_rejected", "_reasoning_floor_required", "_reasoning_effort_rejected",
    "_stream_options_unsupported",
)
# Everything ``reset_route_verdicts`` touches, for code that saves and restores a route's verdicts.
ROUTE_VERDICT_FIELDS = (*ROUTE_REJECTION_FIELDS, "_codex_reasoning_replay_enabled", "_codex_reasoning_replay_rejected")


def reset_route_verdicts(agent: Any) -> None:
    """Clear every per-route rejection verdict on ``agent`` (called whenever the route changes)."""
    for name in ROUTE_REJECTION_FIELDS:
        setattr(agent, name, False)
    from agent.turn_recovery_codex import reset_codex_reasoning_replay
    reset_codex_reasoning_replay(agent)


# The primary's verdicts ride in its ``_primary_runtime`` snapshot, so a ``/model`` switch (which
# rebuilds that snapshot) can never hand one primary's verdicts to the next. They are stamped with
# the endpoint that earned them: a pool rotation can move the primary to another base URL.
PRIMARY_VERDICTS_KEY = "route_verdicts"


def snapshot_route_verdicts(agent: Any) -> dict[str, Any]:
    """The current route's verdict values, for ``apply_route_verdicts`` to put back."""
    return {name: getattr(agent, name) for name in ROUTE_VERDICT_FIELDS if hasattr(agent, name)}


def apply_route_verdicts(agent: Any, values: dict[str, Any]) -> None:
    for name, value in values.items():
        setattr(agent, name, value)


def _endpoint(agent: Any) -> str:
    from hermes_cli.route_identity import normalize_route_base_url
    return normalize_route_base_url(getattr(agent, "base_url", "") or "")


def _stamp_primary_verdicts(agent: Any, rt: Any) -> Any:
    """Record the live verdicts in ``rt`` when the agent is on its primary right now."""
    if isinstance(rt, dict) and not getattr(agent, "_fallback_activated", False):
        rt[PRIMARY_VERDICTS_KEY] = {"base_url": _endpoint(agent), "values": snapshot_route_verdicts(agent)}
    return rt


def save_primary_route_verdicts(agent: Any) -> None:
    """Keep the primary route's verdicts while a fallback runs, so restoring the primary does not
    re-learn each one with a failed request. Only the hop that leaves the primary saves."""
    _stamp_primary_verdicts(agent, getattr(agent, "_primary_runtime", None))


def restore_primary_route_verdicts(agent: Any, rt: dict[str, Any]) -> None:
    """Put back what ``save_primary_route_verdicts`` kept in ``rt``, once the primary (and its
    credential pool entry) is reinstalled, and only on the endpoint that earned them."""
    saved = rt.get(PRIMARY_VERDICTS_KEY)
    if saved and saved.get("base_url") == _endpoint(agent):
        apply_route_verdicts(agent, saved.get("values") or {})
    rt.pop(PRIMARY_VERDICTS_KEY, None)


def primary_runtime_for_restore(agent: Any) -> Any:
    """A copy of ``agent._primary_runtime`` for a ``/model ... --once`` restore, carrying the
    primary's current verdicts (the live ones when the agent is on its primary right now)."""
    return _stamp_primary_verdicts(agent, copy.deepcopy(getattr(agent, "_primary_runtime", None)))
