"""Per-turn outcome telemetry — a retrospective self-observation channel.

The ``pre_llm_call`` plugin hook fires in the turn *prologue*, before the model
call, so a plugin can see what a turn is *about* but never what a turn *did*:
which provider actually answered, whether the call fell back to a backup, how
tool calls failed. Those outcomes are produced during the turn and then wiped at
the top of the next one (``restore_primary_runtime`` clears the fallback flag;
``reset_for_turn`` zeroes the guardrail counts) — so by the time a hook could
read them, they're gone.

This module captures a small, stable record of the turn's *outcome* and stashes
it on ``agent._last_turn_telemetry``; the next turn's ``pre_llm_call`` hook
receives it as ``last_turn=``. That lets a plugin reason about the agent's own
recent behavior — "last turn ran on the backup endpoint" — self-perception the
agent otherwise lacks.

**Where it is captured (this matters):** in the ``AIAgent.run_conversation``
forwarder's ``finally`` — NOT at the top of ``finalize_turn``. ``run_conversation``
has many early-return paths (interrupts, failed calls, policy blocks) that never
reach ``finalize_turn``; those are exactly the turns a self-observation plugin
most wants to see. Capturing in the forwarder's ``finally`` runs on every
exit — every return and every exception — reading the agent state the turn left
behind (fallback flag and guardrail counts are only reset at the *next* turn's
prologue, so they still describe the turn that just ended).

Design contract (deliberately minimal and stable — it is a hook API):
- **Purely additive & inert.** Nothing in core reads the record; it only feeds
  the hook kwarg. Absent a consumer it is one small dict built per turn.
- **Never raises.** Every field read defensively; capture is exception-guarded.
- **Retrospective.** It describes the turn that just finished; a consumer must
  treat it as "last turn", never "this turn".
- **Additive schema.** Fields may be ADDED in future; consumers should ``.get``.

Schema of the returned dict (all keys always present):
    has_data          bool   False only for the first-turn empty sentinel
    turn_id           str    the finished turn's id
    provider          str    provider whose runtime was active at turn end
    model             str    model whose runtime was active at turn end
    base_url          str    endpoint active at turn end (see note on cred-pool)
    was_fallback      bool   True if the turn ran on a fallback runtime
    primary_provider  str    the configured primary provider (backup-vs-home)
    primary_model     str    the configured primary model
    on_primary        bool   True if the active runtime matched the primary
    api_calls         int    API calls made this turn (best-effort from result)
    interrupted       bool   user interrupted this turn (best-effort from result)
    tool_failures     dict   {exact:int, same_tool:int, halted:bool} — counts of
                             tool calls STILL FAILING at turn end (a tool that
                             failed then succeeded is cleared by the guardrail
                             and does not appear here; this is "unresolved at
                             turn end", not a lifetime hammer count).

Note (credential-pool fallback): a fallback that reuses the same ``base_url``
with a different key sets ``was_fallback=True`` (via the flag) but leaves
``base_url``/``provider`` equal to the primary's — so those fields are "the
endpoint routed to", which for a cred-pool fallback is the same host.
"""

from __future__ import annotations

from typing import Any, Dict


def empty_telemetry() -> Dict[str, Any]:
    """The record a consumer sees before any turn has completed (first turn).

    ``has_data`` is False so a consumer can tell this sentinel from a real
    capture and decline to act on it.
    """
    return {
        "has_data": False,
        "turn_id": "",
        "provider": "",
        "model": "",
        "base_url": "",
        "was_fallback": False,
        "primary_provider": "",
        "primary_model": "",
        "on_primary": True,
        "api_calls": 0,
        "interrupted": False,
        "tool_failures": {"exact": 0, "same_tool": 0, "halted": False},
    }


def _tool_failures(agent) -> Dict[str, Any]:
    """Tool-call failures STILL UNRESOLVED at turn end (see module docstring)."""
    out = {"exact": 0, "same_tool": 0, "halted": False}
    try:
        ctrl = getattr(agent, "_tool_guardrails", None)
        if ctrl is None:
            return out
        exact = getattr(ctrl, "_exact_failure_counts", {}) or {}
        same = getattr(ctrl, "_same_tool_failure_counts", {}) or {}
        out["exact"] = len(exact)
        out["same_tool"] = sum(int(v) for v in same.values())
        out["halted"] = getattr(ctrl, "halt_decision", None) is not None
    except Exception:
        pass
    return out


def capture_turn_telemetry(agent, *, result: Any = None) -> Dict[str, Any]:
    """Build the outcome record for the turn that just finished. Never raises.

    Call from the ``run_conversation`` forwarder's ``finally`` so it runs on
    every exit path. Core fields (fallback/provider/tool-failures) are read from
    the agent, which still reflects the just-finished turn at this point; the
    ``result`` dict (may be ``None`` on an exception) supplies best-effort turn
    metadata (``api_calls``, ``interrupted``).
    """
    res = result if isinstance(result, dict) else {}
    try:
        primary = getattr(agent, "_primary_runtime", None) or {}
        provider = str(getattr(agent, "provider", "") or "")
        model = str(getattr(agent, "model", "") or "")
        base_url = str(getattr(agent, "base_url", "") or "")
        primary_provider = str(primary.get("provider", "") or "")
        primary_model = str(primary.get("model", "") or "")

        # `_fallback_activated` is authoritative and still set here (the next
        # turn's restore clears it). The runtime comparison is a cross-check
        # covering divergence the flag might miss.
        flag_fallback = bool(getattr(agent, "_fallback_activated", False))
        primary_base = str(primary.get("base_url", "") or "")
        runtime_diverged = bool(primary_base) and base_url != primary_base
        on_primary = (not flag_fallback) and (not runtime_diverged)

        return {
            "has_data": True,
            "turn_id": str(getattr(agent, "_current_turn_id", "") or ""),
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "was_fallback": flag_fallback or runtime_diverged,
            "primary_provider": primary_provider,
            "primary_model": primary_model,
            "on_primary": on_primary,
            "api_calls": int(res.get("api_calls", 0) or 0),
            "interrupted": bool(res.get("interrupted", False)),
            "tool_failures": _tool_failures(agent),
        }
    except Exception:
        # A broken capture must never break a turn: fall back to the sentinel,
        # marked has_data=False so no consumer acts on an unreliable record.
        return empty_telemetry()
