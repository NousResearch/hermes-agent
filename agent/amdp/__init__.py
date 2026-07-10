"""AMDP (Agentic Military Decision Process) — opt-in planning layer.

An opt-in, default-OFF planning pass modeled on the Mixture-of-Agents wiring:
the commander (aggregator model) proposes distinct courses of action, the
reviewers (reference models) adversarially war-game each, a scored argmax picks
one, and the plan is injected into the turn ephemerally — the same tail-append
pattern MoA uses. It refuses to plan when the system's believed state is blind
or stale (the paper's step 1).

Fail-closed: any AMDP exception falls through to normal agent behavior; a turn
is never broken by the planner. The pure ``schemas``/``prompts``/``scoring``
modules are shared verbatim with the out-of-tree prototype in
``Documents\\hermes-moa-contrib\\amdp\\``, which is where the loop was proven
end-to-end before this integration.
"""
