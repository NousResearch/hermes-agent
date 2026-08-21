"""
Autonomy policy gate for AUTONOMOUS_INITIATION_PATH_V1.

The policy gate is the only place that decides whether a given
objective_spec is admissible under the current autonomy envelope. It
returns a structured Verdict; the initiator consults the verdict and
either proceeds to call kanban_db.create_task or fails closed.

V1 policy:

  * autonomy must be enabled (state.is_enabled() == True)
  * kill switch must not be active (state.is_kill_switch_active() == False)
  * risk_class must be exactly "CLASS_A_AUTONOMOUS_SAFE"
  * policy_version must match the configured one
  * the spec's objective_id must not already be admitted
    (active_objective_id == None OR already == spec.objective_id)
  * no other autonomous objective is currently active
  * profile, if specified in the policy_context, must match the configured
    AUTONOMOUS_PROFILE_SCOPE (the policy_context.profile field is opt-in)

Class B and Class C are explicitly NOT admissible; the policy returns
REQUIRES_HUMAN for any class that is not A.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from . import state


# The only risk class this policy admits. Anything else is REQUIRES_HUMAN.
ADMITTED_RISK_CLASS = "CLASS_A_AUTONOMOUS_SAFE"


@dataclass(frozen=True)
class Verdict:
    """Result of a policy evaluation.

    decision is one of:
      "admit"             - the initiator may proceed to create the task
      "denied_disabled"   - autonomy is not enabled
      "denied_kill"       - the kill switch is active
      "denied_class"      - the spec's risk_class is not CLASS_A_AUTONOMOUS_SAFE
      "denied_version"    - the spec's policy_version does not match
      "denied_concurrency"- another autonomous objective is already active
      "denied_profile"    - the policy_context profile does not match the scope
    reason is a human-readable explanation suitable for an audit log.
    """

    decision: str
    reason: str


def evaluate(
    objective_spec: Dict[str, Any],
    policy_context: Optional[Dict[str, Any]] = None,
) -> Verdict:
    """Run all policy gates and return a Verdict.

    objective_spec must be a dict with at least:
      - "objective_id": str
      - "risk_class":   str (one of the documented classes)
      - "policy_version": str
      - "title":        str (used downstream by the initiator to fill
                                  the kanban title)

    policy_context is optional; when present it may contain:
      - "profile": str (the profile the autonomous run is bound to)
    """
    policy_context = policy_context or {}

    # 1. autonomy enabled?
    if not state.is_enabled():
        return Verdict(
            decision="denied_disabled",
            reason="autonomy is not enabled; call agent.autonomy.state.enable() first",
        )

    # 2. kill switch?
    if state.is_kill_switch_active():
        return Verdict(
            decision="denied_kill",
            reason="kill switch is active; autonomous initiation paused",
        )

    # 3. risk class must be A
    spec_class = objective_spec.get("risk_class")
    if spec_class != ADMITTED_RISK_CLASS:
        return Verdict(
            decision="denied_class",
            reason=(
                f"risk_class={spec_class!r} is not admissible by V1 policy; "
                f"V1 admits only {ADMITTED_RISK_CLASS!r}"
            ),
        )

    # 4. policy version must match
    spec_version = objective_spec.get("policy_version")
    if spec_version != state.get_policy_version():
        return Verdict(
            decision="denied_version",
            reason=(
                f"spec policy_version={spec_version!r} does not match "
                f"configured policy_version={state.get_policy_version()!r}"
            ),
        )

    # 5. concurrency: at most one active autonomous objective
    active = state.get_active_objective_id()
    spec_id = objective_spec.get("objective_id")
    if active is not None and active != spec_id:
        return Verdict(
            decision="denied_concurrency",
            reason=(
                f"another autonomous objective is already active "
                f"(active={active!r}); max_concurrent_autonomous_objectives=1"
            ),
        )

    # 6. profile scope (optional; only enforced if both sides specify a profile)
    spec_profile = policy_context.get("profile")
    state_profile = state.get_profile()
    if spec_profile is not None and state_profile is not None:
        if spec_profile != state_profile:
            return Verdict(
                decision="denied_profile",
                reason=(
                    f"spec profile={spec_profile!r} does not match "
                    f"configured profile={state_profile!r}"
                ),
            )

    return Verdict(
        decision="admit",
        reason="all policy gates passed",
    )
