"""Policy and governance: what an agent may do, and what needs a human first.

Three pieces, deliberately separated:

* :mod:`nova.policy.model` — the tenant's declaration: business actions, the tools that
  perform them, permissions, and the baseline every worker needs.
* :mod:`nova.policy.decide` — the pure decision function. Runtime-agnostic, dependency-free,
  and shipped verbatim into the runtime so the same code decides in both places.
* :mod:`nova.policy.compile` — folds an agent's spec together with the tenant policy into
  one compiled document the enforcement point reads.

The decision is made in *one* function. The control plane explaining a decision and the
runtime enforcing it must never be two implementations that can disagree.
"""

from nova.policy.compile import CompiledPolicy, agent_digest, compile_policy
from nova.policy.decide import ALLOW, DENY, REQUIRE_APPROVAL, Decision, decide
from nova.policy.model import ActionSpec, PermissionSpec, PolicySpec

__all__ = [
    "ALLOW",
    "DENY",
    "REQUIRE_APPROVAL",
    "ActionSpec",
    "CompiledPolicy",
    "agent_digest",
    "Decision",
    "PermissionSpec",
    "PolicySpec",
    "compile_policy",
    "decide",
]
